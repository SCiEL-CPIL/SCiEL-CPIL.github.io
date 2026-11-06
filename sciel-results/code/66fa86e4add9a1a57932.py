"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    import math
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    def bron_kerbosch_bitmask(n, adj_mask, cliques):
        """Bron-Kerbosch with pivoting, using bitmasks. Collects cliques of size >= 2."""
        def bk(R, P, X):
            if P == 0 and X == 0:
                if R.bit_count() >= 2:
                    cliques.append(R)
                return
            PX = P | X
            if PX == 0:
                return
            best_v = -1
            best_count = -1
            temp = PX
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                cnt = (P & adj_mask[v]).bit_count()
                if cnt > best_count:
                    best_count = cnt
                    best_v = v
                temp ^= lsb
            if best_v >= 0:
                candidates = P & ~adj_mask[best_v]
            else:
                candidates = P
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                bk(R | lsb, P & adj_mask[v], X & adj_mask[v])
                P ^= lsb
                X |= lsb
                temp ^= lsb

        bk(0, (1 << n) - 1, 0)

    def build_graph(n, p, seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    def greedy_transversal(cliques_masks, n):
        """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
        if not cliques_masks:
            return 0
        num_cliques = len(cliques_masks)
        vertex_cliques = [[] for _ in range(n)]
        for ci, cm in enumerate(cliques_masks):
            temp = cm
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                vertex_cliques[v].append(ci)
                temp ^= lsb
        uncovered = [True] * num_cliques
        selected = 0
        uncovered_count = num_cliques
        while uncovered_count > 0:
            best_v = -1
            best_score = -1
            for v in range(n):
                if selected & (1 << v):
                    continue
                score = 0
                for ci in vertex_cliques[v]:
                    if uncovered[ci]:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v < 0 or best_score <= 0:
                break
            selected |= (1 << best_v)
            for ci in vertex_cliques[best_v]:
                if uncovered[ci]:
                    uncovered[ci] = False
                    uncovered_count -= 1
        return selected

    def local_search_reduce(selected_mask, cliques_masks, n, max_iters=200):
        """Try to remove vertices; if it breaks coverage, stop."""
        if not cliques_masks:
            return selected_mask
        num_cliques = len(cliques_masks)
        cover_count = [0] * num_cliques
        for ci, cm in enumerate(cliques_masks):
            cnt = 0
            temp = cm & selected_mask
            while temp:
                temp &= temp - 1
                cnt += 1
            cover_count[ci] = cnt
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if not (selected_mask & (1 << v)):
                    continue
                v_bit = 1 << v
                removable = True
                for ci, cm in enumerate(cliques_masks):
                    if cover_count[ci] == 1 and (cm & v_bit):
                        removable = False
                        break
                if removable:
                    selected_mask ^= v_bit
                    for ci, cm in enumerate(cliques_masks):
                        if cm & v_bit:
                            cover_count[ci] -= 1
                    improved = True
                    break
        return selected_mask

    def solve_milp(n, cliques_masks, deadline, initial_mask=None):
        """Solve the clique transversal as a MILP using scipy.optimize.milp."""
        num_cliques = len(cliques_masks)
        if num_cliques == 0:
            return 0

        # Build constraint matrix: each row is a clique, entries are 1 if vertex is in clique
        # A * x >= 1 for each clique
        # We use: -A * x <= -1
        A = np.zeros((num_cliques, n), dtype=np.float64)
        for ci, cm in enumerate(cliques_masks):
            temp = cm
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                A[ci, v] = 1.0
                temp ^= lsb

        # Objective: minimize sum(x)
        c = np.ones(n, dtype=np.float64)

        # Constraints: A @ x >= 1  =>  -A @ x <= -1
        lb = -np.ones(num_cliques)
        ub = np.full(num_cliques, np.inf)
        constraints = LinearConstraint(-A, lb, ub)

        # Variable bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

        # Integrality: all variables integer
        integrality = np.ones(n, dtype=int)

        # Initial hint if available
        x0 = None
        if initial_mask is not None:
            x0 = np.zeros(n)
            for v in range(n):
                if initial_mask & (1 << v):
                    x0[v] = 1.0

        try:
            res = milp(c=c, constraints=constraints, integrality=integrality,
                       bounds=bounds, options={"time_limit": max(1, deadline - time.time())})
            if res.status == 0 and res.x is not None:
                mask = 0
                for v in range(n):
                    if res.x[v] > 0.5:
                        mask |= (1 << v)
                return mask
        except Exception:
            pass
        return None

    def solve_instance(n, p, seed, deadline):
        adj_mask = build_graph(n, p, seed)
        cliques = []
        bron_kerbosch_bitmask(n, adj_mask, cliques)

        if not cliques:
            return []

        # Get a good initial solution via greedy + local search
        greedy_mask = greedy_transversal(cliques, n)
        greedy_mask = local_search_reduce(greedy_mask, cliques, n)
        greedy_size = greedy_mask.bit_count()

        # Try MILP
        best_mask = greedy_mask
        best_size = greedy_size

        if time.time() < deadline - 5:
            milp_mask = solve_milp(n, cliques, deadline, initial_mask=greedy_mask)
            if milp_mask is not None:
                milp_mask = local_search_reduce(milp_mask, cliques, n)
                milp_size = milp_mask.bit_count()
                if milp_size < best_size:
                    best_size = milp_size
                    best_mask = milp_mask

        # Final local search polish
        best_mask = local_search_reduce(best_mask, cliques, n)

        return [i for i in range(n) if best_mask & (1 << i)]

    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            out.append(list(range(n)))
            continue
        result = solve_instance(n, p, seed, deadline)
        out.append(result)

    return out
# EVOLVE-BLOCK-END
