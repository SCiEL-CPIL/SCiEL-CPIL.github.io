"""Strategy: Reframe as Minimum Vertex Cover on the line graph G' (clique hitting set), solve via LP relaxation + branch-and-bound with strong LP-based pruning, plus iterative improvement."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def bron_kerbosch_maximal_cliques(n, adj):
        max_cliques = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    max_cliques.append(frozenset(R))
                return
            if not P:
                return

            pivot_candidates = list(P) + list(X)
            if not pivot_candidates:
                return
            best_pivot = None
            max_common = -1
            for u in pivot_candidates:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_pivot = u

            if best_pivot is not None:
                candidates = [v for v in P if v not in adj[best_pivot]]
            else:
                candidates = list(P)

            P_set = set(P)
            X_set = set(X)

            for v in candidates:
                new_R = R | {v}
                new_P = P_set & adj[v]
                new_X = X_set & adj[v]
                bk(new_R, new_P, new_X)
                P_set.discard(v)
                X_set.add(v)

        bk(frozenset(), set(range(n)), set())
        return max_cliques

    def solve_mvc_lp_branch(edges, n, time_budget, seed=42):
        """
        Solve Minimum Vertex Cover via LP relaxation + branch-and-bound.
        Variables x_v in [0,1], minimize sum(x_v) s.t. x_u + x_v >= 1 for each edge (u,v).
        Uses a simple LP solver via scipy if available, otherwise falls back to greedy.
        """
        try:
            from scipy.optimize import linprog
            has_scipy = True
        except ImportError:
            has_scipy = False

        if not edges:
            return set()

        # Build constraint matrix for LP
        m = len(edges)
        # c = [1]*n
        # A_ub: for each edge (u,v), -x_u - x_v <= -1  =>  x_u + x_v >= 1
        # Actually linprog minimizes c^T x s.t. A_ub x <= b_ub, A_eq x = b_eq, bounds
        # Constraint: x_u + x_v >= 1  =>  -x_u - x_v <= -1
        A_ub = []
        b_ub = []
        for u, v in edges:
            row = [0.0] * n
            row[u] = -1.0
            row[v] = -1.0
            A_ub.append(row)
            b_ub.append(-1.0)

        c = [1.0] * n
        bounds = [(0, 1)] * n

        def lp_solve(A_ub_list, b_ub_list, c_list, bounds_list, n_vars):
            """Solve LP using scipy linprog."""
            if not has_scipy:
                return None, None
            try:
                res = linprog(c_list, A_ub=A_ub_list, b_ub=b_ub_list, bounds=bounds_list, method='highs')
                if res.success:
                    return res.x, res.fun
                else:
                    return None, None
            except Exception:
                return None, None

        def round_lp(x):
            """Round LP solution to integral vertex cover."""
            if x is None:
                return None
            # Simple rounding: x_v >= 0.5 -> include
            selected = set()
            for v in range(n):
                if x[v] >= 0.5:
                    selected.add(v)
            # Repair: ensure all edges are covered
            for u, v in edges:
                if u not in selected and v not in selected:
                    # Add the one with higher LP value
                    if x[u] >= x[v]:
                        selected.add(u)
                    else:
                        selected.add(v)
            return selected

        # Initial LP solve
        x_lp, lp_val = lp_solve(A_ub, b_ub, c, bounds, n)
        if x_lp is None:
            # Fallback to greedy
            return greedy_mvc(edges, n)

        best_solution = round_lp(x_lp)
        if best_solution is None:
            best_solution = set(range(n))
        best_size = len(best_solution)

        # Branch-and-bound
        # We branch on fractional variables
        # State: (A_ub, b_ub, c, bounds, fixed_vars dict)
        # To keep it simple, we'll do a depth-first search with LP pruning

        def greedy_mvc_local(edge_list, n_local):
            """Greedy MVC for a set of edges."""
            if not edge_list:
                return set()
            uncovered = set(range(len(edge_list)))
            selected = set()
            # Compute degree for each vertex
            deg = defaultdict(int)
            for u, v in edge_list:
                deg[u] += 1
                deg[v] += 1
            while uncovered:
                best_v = -1
                best_gain = 0
                for v in range(n_local):
                    if v in selected:
                        continue
                    gain = 0
                    for ei in uncovered:
                        u, w = edge_list[ei]
                        if u == v or w == v:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                selected.add(best_v)
                for ei in list(uncovered):
                    u, w = edge_list[ei]
                    if u == best_v or w == best_v:
                        uncovered.discard(ei)
            return selected

        # Branch and bound using recursive LP
        # To avoid deep recursion, we'll use a stack
        # Each node: (A_ub, b_ub, bounds, fixed)
        # fixed: dict var -> value (0 or 1)

        # Precompute: for each variable, which constraints it appears in
        # This is complex, so we'll use a simpler approach:
        # Branch on the most fractional variable, solve LP with that variable fixed.

        # For efficiency, we'll limit the number of nodes
        max_nodes = 200
        nodes_explored = 0
        t_start = time.time()

        # Stack of (A_ub, b_ub, bounds, fixed_vars, lp_val)
        # We'll modify bounds to fix variables

        def fix_var_in_bounds(bounds, var, val):
            """Return new bounds with var fixed to val."""
            new_bounds = list(bounds)
            new_bounds[var] = (val, val)
            return new_bounds

        # We need to track which variables are fixed and their values
        # To avoid copying large arrays, we'll use a different approach:
        # Keep a list of fixed assignments and rebuild bounds each time.
        # For n <= 40, this is fine.

        # Actually, let's use a simpler B&B:
        # 1. Solve LP
        # 2. If integral, update best
        # 3. Else, branch on most fractional variable
        # 4. Recurse with x_v = 0 and x_v = 1

        # We'll implement this with a stack to avoid recursion depth issues

        # Stack items: (bounds, fixed_list)
        # fixed_list: list of (var, val)
        # We rebuild A_ub, b_ub, bounds from scratch each time (slow but correct for n<=40)

        # Optimization: if a variable is fixed to 0, remove it from constraints
        # if a variable is fixed to 1, remove constraints it satisfies

        # For simplicity and correctness, we'll just modify bounds and let LP handle it.

        # Let's implement a cleaner B&B:

        def branch_and_bound(A_ub_init, b_ub_init, c_init, bounds_init, n_vars, time_limit):
            best_sol = None
            best_val = float('inf')
            stack = [(bounds_init, [])]
            nodes = 0

            while stack and nodes < max_nodes and time.time() - t_start < time_limit:
                bounds, fixed = stack.pop()
                nodes += 1

                # Solve LP
                x, val = lp_solve(A_ub_init, b_ub_init, c_init, bounds, n_vars)
                if x is None:
                    continue

                # Check if integral
                integral = all(abs(x[i] - round(x[i])) < 1e-6 for i in range(n_vars))
                if integral:
                    sol = set(i for i in range(n_vars) if x[i] > 0.5)
                    # Verify it's a valid vertex cover
                    valid = True
                    for u, v in edges:
                        if u not in sol and v not in sol:
                            valid = False
                            break
                    if valid and len(sol) < best_val:
                        best_val = len(sol)
                        best_sol = sol
                    continue

                # If LP value >= best_val, prune
                if val >= best_val - 1e-6:
                    continue

                # Find most fractional variable
                max_frac = 0
                frac_var = -1
                for i in range(n_vars):
                    frac = min(x[i], 1 - x[i])
                    if frac > max_frac:
                        max_frac = frac
                        frac_var = i

                if frac_var == -1:
                    continue

                # Branch: x_frac_var = 0 and x_frac_var = 1
                # Order: try the one closer to 1 first (more likely to be in cover)
                if x[frac_var] >= 0.5:
                    order = [1, 0]
                else:
                    order = [0, 1]

                for val_fix in order:
                    new_bounds = list(bounds)
                    new_bounds[frac_var] = (val_fix, val_fix)
                    new_fixed = fixed + [(frac_var, val_fix)]
                    stack.append((new_bounds, new_fixed))

            return best_sol, best_val

        # Run B&B with time limit
        remaining = time_budget - (time.time() - start_time)
        bb_time = max(0.5, remaining - 1.0)
        bb_sol, bb_val = branch_and_bound(A_ub, b_ub, c, bounds, n, bb_time)

        if bb_sol is not None and len(bb_sol) < best_size:
            best_solution = bb_sol
            best_size = len(bb_sol)

        # Final repair
        for u, v in edges:
            if u not in best_solution and v not in best_solution:
                best_solution.add(u)

        return best_solution

    def greedy_mvc(edges_list, n):
        """Greedy minimum vertex cover."""
        if not edges_list:
            return set()
        uncovered = set(range(len(edges_list)))
        selected = set()
        while uncovered:
            best_v = -1
            best_gain = 0
            for v in range(n):
                if v in selected:
                    continue
                gain = 0
                for ei in uncovered:
                    u, w = edges_list[ei]
                    if u == v or w == v:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_v = v
            if best_v == -1 or best_gain == 0:
                break
            selected.add(best_v)
            for ei in list(uncovered):
                u, w = edges_list[ei]
                if u == best_v or w == best_v:
                    uncovered.discard(ei)
        return selected

    def solve_instance(n, adj, cliques, time_budget):
        """
        Solve minimum hitting set for cliques (all size >= 2).
        Key insight: hitting all maximal cliques of size >= 2 is equivalent to
        finding a minimum vertex cover in the line graph of G, where each edge
        in G corresponds to a vertex in G', and two vertices in G' are adjacent
        if the corresponding edges in G share a vertex.
        
        However, this is not quite right. Let's think again.
        
        Actually, the problem is: given a set of cliques (each of size >= 2),
        find the minimum set of vertices that hits every clique.
        
        This is the Hitting Set problem, which is NP-hard.
        
        But note: if we consider the hypergraph where each clique is a hyperedge,
        we want the minimum hitting set.
        
        For cliques of size 2 (edges), this is Minimum Vertex Cover.
        For larger cliques, it's more complex.
        
        However, in a random graph with p < 0.5, most maximal cliques are of size 2 or 3.
        We can solve this as a general hitting set problem using LP + B&B.
        
        Let's formulate as ILP:
        Minimize sum(x_v) for v in V
        Subject to: sum(x_v for v in C) >= 1 for each clique C
        x_v in {0, 1}
        
        LP relaxation: x_v in [0, 1]
        """
        if not cliques:
            return set()

        m = len(cliques)
        # Build constraint matrix
        # A: m x n, A[i][v] = 1 if v in cliques[i]
        # b: [1]*m
        # c: [1]*n

        # For LP, we need A_ub x <= b_ub
        # Constraint: sum(x_v for v in C) >= 1  =>  -sum(x_v for v in C) <= -1
        A_ub = []
        b_ub = []
        for C in cliques:
            row = [0.0] * n
            for v in C:
                row[v] = -1.0
            A_ub.append(row)
            b_ub.append(-1.0)

        c = [1.0] * n
        bounds = [(0, 1)] * n

        try:
            from scipy.optimize import linprog
            has_scipy = True
        except ImportError:
            has_scipy = False

        def lp_solve(A_ub_list, b_ub_list, c_list, bounds_list, n_vars):
            if not has_scipy:
                return None, None
            try:
                res = linprog(c_list, A_ub=A_ub_list, b_ub=b_ub_list, bounds=bounds_list, method='highs')
                if res.success:
                    return res.x, res.fun
                else:
                    return None, None
            except Exception:
                return None, None

        def round_lp(x):
            if x is None:
                return None
            selected = set()
            for v in range(n):
                if x[v] >= 0.5:
                    selected.add(v)
            # Repair
            for C in cliques:
                if not any(v in selected for v in C):
                    # Add the vertex in C with highest LP value
                    best_v = max(C, key=lambda v: x[v])
                    selected.add(best_v)
            return selected

        # Initial LP
        x_lp, lp_val = lp_solve(A_ub, b_ub, c, bounds, n)
        if x_lp is None:
            # Fallback: greedy hitting set
            selected = set()
            uncovered = set(range(m))
            while uncovered:
                best_v = -1
                best_gain = 0
                for v in range(n):
                    if v in selected:
                        continue
                    gain = 0
                    for ci in uncovered:
                        if v in cliques[ci]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                if best_v == -1 or best_gain == 0:
                    break
                selected.add(best_v)
                for ci in list(uncovered):
                    if best_v in cliques[ci]:
                        uncovered.discard(ci)
            return selected

        best_solution = round_lp(x_lp)
        if best_solution is None:
            best_solution = set(range(n))
        best_size = len(best_solution)

        # Branch and bound
        max_nodes = 500
        t_start = time.time()
        remaining = time_budget - (time.time() - start_time)
        bb_time = max(0.5, remaining - 1.0)

        stack = [(bounds, [])]
        nodes = 0

        while stack and nodes < max_nodes and time.time() - t_start < bb_time:
            bnds, fixed = stack.pop()
            nodes += 1

            x, val = lp_solve(A_ub, b_ub, c, bnds, n)
            if x is None:
                continue

            integral = all(abs(x[i] - round(x[i])) < 1e-6 for i in range(n))
            if integral:
                sol = set(i for i in range(n) if x[i] > 0.5)
                # Verify
                valid = True
                for C in cliques:
                    if not any(v in sol for v in C):
                        valid = False
                        break
                if valid and len(sol) < best_size:
                    best_size = len(sol)
                    best_solution = sol
                continue

            if val >= best_size - 1e-6:
                continue

            # Find most fractional
            max_frac = 0
            frac_var = -1
            for i in range(n):
                frac = min(x[i], 1 - x[i])
                if frac > max_frac:
                    max_frac = frac
                    frac_var = i

            if frac_var == -1:
                continue

            if x[frac_var] >= 0.5:
                order = [1, 0]
            else:
                order = [0, 1]

            for val_fix in order:
                new_bounds = list(bnds)
                new_bounds[frac_var] = (val_fix, val_fix)
                stack.append((new_bounds, fixed + [(frac_var, val_fix)]))

        # Final repair
        for C in cliques:
            if not any(v in best_solution for v in C):
                best_solution.add(max(C, key=lambda v: x_lp[v] if x_lp else 0))

        return best_solution

    results = []

    for idx, (n, p, seed) in enumerate(((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203))):
        if time.time() > deadline - 15:
            results.append(list(range(n)))
            continue

        edges = random_graph(n, p, seed)
        adj = build_adj(n, edges)
        cliques = bron_kerbosch_maximal_cliques(n, adj)
        cliques = [c for c in cliques if len(c) >= 2]

        if not cliques:
            results.append([])
            continue

        elapsed = time.time() - start_time
        remaining = deadline - 10 - elapsed
        per_instance = remaining / max(1, 3 - idx)
        per_instance = max(2.0, min(per_instance, remaining))

        sol = solve_instance(n, adj, cliques, per_instance)
        results.append(sorted(sol))

    return results
# EVOLVE-BLOCK-END
