"""Combined program: bitmask K4-free local search with branch-and-bound evaluator."""
import time
import random
import sys

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the max triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    N = 18
    FULL_MASK = (1 << N) - 1

    # ---------- helpers ----------

    def adj_to_mask(adj):
        """adj is list of int masks; return same (already masks)."""
        return adj

    def make_edges_from_mask(adj):
        edges = []
        for i in range(N):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                m ^= lsb
        return edges

    def is_k4_free(adj):
        """Check K4-free using bitmasks."""
        for i in range(N):
            mi = adj[i]
            m = mi
            while m:
                lsb = m & -m
                m ^= lsb
                j = lsb.bit_length() - 1
                if j <= i:
                    continue
                # common neighbors of i and j
                common = mi & adj[j]
                common &= ~(1 << i)
                common &= ~(1 << j)
                # check if common contains an edge
                c = common
                while c:
                    lsb2 = c & -c
                    c ^= lsb2
                    k = lsb2.bit_length() - 1
                    if adj[k] & common & ~(1 << k):
                        return False
        return True

    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        c = common
        while c:
            lsb = c & -c
            c ^= lsb
            w = lsb.bit_length() - 1
            if adj[w] & common & ~(1 << w):
                return False
        return True

    def can_remove_edge(adj, u, v):
        """Removing an edge always preserves K4-free."""
        return True

    # ---------- exact branch-and-bound for max triangle-free induced subgraph ----------

    def max_triangle_free_size(adj):
        """Exact B&B: find largest subset with no triangle in induced subgraph."""
        best = [0]

        # Precompute: for each vertex v, triangle_mask[v] = set of vertices w
        # such that v and w are adjacent and share a common neighbor (i.e., adding w
        # to a set containing v would create a triangle with that common neighbor).
        # More precisely: when we include v in the current set, any candidate w that is
        # adjacent to v AND adjacent to some u already in the set forms a triangle.
        # We handle this dynamically.

        def dfs(current, candidates):
            if not candidates:
                if current.bit_count() > best[0]:
                    best[0] = current.bit_count()
                return

            # Upper bound
            if current.bit_count() + candidates.bit_count() <= best[0]:
                return

            # Branching heuristic: pick vertex with max degree in candidates
            # to prune more
            c = candidates
            best_v = -1
            best_deg = -1
            while c:
                lsb = c & -c
                c ^= lsb
                v = lsb.bit_length() - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            v = best_v
            v_mask = 1 << v

            # Branch 1: exclude v
            dfs(current, candidates & ~v_mask)

            # Branch 2: include v
            # Check if v can be added without creating a triangle with current
            # v forms triangle with u1, u2 in current if u1-u2 edge and both adj to v
            neighbors_in_current = adj[v] & current
            # Check if neighbors_in_current contains an edge
            check = neighbors_in_current
            valid = True
            while check:
                lsb = check & -check
                check ^= lsb
                u = lsb.bit_length() - 1
                if adj[u] & neighbors_in_current & ~(1 << u):
                    valid = False
                    break

            if valid:
                new_current = current | v_mask
                # New forbidden: any w in candidates that is adjacent to v and
                # adjacent to some u in neighbors_in_current (would form triangle)
                forbidden = 0
                check_u = neighbors_in_current
                while check_u:
                    lsb_u = check_u & -check_u
                    check_u ^= lsb_u
                    u = lsb_u.bit_length() - 1
                    forbidden |= adj[u]
                forbidden &= adj[v]
                new_candidates = (candidates & ~v_mask) & ~forbidden
                dfs(new_current, new_candidates)

        dfs(0, FULL_MASK)
        return best[0]

    # ---------- initial constructions ----------

    def make_turan3_adj():
        """Complete 3-partite graph with parts of size 6."""
        adj = [0] * N
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    def make_c9_union_adj():
        """Two disjoint C9 cycles."""
        adj = [0] * N
        for i in range(9):
            u = i
            v = (i + 1) % 9
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for i in range(9):
            u = 9 + i
            v = 9 + ((i + 1) % 9)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    # ---------- local search ----------

    def local_search(adj, current_score, deadline, max_iter=2000):
        """Hill-climbing on edge flips, maintaining K4-free."""
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            # Try all possible edge flips in random order
            pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            random.shuffle(pairs)
            for u, v in pairs:
                if time.time() >= deadline:
                    break
                old = (adj[u] >> v) & 1
                if old == 0:
                    # Try adding edge
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if is_k4_free(adj):
                            s = max_triangle_free_size(adj)
                            if s < current_score:
                                current_score = s
                                improved = True
                            else:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                else:
                    # Try removing edge
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    s = max_triangle_free_size(adj)
                    if s < current_score:
                        current_score = s
                        improved = True
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return current_score

    def simulated_annealing(adj, current_score, deadline, start_temp=10.0, cool=0.95):
        """SA on edge flips."""
        temp = start_temp
        best_score = current_score
        best_adj = adj[:]
        iters = 0
        while time.time() < deadline and temp > 0.1:
            iters += 1
            u, v = random.sample(range(N), 2)
            old = (adj[u] >> v) & 1
            if old == 0:
                if not can_add_edge(adj, u, v):
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)

            if not is_k4_free(adj):
                # revert
                if old == 0:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue

            s = max_triangle_free_size(adj)
            delta = s - current_score
            if delta <= 0 or random.random() < pow(2.718281828, -delta / temp):
                current_score = s
                if s < best_score:
                    best_score = s
                    best_adj = adj[:]
            else:
                # revert
                if old == 0:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            temp *= cool
        return best_score, best_adj

    # ---------- main ----------

    random.seed(42)

    # Start with Turan3
    best_adj = make_turan3_adj()
    best_score = max_triangle_free_size(best_adj)

    # Local search from Turan3
    best_score = local_search(best_adj, best_score, deadline)

    # Try C9 union as another starting point
    c9_adj = make_c9_union_adj()
    c9_score = max_triangle_free_size(c9_adj)
    if c9_score < best_score:
        best_score = c9_score
        best_adj = c9_adj
    best_score = local_search(best_adj, best_score, deadline)

    # Random restarts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate random K4-free graph
        # Strategy: start from Turan3, remove random edges
        adj = make_turan3_adj()
        num_removals = random.randint(5, 30)
        for _ in range(num_removals):
            i, j = random.sample(range(N), 2)
            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)

        if not is_k4_free(adj):
            continue

        s = max_triangle_free_size(adj)
        if s < best_score:
            best_score = s
            best_adj = adj[:]
            # Local search from here
            best_score = local_search(best_adj, best_score, deadline)

        # Also try from C9
        if attempts % 3 == 0:
            adj2 = make_c9_union_adj()
            # Add some random edges
            num_adds = random.randint(3, 15)
            for _ in range(num_adds):
                i, j = random.sample(range(N), 2)
                if not ((adj2[i] >> j) & 1):
                    if can_add_edge(adj2, i, j):
                        adj2[i] |= (1 << j)
                        adj2[j] |= (1 << i)
            if is_k4_free(adj2):
                s2 = max_triangle_free_size(adj2)
                if s2 < best_score:
                    best_score = s2
                    best_adj = adj2[:]
                    best_score = local_search(best_adj, best_score, deadline)

        # SA from current best
        if attempts % 5 == 0 and time.time() < deadline - 10:
            sa_adj = best_adj[:]
            sa_score, sa_best = simulated_annealing(sa_adj, best_score, min(deadline, time.time() + 10))
            if sa_score < best_score:
                best_score = sa_score
                best_adj = sa_best

    return make_edges_from_mask(best_adj)
