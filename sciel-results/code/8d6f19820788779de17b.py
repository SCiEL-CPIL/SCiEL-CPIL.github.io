import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118

    def popcount(x):
        return bin(x).count('1')

    def build_adj_bitset(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free(adj, n):
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                if common:
                    return False
        return True

    def greedy_maximal_independent_set(candidates, adj):
        """Return a maximal independent set (as bitset) from candidates."""
        result = 0
        rem = candidates
        while rem:
            # Pick vertex with min degree in rem
            best_v = -1
            best_deg = float('inf')
            temp = rem
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & rem)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            if best_v < 0:
                break
            bit_v = 1 << best_v
            result |= bit_v
            rem &= ~adj[best_v] & ~bit_v
        return result

    def max_independent_set_size_upper_bound(candidates, adj, n, depth=4):
        """Greedy coloring to upper-bound the independence number of the subgraph on candidates."""
        remaining = candidates
        total = 0
        for _ in range(depth):
            if not remaining:
                break
            color_class = greedy_maximal_independent_set(remaining, adj)
            if not color_class:
                break
            total += popcount(color_class)
            remaining &= ~color_class
        if remaining:
            total += popcount(remaining)
        return total

    def has_independent_set_size(adj, n, k, time_limit=3.0):
        """Returns True if there exists an independent set of size k."""
        start_t = time.time()
        found = [False]

        def dfs(candidates, current_size):
            if found[0] or time.time() - start_t > time_limit:
                return
            if current_size >= k:
                found[0] = True
                return
            if not candidates:
                return

            # Upper bound via greedy coloring
            ub = current_size + max_independent_set_size_upper_bound(candidates, adj, n, depth=3)
            if ub < k:
                return

            # Pick vertex: min degree in candidates
            best_v = -1
            best_deg = float('inf')
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & candidates)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v

            v = best_v
            bit_v = 1 << v
            rest = candidates & ~bit_v
            non_neighbors = rest & ~adj[v]

            # Include v
            dfs(non_neighbors, current_size + 1)
            if found[0]:
                return
            # Exclude v
            dfs(rest, current_size)

        dfs((1 << n) - 1, 0)
        return found[0]

    def independent_set_size(adj, n):
        """Compute the independence number via branch and bound."""
        best = [0]
        start_t = time.time()

        def dfs(candidates, current_size, current_set):
            if time.time() - start_t > 5.0:
                return
            if current_size > best[0]:
                best[0] = current_size
            if not candidates:
                return
            # Upper bound
            ub = current_size + max_independent_set_size_upper_bound(candidates, adj, n, depth=3)
            if ub <= best[0]:
                return
            # Pick vertex
            best_v = -1
            best_deg = float('inf')
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & candidates)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            v = best_v
            bit_v = 1 << v
            rest = candidates & ~bit_v
            non_neighbors = rest & ~adj[v]
            dfs(non_neighbors, current_size + 1, current_set | bit_v)
            dfs(rest, current_size, current_set)

        dfs((1 << n) - 1, 0, 0)
        return best[0]

    def find_safe_edge(adj, n):
        """Find a non-edge (i,j) such that i and j have no common neighbor (safe to add)."""
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    if not (adj[i] & adj[j]):
                        return (i, j)
        return None

    def add_safe_edges_to_reduce_alpha(adj, n, target_alpha, max_iter=200):
        """Add safe edges to reduce independence number to at most target_alpha."""
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            if not has_independent_set_size(adj, n, target_alpha + 1, time_limit=1.5):
                break
            # Find a safe edge that helps
            safe_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if not (adj[i] & adj[j]):
                            safe_edges.append((i, j))
            random.shuffle(safe_edges)
            improved = False
            for i, j in safe_edges[:30]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                if not has_independent_set_size(adj, n, target_alpha + 1, time_limit=1.5):
                    improved = True
                    break
                else:
                    # Revert
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
            if not improved:
                break
        return adj

    def andrasfai_graph(k):
        """Andrásfai graph A_k on 3k vertices. Triangle-free, alpha = k for k>=2."""
        n = 3 * k
        adj = [0] * n
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    if (b - a) % 3 == 1:
                        for j in range(k):
                            if (j - i) % k in (0, 1):
                                u = a * k + i
                                v = b * k + j
                                if u != v:
                                    adj[u] |= (1 << v)
                                    adj[v] |= (1 << u)
        return n, adj

    def clebsch_derived_graph(n_target):
        """Generate a triangle-free graph based on Clebsch graph structure extended."""
        # Clebsch graph: 16 vertices, triangle-free, alpha=5
        # We use it as a seed and extend
        adj = [0] * n_target
        # Start with a random triangle-free graph
        random.seed(42)
        for i in range(n_target):
            for j in range(i + 1, n_target):
                if random.random() < 0.35:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Remove triangles
        for _ in range(2000):
            found = False
            for i in range(n_target):
                ni = adj[i]
                while ni:
                    lsb1 = ni & (-ni)
                    j = lsb1.bit_length() - 1
                    ni ^= lsb1
                    common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                    if common:
                        lsb2 = common & (-common)
                        k = lsb2.bit_length() - 1
                        # Remove edge (i,j)
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return adj

    def extend_vertex(adj, n, target_n):
        """Try to extend graph from n to target_n vertices by adding vertices connected to independent sets."""
        if target_n <= n:
            return adj
        new_n = target_n
        new_adj = [0] * new_n
        for i in range(n):
            new_adj[i] = adj[i]

        # Add new vertices one at a time
        for v_new in range(n, new_n):
            if time.time() - start_time > deadline - 3:
                return None
            # Find an independent set S in the current graph (0..v_new-1)
            # Connect v_new to all vertices in S
            # This preserves triangle-freeness if S is independent
            # The new vertex's neighborhood is S, which is independent
            # For the new vertex to not create triangles with existing edges,
            # we need S to be independent (no edges within S)
            # Also, for any u in S, v_new - u - w triangle: w must be neighbor of u and v_new
            # v_new is only connected to S, so w must be in S. Since S is independent, no w in S adjacent to u.
            # So triangle-freeness is preserved.

            # Find a maximal independent set in the current graph
            S = greedy_maximal_independent_set((1 << v_new) - 1, new_adj)
            if not S:
                return None
            # Connect v_new to S
            for u in range(v_new):
                if S & (1 << u):
                    new_adj[v_new] |= (1 << u)
                    new_adj[u] |= (1 << v_new)

            # Check triangle-freeness
            if not is_triangle_free(new_adj, v_new + 1):
                return None

        return new_adj

    def local_search_reduce_alpha(adj, n, target_alpha, max_iter=100):
        """Local search: add safe edges to reduce alpha to target_alpha."""
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            if not has_independent_set_size(adj, n, target_alpha + 1, time_limit=1.5):
                break
            # Find all safe edges
            safe_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if not (adj[i] & adj[j]):
                            safe_edges.append((i, j))
            random.shuffle(safe_edges)
            improved = False
            for i, j in safe_edges[:40]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                if not has_independent_set_size(adj, n, target_alpha + 1, time_limit=1.5):
                    improved = True
                    break
                else:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
            if not improved:
                break
        return adj

    # --- Main search ---
    best_n = 0
    best_adj = None

    # Strategy 1: Andrásfai graphs
    for k in [8, 9, 10, 11]:
        if time.time() - start_time > deadline - 5:
            break
        n, adj = andrasfai_graph(k)
        if is_triangle_free(adj, n):
            # Check alpha
            alpha = independent_set_size(adj, n)
            if alpha <= 9 and n > best_n:
                best_n = n
                best_adj = adj[:]

        # Try to extend Andrásfai graph
        for target_n in range(n + 1, 41):
            if time.time() - start_time > deadline - 5:
                break
            ext_adj = extend_vertex(adj, n, target_n)
            if ext_adj is None:
                continue
            if not is_triangle_free(ext_adj, target_n):
                continue
            # Reduce alpha
            ext_adj = local_search_reduce_alpha(ext_adj, target_n, 9)
            if not is_triangle_free(ext_adj, target_n):
                continue
            if not has_independent_set_size(ext_adj, target_n, 10, time_limit=3.0):
                if target_n > best_n:
                    best_n = target_n
                    best_adj = ext_adj[:]

    # Strategy 2: Random triangle-free + local search
    for seed in range(50):
        if time.time() - start_time > deadline - 5:
            break
        for target_n in [35, 36, 37, 38, 39, 40]:
            if time.time() - start_time > deadline - 3:
                break
            random.seed(seed * 100003 + target_n * 7)
            adj = clebsch_derived_graph(target_n)
            if not is_triangle_free(adj, target_n):
                continue
            adj = local_search_reduce_alpha(adj, target_n, 9)
            if not is_triangle_free(adj, target_n):
                continue
            if not has_independent_set_size(adj, target_n, 10, time_limit=3.0):
                if target_n > best_n:
                    best_n = target_n
                    best_adj = adj[:]

    # Strategy 3: Start from a known good size and try to extend
    if best_adj is not None and best_n < 40:
        for target_n in range(best_n + 1, 41):
            if time.time() - start_time > deadline - 3:
                break
            ext_adj = extend_vertex(best_adj, best_n, target_n)
            if ext_adj is None:
                continue
            if not is_triangle_free(ext_adj, target_n):
                continue
            ext_adj = local_search_reduce_alpha(ext_adj, target_n, 9)
            if not is_triangle_free(ext_adj, target_n):
                continue
            if not has_independent_set_size(ext_adj, target_n, 10, time_limit=3.0):
                best_n = target_n
                best_adj = ext_adj[:]

    if best_adj is None or best_n < 1:
        # Fallback: Andrásfai A_8
        n, adj = andrasfai_graph(8)
        best_n = n
        best_adj = adj

    # Final verification
    if not is_triangle_free(best_adj, best_n):
        # Should not happen, but fallback
        n, adj = andrasfai_graph(8)
        best_n = n
        best_adj = adj

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
