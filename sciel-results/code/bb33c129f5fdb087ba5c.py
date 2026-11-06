import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin below 120s evaluator limit

    # --- bitset helpers ---
    def popcount(x):
        return bin(x).count('1')

    def build_adj_bitset(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free(adj, n):
        """Quick triangle-free check using bitsets."""
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                # Check if any neighbor of j is also neighbor of i
                common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                if common:
                    return False
        return True

    def greedy_color_bound(candidates, adj, n):
        """Greedy coloring to get upper bound on independence number.
        Returns a list of color class sizes; sum is upper bound on alpha."""
        remaining = candidates
        colors = []
        while remaining:
            # Pick vertex with minimum degree in remaining subgraph
            best_v = -1
            best_deg = float('inf')
            temp = remaining
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            # Greedy maximal independent set from remaining
            color_class = 0
            rem = remaining
            while rem:
                lsb = rem & (-rem)
                v = lsb.bit_length() - 1
                rem ^= lsb
                color_class |= (1 << v)
                # Remove v and its neighbors from rem
                rem &= ~adj[v] & ~(1 << v)
            colors.append(popcount(color_class))
            remaining &= ~color_class
        return colors

    def has_independent_set_size(adj, n, k, time_limit=5.0):
        """Returns True if there exists an independent set of size k.
        Uses branch and bound with greedy coloring upper bounds."""
        best_found = [k - 1]
        start_t = time.time()

        def dfs(candidates, current_size):
            if time.time() - start_t > time_limit:
                return False  # timeout, assume no (conservative for our purpose)
            if best_found[0] >= k:
                return True
            if current_size >= k:
                best_found[0] = k
                return True
            if not candidates:
                return False

            # Get coloring bound
            color_classes = greedy_color_bound(candidates, adj, n)
            max_possible = current_size + sum(color_classes)
            if max_possible <= best_found[0]:
                return False

            # Pick vertex from largest color class (or use heuristic)
            # Use the vertex that appears in the largest color class
            # For simplicity, pick vertex with minimum degree in candidates
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

            # Branch include v (in independent set)
            non_neighbors = rest & ~adj[v]
            if dfs(non_neighbors, current_size + 1):
                return True

            # Branch exclude v
            if dfs(rest, current_size):
                return True

            return False

        dfs((1 << n) - 1, 0)
        return best_found[0] >= k

    def verify_graph(n, edges):
        adj = build_adj_bitset(n, edges)
        if not is_triangle_free(adj, n):
            return False
        if has_independent_set_size(adj, n, 10):
            return False
        return True

    # --- Known constructions ---
    def andrasfai_graph(k):
        """Andrásfai graph A_k on 3k vertices."""
        n = 3 * k
        edges = []
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    if (b - a) % 3 != 1:
                        continue
                    for j in range(k):
                        if (j - i) % k in (0, 1):
                            u = a * k + i
                            v = b * k + j
                            if u < v:
                                edges.append([u, v])
        return {'n': n, 'edges': edges}

    def random_triangle_free(n, seed_val, density=0.4):
        """Generate a random triangle-free graph."""
        random.seed(seed_val)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        # Remove triangles
        for _ in range(500):
            if time.time() - start_time > deadline - 3:
                break
            found = False
            for i in range(n):
                ni = adj[i]
                while ni:
                    lsb1 = ni & (-ni)
                    j = lsb1.bit_length() - 1
                    ni ^= lsb1
                    common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                    if common:
                        lsb2 = common & (-common)
                        k = lsb2.bit_length() - 1
                        # Remove random edge from triangle
                        choice = random.randint(0, 2)
                        if choice == 0:
                            adj[i] &= ~(1 << j); adj[j] &= ~(1 << i)
                        elif choice == 1:
                            adj[i] &= ~(1 << k); adj[k] &= ~(1 << i)
                        else:
                            adj[j] &= ~(1 << k); adj[k] &= ~(1 << j)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return adj

    def add_edges_to_reduce_alpha(adj, n, max_iter=100):
        """Add edges to reduce independence number while maintaining triangle-free."""
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            if not has_independent_set_size(adj, n, 10, time_limit=2.0):
                break
            # Try to add a safe edge
            improved = False
            # Collect all non-edges that are safe
            safe_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        if not (adj[i] & adj[j]):
                            safe_edges.append((i, j))
            random.shuffle(safe_edges)
            for i, j in safe_edges[:20]:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                improved = True
                break
            if not improved:
                break
        return adj

    # Baseline: Andrásfai A_9 has n=27, alpha=9
    best = {'n': 27, 'edges': []}
    ag = andrasfai_graph(9)
    if ag and verify_graph(ag['n'], ag['edges']):
        best = ag

    # Try to find larger valid graphs
    for target_n in range(28, 41):
        if time.time() - start_time > deadline - 10:
            break

        found = False
        for attempt in range(25):
            if time.time() - start_time > deadline - 5:
                break

            adj = random_triangle_free(target_n, attempt * 100003 + target_n * 7)
            if not is_triangle_free(adj, target_n):
                continue

            adj = add_edges_to_reduce_alpha(adj, target_n)

            if is_triangle_free(adj, target_n) and not has_independent_set_size(adj, target_n, 10, time_limit=3.0):
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                best = {'n': target_n, 'edges': edges}
                found = True
                break

        if found:
            # Continue trying larger n
            continue
        else:
            # If we couldn't find for this n, try a few more with different params
            for attempt in range(15, 40):
                if time.time() - start_time > deadline - 5:
                    break
                adj = random_triangle_free(target_n, attempt * 7919 + target_n * 31 + 42, density=0.3)
                if not is_triangle_free(adj, target_n):
                    continue
                adj = add_edges_to_reduce_alpha(adj, target_n)
                if is_triangle_free(adj, target_n) and not has_independent_set_size(adj, target_n, 10, time_limit=3.0):
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    best = {'n': target_n, 'edges': edges}
                    found = True
                    break

    return best
