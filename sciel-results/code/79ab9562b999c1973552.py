import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin below 120s evaluator limit

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

    def has_independent_set_size(adj, n, k):
        """Returns True if there exists an independent set of size k."""
        found = [False]

        def dfs(candidates, current_size):
            if found[0]:
                return True
            if current_size >= k:
                found[0] = True
                return True
            if not candidates:
                return False
            if current_size + popcount(candidates) < k:
                return False

            # Pick vertex with highest degree in candidate subgraph
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            if best_v == -1:
                return False

            bit_v = 1 << best_v
            rest = candidates & ~bit_v

            # Branch include best_v
            non_neighbors = rest & ~adj[best_v]
            if dfs(non_neighbors, current_size + 1):
                return True

            # Branch exclude best_v
            if dfs(rest, current_size):
                return True

            return False

        dfs((1 << n) - 1, 0)
        return found[0]

    def verify_graph(n, edges):
        adj = build_adj_bitset(n, edges)
        if not is_triangle_free(adj, n):
            return False
        if has_independent_set_size(adj, n, 10):
            return False
        return True

    def find_triangles(adj, n):
        """Find all triangles in the graph. Returns list of (i,j,k)."""
        triangles = []
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                if j <= i:
                    continue
                common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                while common:
                    lsb2 = common & (-common)
                    k = lsb2.bit_length() - 1
                    common ^= lsb2
                    if k > j:
                        triangles.append((i, j, k))
        return triangles

    def make_triangle_free(adj, n, max_iter=500):
        """Remove edges to make graph triangle-free."""
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            triangles = find_triangles(adj, n)
            if not triangles:
                break
            # Remove one random edge from a random triangle
            i, j, k = random.choice(triangles)
            choice = random.randint(0, 2)
            if choice == 0:
                adj[i] &= ~(1 << j); adj[j] &= ~(1 << i)
            elif choice == 1:
                adj[i] &= ~(1 << k); adj[k] &= ~(1 << i)
            else:
                adj[j] &= ~(1 << k); adj[k] &= ~(1 << j)
        return adj

    def try_add_edges(adj, n, max_iter=200):
        """Try to add edges that don't create triangles."""
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            # Find all non-edges that can be added without creating a triangle
            safe_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        continue
                    common = adj[i] & adj[j]
                    if not common:
                        safe_edges.append((i, j))
            if not safe_edges:
                break
            # Add a random safe edge
            i, j = random.choice(safe_edges)
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        return adj

    def remove_edges_to_reduce_alpha(adj, n, max_iter=100):
        """Remove edges to try to reduce independence number."""
        for _ in range(max_iter):
            if time.time() - start_time > deadline - 3:
                break
            if not has_independent_set_size(adj, n, 10):
                break
            # Find an independent set of size 10
            # Use a greedy approach to find one
            indep = []
            candidates = (1 << n) - 1
            while candidates and len(indep) < 10:
                # Pick vertex with minimum degree in candidate subgraph
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
                if best_v == -1:
                    break
                indep.append(best_v)
                candidates &= ~adj[best_v]
                candidates &= ~(1 << best_v)
            
            if len(indep) < 10:
                break
            
            # Remove a random edge within this independent set... wait, there are no edges in an independent set.
            # Instead, we should add edges. But we already tried that.
            # Let's try a different strategy: pick two vertices in the independent set and see if we can add an edge.
            # But they are non-adjacent, and if they have a common neighbor, adding edge creates triangle.
            # So we need to remove edges from their common neighbors first.
            # This is getting complex. Let's just try random edge removal and re-addition.
            if indep:
                i, j = random.sample(indep, 2)
                common = adj[i] & adj[j]
                if common:
                    # Remove a random edge from common neighbors
                    lsb = common & (-common)
                    k = lsb.bit_length() - 1
                    # Remove edge (i,k) or (j,k)
                    if random.random() < 0.5:
                        adj[i] &= ~(1 << k); adj[k] &= ~(1 << i)
                    else:
                        adj[j] &= ~(1 << k); adj[k] &= ~(1 << j)
        return adj

    # --- Known constructions ---
    def andrasfai_graph(n):
        """Andrásfai graph A_k where n = 3k, k >= 4.
        Triangle-free, alpha = k."""
        k = n // 3
        if 3 * k != n:
            return None
        edges = []
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    for j in range(k):
                        if (b - a) % 3 == 1 and (j == i or (j - i) % k == 1):
                            u = a * k + i
                            v = b * k + j
                            if u < v:
                                edges.append([u, v])
        return {'n': n, 'edges': edges}

    # Baseline: Andrásfai A_9 has n=27, alpha=9
    best = {'n': 27, 'edges': []}
    ag = andrasfai_graph(27)
    if ag:
        best = ag

    # Also try A_10 on 30 vertices (alpha=10, so invalid) - skip
    # Try A_8 on 24 vertices (alpha=8, valid but smaller)

    # Try to build larger graphs
    for target_n in range(28, 41):
        if time.time() - start_time > deadline - 10:
            break

        found = False
        for attempt in range(20):
            if time.time() - start_time > deadline - 5:
                break

            random.seed(attempt * 7919 + target_n * 31 + 42)

            # Start with various initial densities
            density = 0.25 + 0.02 * attempt
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < density:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Phase 1: Make triangle-free
            adj = make_triangle_free(adj, target_n, max_iter=300)
            if not is_triangle_free(adj, target_n):
                continue

            # Phase 2: Add edges to reduce independence number
            adj = try_add_edges(adj, target_n, max_iter=150)

            # Phase 3: If still has independent set of size 10, try more aggressive approach
            if has_independent_set_size(adj, target_n, 10):
                # Try removing edges strategically and re-adding
                for sub_iter in range(50):
                    if time.time() - start_time > deadline - 3:
                        break
                    if not has_independent_set_size(adj, target_n, 10):
                        break
                    adj = remove_edges_to_reduce_alpha(adj, target_n, max_iter=30)
                    adj = try_add_edges(adj, target_n, max_iter=50)

            # Final verification
            if is_triangle_free(adj, target_n) and not has_independent_set_size(adj, target_n, 10):
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

    return best
