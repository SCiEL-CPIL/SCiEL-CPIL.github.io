"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    """Construct a triangle-free graph with independence number < 10, maximizing n."""
    start_time = time.time()
    deadline = start_time + 105  # 105s safety margin (150s limit)

    # The Andrasfai graph And(9) on 26 vertices is the baseline
    # It's triangle-free with alpha = 9
    def andrasfai_26():
        n = 26
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 26 % 3 == 1]
        return {'n': n, 'edges': edges}

    best = andrasfai_26()

    # Fast verification using bitsets
    def build_adj_bitset(n, edges):
        """Build adjacency as list of Python ints (bitsets)."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def is_triangle_free_bitset(adj, n):
        """Check triangle-free using bitset operations."""
        for i in range(n):
            # For each pair of neighbors j,k of i, check if j-k edge exists
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                # Check if any neighbor of j (other than i) is also neighbor of i
                common = adj[j] & adj[i]
                # Remove i from common
                common &= ~(1 << i)
                if common:
                    return False
        return True

    def max_independent_set_size_fast(adj, n):
        """Exact maximum independent set size using branch and bound with bitsets.
        Returns the size, and stops early if it reaches 10."""
        best_size = [0]
        
        def popcount(x):
            return bin(x).count('1')
        
        def dfs(candidates, current_size):
            if not candidates:
                if current_size > best_size[0]:
                    best_size[0] = current_size
                return
            if current_size + popcount(candidates) <= best_size[0]:
                return
            # Early exit if we already found size >= 10
            if best_size[0] >= 10:
                return
            
            # Pick vertex with max degree in candidate subgraph for better pruning
            # Use a heuristic: pick vertex with highest degree among candidates
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
            
            v = best_v
            bit_v = 1 << v
            rest = candidates & ~bit_v
            
            # Branch 1: include v
            non_neighbors = rest & ~adj[v]
            dfs(non_neighbors, current_size + 1)
            
            # Branch 2: exclude v
            dfs(rest, current_size)
        
        dfs((1 << n) - 1, 0)
        return best_size[0]

    def verify_graph(n, edges):
        """Check if graph is triangle-free and alpha < 10."""
        adj = build_adj_bitset(n, edges)
        if not is_triangle_free_bitset(adj, n):
            return False
        if max_independent_set_size_fast(adj, n) >= 10:
            return False
        return True

    # Try to extend the Andrasfai graph or find better constructions
    # Strategy: use random restarts with local search
    for target_n in range(27, 41):
        if time.time() - start_time > deadline - 5:
            break

        # Try multiple random attempts
        for attempt in range(20):
            if time.time() - start_time > deadline - 5:
                break

            random.seed(attempt * target_n + 12345)

            # Start with a random graph with moderate edge density
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.4:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)

            # Phase 1: Remove edges to make triangle-free
            for _ in range(200):
                # Find a triangle using bitsets
                triangle = None
                for i in range(target_n):
                    neighbors = adj[i]
                    while neighbors:
                        lsb1 = neighbors & (-neighbors)
                        j = lsb1.bit_length() - 1
                        neighbors ^= lsb1
                        # Find common neighbors of i and j (excluding i and j)
                        common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                        if common:
                            lsb2 = common & (-common)
                            k = lsb2.bit_length() - 1
                            triangle = (i, j, k)
                            break
                    if triangle:
                        break
                if triangle is None:
                    break

                # Remove a random edge from the triangle
                i, j, k = triangle
                edge_idx = random.randint(0, 2)
                if edge_idx == 0:
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                elif edge_idx == 1:
                    adj[i] &= ~(1 << k)
                    adj[k] &= ~(1 << i)
                else:
                    adj[j] &= ~(1 << k)
                    adj[k] &= ~(1 << j)

            # Check if triangle-free
            if not is_triangle_free_bitset(adj, target_n):
                continue

            # Phase 2: Add edges to reduce independence number
            for _ in range(100):
                # Check current independence number (with early exit)
                alpha = max_independent_set_size_fast(adj, target_n)
                if alpha < 10:
                    break

                # Try to add a random non-edge that doesn't create a triangle
                i = random.randint(0, target_n - 1)
                j = random.randint(0, target_n - 1)
                if i == j or (adj[i] & (1 << j)):
                    continue

                # Check if adding this edge creates a triangle
                common = adj[i] & adj[j]
                if common:
                    continue

                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

            # Final check
            if is_triangle_free_bitset(adj, target_n):
                alpha = max_independent_set_size_fast(adj, target_n)
                if alpha < 10:
                    edges = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    if target_n > best['n']:
                        best = {'n': target_n, 'edges': edges}
                    break

    return best
# EVOLVE-BLOCK-END
