import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    def has_independent_set_10(adj, n):
        """Check if graph has independent set of size 10. Returns True if it does."""
        mask_all = (1 << n) - 1

        # Greedy lower bound: try to find a large independent set quickly
        candidates = mask_all
        size = 0
        while candidates and size < 10:
            best_v = -1
            min_deg = n + 1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_v = v

            if best_v == -1:
                break

            size += 1
            candidates &= ~adj[best_v] & ~(1 << best_v)

        if size >= 10:
            return True

        # Exact branch and bound
        sys.setrecursionlimit(10000)

        def backtrack(candidates, size):
            if size == 10:
                return True
            if candidates == 0:
                return False

            remaining = candidates.bit_count()
            if size + remaining < 10:
                return False

            # Pick vertex with max degree in candidates (removes most)
            best_v = -1
            max_deg = -1
            temp = candidates
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (adj[v] & candidates).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v

            if best_v == -1:
                return False

            # Branch 1: include best_v
            if backtrack(candidates & ~adj[best_v] & ~(1 << best_v), size + 1):
                return True
            # Branch 2: exclude best_v
            if backtrack(candidates & ~(1 << best_v), size):
                return True
            return False

        return backtrack(mask_all, 0)

    def is_triangle_free(adj, n):
        """Check if graph is triangle-free."""
        for u in range(n):
            neighbors = adj[u]
            temp = neighbors
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                if v > u:
                    if adj[u] & adj[v]:
                        return False
        return True

    def build_edge_list(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            temp = adj[i]
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                if j > i:
                    edges.append([i, j])
                temp &= temp - 1
        return edges

    def generate_random_triangle_free(n):
        """Generate a random triangle-free graph using greedy edge addition."""
        adj = [0] * n
        # List of all potential edges
        potential_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(potential_edges)
        
        for u, v in potential_edges:
            # Check if adding (u, v) creates a triangle
            # A triangle is formed if there is a common neighbor w of u and v
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def try_find_graph(n, max_restarts=1000):
        """Try to find a valid graph on n vertices."""
        for _ in range(max_restarts):
            if time.time() > deadline:
                return None
            
            adj = generate_random_triangle_free(n)
            
            # Check if valid
            if not has_independent_set_10(adj, n):
                return build_edge_list(adj, n)
                
        return None

    # Main search loop: Try largest n first
    # n=40 is the theoretical max if R(3,10)=41
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
            
        # For n=40, try more restarts if time permits
        # For smaller n, fewer restarts might suffice
        edges = try_find_graph(n, max_restarts=5000 if n == 40 else 2000)
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback: Return a known valid graph
    # Andrasfai graph And(9) has n=27, alpha=9.
    def make_andrasfai(k):
        n = 3 * k
        adj = [0] * n
        # Distances k+1 to 2k-1
        for i in range(n):
            for d in range(k + 1, 2 * k):
                j1 = (i + d) % n
                j2 = (i - d) % n
                adj[i] |= (1 << j1)
                adj[i] |= (1 << j2)
        return adj, n

    # Try And(9) n=27
    adj, n = make_andrasfai(9)
    # Verify it (should be valid)
    if is_triangle_free(adj, n) and not has_independent_set_10(adj, n):
        return {'n': n, 'edges': build_edge_list(adj, n)}
        
    # Absolute fallback: C5 (n=5, alpha=2)
    c5_adj = [0] * 5
    for i in range(5):
        c5_adj[i] |= (1 << ((i + 1) % 5))
        c5_adj[i] |= (1 << ((i + 4) % 5))
    return {'n': 5, 'edges': build_edge_list(c5_adj, 5)}

if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}, edges={len(res['edges'])}")
