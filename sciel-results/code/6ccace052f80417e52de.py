import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def check_regular(mask, adj):
        """Check if induced subgraph on mask is regular. Returns True if regular."""
        # Extract vertices
        m = mask
        if m == 0:
            return False
        first_deg = None
        count = 0
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if first_deg is None:
                first_deg = deg
            elif deg != first_deg:
                return False
            count += 1
            m ^= lsb
        return count >= 6

    def full_check(adj, n):
        """
        Check all subsets of size >= 6.
        Returns (is_valid, violating_mask, violating_subset)
        """
        vertices = list(range(n))
        # Check sizes from 6 up to n
        for k in range(6, n + 1):
            for subset in combinations(vertices, k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if check_regular(mask, adj):
                    return False, mask, subset
        return True, None, None

    def edges_to_adj(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    edges.append([u, v])
        return edges

    # Hardcoded valid 20-vertex graph (from inspiration 2, assumed valid)
    # If this is invalid, the program will fall back to search for n=20
    hardcoded_edges_20 = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
        [1, 2], [1, 3], [1, 4], [1, 5], [1, 10], [1, 11], [1, 12], [1, 13],
        [2, 3], [2, 4], [2, 6], [2, 7], [2, 14], [2, 15], [2, 16], [2, 17],
        [3, 4], [3, 5], [3, 8], [3, 9], [3, 18], [3, 19],
        [4, 5], [4, 6], [4, 7], [4, 10], [4, 11],
        [5, 6], [5, 7], [5, 12], [5, 13],
        [6, 7], [6, 8], [6, 9], [6, 14], [6, 15],
        [7, 8], [7, 9], [7, 16], [7, 17],
        [8, 9], [8, 10], [8, 11], [8, 18], [8, 19],
        [9, 10], [9, 11], [9, 12], [9, 13],
        [10, 11], [10, 14], [10, 15],
        [11, 12], [11, 16], [11, 17],
        [12, 13], [12, 18], [12, 19],
        [13, 14], [13, 15],
        [14, 15], [14, 18], [14, 19],
        [15, 16], [15, 17],
        [16, 17], [16, 18], [16, 19],
        [17, 18], [17, 19],
        [18, 19]
    ]

    best_n = 0
    best_edges = []

    # First, verify and set baseline with hardcoded graph
    if len(hardcoded_edges_20) > 0:
        adj_20 = edges_to_adj(hardcoded_edges_20, 20)
        # Quick check: if it's valid, set as best
        # full_check is expensive, so we do it only once
        is_valid, _, _ = full_check(adj_20, 20)
        if is_valid:
            best_n = 20
            best_edges = hardcoded_edges_20
        else:
            # If hardcoded is invalid (unlikely if from inspiration), we rely on search
            pass

    # Try to find larger graphs: 22, 21
    # We try 22 first, then 21. If we find one, we return it.
    # If we don't, we return the best found (likely 20).

    for target_n in [22, 21]:
        if time.time() > deadline - 10:
            break
        if target_n <= best_n:
            continue

        # Local search for target_n
        # We use multiple restarts
        found = False
        max_restarts = 50 if target_n == 21 else 20
        
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break

            # Initialize random graph
            adj = [0] * target_n
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

            # Local search loop
            # We limit iterations per restart to avoid getting stuck
            max_iters = 1000 if target_n == 21 else 500
            
            for it in range(max_iters):
                if time.time() > deadline - 2:
                    break

                # Find a violation
                # To speed this up, we can check only k=6 first, then 7, etc.
                # full_check does this, but we can optimize by returning early.
                is_valid, viol_mask, viol_subset = full_check(adj, target_n)
                
                if is_valid:
                    # Found a valid graph!
                    best_n = target_n
                    best_edges = adj_to_edges(adj, target_n)
                    found = True
                    break
                else:
                    # Flip an edge in the violating subset
                    if viol_subset:
                        u, v = random.sample(list(viol_subset), 2)
                        # Toggle edge
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)

            if found:
                break

    # If we didn't find anything better than 20, and 20 wasn't verified, try to find 20 via search
    if best_n < 20:
        if time.time() < deadline - 5:
            # Try to find a valid 20-vertex graph via search
            adj = [0] * 20
            for u in range(20):
                for v in range(u + 1, 20):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            for it in range(10000):
                if time.time() > deadline - 2:
                    break
                is_valid, viol_mask, viol_subset = full_check(adj, 20)
                if is_valid:
                    best_n = 20
                    best_edges = adj_to_edges(adj, 20)
                    break
                else:
                    if viol_subset:
                        u, v = random.sample(list(viol_subset), 2)
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)

    # Fallback: if still no graph, return empty or small graph
    if best_n == 0:
        # Return a valid graph on 5 vertices (trivially valid)
        best_n = 5
        best_edges = []

    return {'n': best_n, 'edges': best_edges}
