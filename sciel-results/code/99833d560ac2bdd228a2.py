"""Optimized brute-force with local search refinement to maximize distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Precompute adjacency as integer bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Fast popcount using bit manipulation
    def popcount(x):
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c

    def compute_distinct_degrees(combo):
        """Compute number of distinct degrees in induced subgraph on combo."""
        mask = 0
        for v in combo:
            mask |= (1 << v)
        degs = set()
        for v in combo:
            degs.add(popcount(adj[v] & mask))
        return len(degs)

    best_count = -1
    best_vertices = list(range(minv))

    vertices = list(range(n))

    # Phase 1: Brute-force from largest subset size to smallest
    for size in range(n, minv - 1, -1):
        if size <= best_count:
            break
        
        if time.time() > deadline:
            break

        for combo in itertools.combinations(vertices, size):
            if time.time() > deadline:
                break
            
            c = compute_distinct_degrees(combo)
            if c > best_count:
                best_count = c
                best_vertices = list(combo)
                
                # If we found a "graded" subset (all distinct degrees), 
                # this is the max for this size. Continue to smaller sizes.

    # Phase 2: Local search refinement
    # Try swapping vertices in and out to improve distinct degree count
    if time.time() < deadline:
        current = set(best_vertices)
        current_count = best_count
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for v_in in range(n):
                if v_in in current:
                    continue
                for v_out in current:
                    new_set = current - {v_out} | {v_in}
                    if len(new_set) < minv:
                        continue
                    c = compute_distinct_degrees(tuple(new_set))
                    if c > current_count:
                        current_count = c
                        current = new_set
                        best_count = c
                        best_vertices = list(current)
                        improved = True
                        break
                if improved:
                    break
            # Also try removing one vertex if size > minv
            if not improved and len(current) > minv:
                for v_out in current:
                    new_set = current - {v_out}
                    c = compute_distinct_degrees(tuple(new_set))
                    if c > current_count:
                        current_count = c
                        current = new_set
                        best_count = c
                        best_vertices = list(current)
                        improved = True
                        break
            # Also try adding one vertex if size < n
            if not improved and len(current) < n:
                for v_in in range(n):
                    if v_in in current:
                        continue
                    new_set = current | {v_in}
                    c = compute_distinct_degrees(tuple(new_set))
                    if c > current_count:
                        current_count = c
                        current = new_set
                        best_count = c
                        best_vertices = list(current)
                        improved = True
                        break

    # Ensure exact Python int types
    out = [int(v) for v in best_vertices]
    return out
# EVOLVE-BLOCK-END
