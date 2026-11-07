"""Optimized search by subset size with early termination."""
import time
import itertools

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    best_score = 0
    best_vertices = list(range(8)) # Fallback

    # Iterate by subset size k from n down to 8
    # If we find a subset of size k with k distinct degrees, it's the global optimum
    # because any smaller subset j < k can have at most j distinct degrees.
    for k in range(n, 7, -1):
        if time.time() > deadline - 2.0:
            break
            
        # If current best score is already >= k, we can't beat it with size k
        # (max distinct degrees for size k is k)
        if k <= best_score:
            break

        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            degrees = set()
            for v in combo:
                deg = (adj[v] & mask).bit_count()
                degrees.add(deg)
                # Early exit if we have k distinct degrees (perfect)
                if len(degrees) == k:
                    # This is the maximum possible score for this k
                    # And since we iterate k downwards, this is the global max
                    return list(combo)
            
            score = len(degrees)
            if score > best_score:
                best_score = score
                best_vertices = list(combo)

    return list(best_vertices)
# EVOLVE-BLOCK-END
