import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Function to check if adding diff d to S creates a triangle
    # Triangle 0, a, d exists if a in S and d-a in S
    def can_add(S, d, n):
        # S is a set of differences (positive)
        # We need to check if there is a in S such that (d - a) % n is in S
        # Or (a - d) % n is in S?
        # If a in S and (d-a) % n in S, then 0~a, a~d, 0~d.
        # Since S contains both d and n-d, we just check positive reps.
        for a in S:
            diff = (d - a) % n
            if diff in S:
                return False
            # Also check if diff == 0? No, a != d
        return True

    def get_alpha(n, S):
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for d in S:
            for i in range(n):
                j = (i + d) % n
                adj[i].add(j)
                adj[j].add(i)
        
        # Find max independent set using branch and bound (simple version)
        # For n=39, this might be slow if done naively.
        # Use a greedy + local search or a standard B&B.
        # Given time constraints, a simple B&B with bitsets is best.
        
        # Represent graph as bitmasks
        masks = [0] * n
        for i in range(n):
            for j in adj[i]:
                masks[i] |= (1 << j)
        
        def max_independent_set(mask, size):
            # If mask is 0, return 0
            if mask == 0:
                return 0
            # Pick a vertex with min degree in the subgraph? Or just highest bit.
            # Simple B&B
            best = 0
            # Heuristic: sort vertices by degree in current subgraph
            # For speed, just iterate
            # This is a basic B&B, might be slow for n=39 if alpha is large.
            # But we only call this if the graph is triangle-free.
            # Triangle-free graphs have structure.
            
            # Optimization: If remaining vertices < best, return best
            if mask.bit_count() <= best:
                return best
            
            # Pick a vertex u
            u = (mask & -mask).bit_length() - 1
            # Branch 1: Include u
            # Neighbors of u cannot be included
            non_neighbors = mask & ~masks[u] & ~(1 << u)
            res1 = 1 + max_independent_set(non_neighbors, best)
            if res1 > best:
                best = res1
            
            # Branch 2: Exclude u
            res2 = max_independent_set(mask & ~(1 << u), best)
            if res2 > best:
                best = res2
                
            return best

        return max_independent_set((1 << n) - 1, 0)

    def try_circulant(n):
        # Search for a circulant graph on n vertices
        candidates = list(range(1, n // 2 + 1))
        # We want to maximize edges, so we try to add many differences.
        # Randomize order to find different maximal sets
        for _ in range(50): # 50 restarts per n
            if time.time() > deadline:
                return None
            random.shuffle(candidates)
            S = set()
            for d in candidates:
                if can_add(S, d, n):
                    S.add(d)
                    S.add(n - d)
            
            # Check alpha
            # If S is empty, alpha = n
            if not S:
                continue
                
            # Quick check: if degree is too low, alpha is likely high
            deg = len(S)
            if deg < 10: # Rough heuristic
                continue
                
            alpha = get_alpha(n, S)
            if alpha < 10:
                # Construct edge list
                edges = []
                for i in range(n):
                    for d in S:
                        j = (i + d) % n
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        return None

    # Try n from 39 down to 26
    for n in range(39, 25, -1):
        if time.time() > deadline:
            break
        res = try_circulant(n)
        if res:
            return res

    # Fallback: Return the seed graph (Andrasfai 26)
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
