import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s margin for 120s limit
    
    def is_valid(adj, n):
        # Check all subsets of size >= 6
        # We can iterate masks from 0 to 2^n - 1
        # But 2^21 is ~2M, which is okay for a few checks, but not for thousands.
        # We need a fast invalidity detector.
        
        # If we just do a full check, it's expensive.
        # Let's use a heuristic check during search, and full check at the end.
        pass

    def full_check(adj, n):
        # Check all subsets of size k >= 6
        # Using bitmasks
        for k in range(6, n + 1):
            # Iterate combinations of size k
            # itertools.combinations is slow for large n
            # Bitmask iteration is faster
            pass

    # Better: Use a specific known graph or a very fast search.
    # Given the difficulty, I will try to find a 21-vertex graph.
    # If not, 20.
    
    # Helper: Check if induced subgraph on mask is regular
    def is_regular_mask(adj, mask, n):
        # Get vertices in mask
        verts = [i for i in range(n) if (mask >> i) & 1]
        if len(verts) < 6:
            return False
        # Calculate degrees within mask
        degs = []
        for v in verts:
            # Count neighbors in mask
            # adj[v] is a bitmask of neighbors
            neighbors_in_mask = adj[v] & mask
            degs.append(bin(neighbors_in_mask).count('1'))
        # Check if all degrees are equal
        if all(d == degs[0] for d in degs):
            return True
        return False

    def find_violation(adj, n):
        # Returns a mask of a regular induced subgraph of size >= 6, or 0
        # Check sizes from 6 to n
        # To speed up, we can check random subsets first?
        # No, we need to be sure.
        # Let's iterate masks.
        # For n=21, 2^21 is 2M. 
        # Checking each mask takes O(n) time. 2M * 21 ~ 42M ops. 
        # In Python, this is too slow (seconds per check).
        
        # Optimization: 
        # If a mask is regular, its subsets might not be.
        # We can use a recursive check or just accept the slowness for a few iterations.
        # But we need many iterations for search.
        
        # Alternative: 
        # Only check masks of size 6. 
        # If no size-6 regular subgraph, is it possible to have size-7?
        # Yes. But size 6 is the most restrictive.
        # Let's assume that if we avoid size-6 regular subgraphs, we are close to valid.
        # We will do a full check only at the end of a candidate.
        
        # For the search step, finding a size-6 regular subgraph is sufficient to guide the move.
        
        # Iterate over all masks of size 6?
        # C(21, 6) = 54264. This is fast!
        # If we find a size-6 regular subgraph, return it.
        # If not, we might still have a size-7 regular subgraph.
        # But size-6 violations are much more common in random graphs.
        
        # So: 
        # 1. Check all size-6 subsets. If any regular, return mask.
        # 2. If none, do a full check (or check size 7, 8...).
        
        # Let's implement the size-6 check first.
        from itertools import combinations
        for comb in combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            if is_regular_mask(adj, mask, n):
                return mask
        
        # If no size 6, check larger sizes?
        # For n=20, C(20, 7) = 77520. C(20, 8) = 125970.
        # Total subsets is 2^20 ~ 1M.
        # Let's just do a full bitmask scan for the "full_check" function.
        # For "find_violation" during search, just checking size 6 is a good heuristic.
        return 0

    def full_check_slow(adj, n):
        # Exhaustive check
        for mask in range(1 << n):
            # Count bits
            if bin(mask).count('1') < 6:
                continue
            if is_regular_mask(adj, mask, n):
                return False
        return True

    # Search for n=21
    n_target = 21
    # Initialize random graph
    # Use adjacency as list of ints (bitmasks)
    adj = [0] * n_target
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    # Local Search
    best_n = 0
    best_adj = None
    
    # Try to find valid graph on 21
    found_21 = False
    iterations = 0
    while time.time() - start_time < time_limit * 0.6: # Use 60% time for n=21
        violation_mask = find_violation(adj, n_target)
        if violation_mask == 0:
            # Check full validity to be sure (size > 6)
            if full_check_slow(adj, n_target):
                found_21 = True
                best_n = 21
                best_adj = adj
                break
            else:
                # Find the actual violation from full check? 
                # full_check_slow returns bool. Let's modify it to return mask.
                pass
        
        # If violation found, flip an edge in the violation mask
        if violation_mask:
            # Get vertices in mask
            verts = [i for i in range(n_target) if (violation_mask >> i) & 1]
            # Pick a random pair in verts
            u, v = random.sample(verts, 2)
            # Flip edge u-v
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        else:
            # No size-6 violation, but maybe larger?
            # Flip random edge
            u, v = random.sample(range(n_target), 2)
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        iterations += 1
        if iterations % 100 == 0:
             # Optional: restart or diversify
             pass

    if found_21:
        return {'n': 21, 'edges': [[i, j] for i in range(21) for j in range(i+1, 21) if best_adj[i] & (1 << j)]}

    # Try n=20
    n_target = 20
    adj = [0] * n_target
    for i in range(n_target):
        for j in range(i + 1, n_target):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    found_20 = False
    iterations = 0
    while time.time() - start_time < time_limit:
        violation_mask = find_violation(adj, n_target)
        if violation_mask == 0:
            if full_check_slow(adj, n_target):
                found_20 = True
                best_n = 20
                best_adj = adj
                break
        
        if violation_mask:
            verts = [i for i in range(n_target) if (violation_mask >> i) & 1]
            u, v = random.sample(verts, 2)
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        else:
            u, v = random.sample(range(n_target), 2)
            if adj[u] & (1 << v):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        iterations += 1

    if found_20:
        return {'n': 20, 'edges': [[i, j] for i in range(20) for j in range(i+1, 20) if best_adj[i] & (1 << j)]}

    # Fallback: return a small valid graph
    # Empty graph on 5 vertices is valid
    return {'n': 5, 'edges': []}
