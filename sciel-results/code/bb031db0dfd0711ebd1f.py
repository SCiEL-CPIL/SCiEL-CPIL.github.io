import random
import time
import itertools

def construct():
    # Target n values to try, in descending order of preference
    # We try 22 first, then 21, then 20.
    # If we find a valid graph for a larger n, we return it immediately.
    # This maximizes the score.
    
    deadline = time.time() + 115 # Safety margin for 120s limit
    best_result = None
    
    # We will try to construct for n=22, 21, 20
    # Note: The problem says "Maximize n".
    # A valid graph on n vertices certifies G(6) > n.
    # So n=22 is better than n=21.
    
    for n in [22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        # Precompute all 6-subsets as lists of vertices
        # This is expensive for n=22 (74k subsets), but we only do it once per n.
        # Actually, we can just iterate in the loop.
        
        # Initialize random graph
        # Using a list of bitmasks for adjacency
        adj = [0] * n
        
        # Helper to get list of vertices from bitmask (not used if we use itertools directly)
        
        # Helper to check if a specific subset (list of vertices) is regular
        def is_regular_subset(verts, adj):
            if len(verts) < 6:
                return False # We only care about >= 6
            
            # Create a mask for the subset
            mask = 0
            for v in verts:
                mask |= (1 << v)
                
            # Calculate degrees
            degs = set()
            for v in verts:
                # Count bits in adj[v] that are also in mask
                # adj[v] & mask gives the neighbors of v within the subset
                deg = (adj[v] & mask).bit_count()
                degs.add(deg)
                if len(degs) > 1:
                    return False
            return True

        # Helper to find a violation (subset of size >= 6 that is regular)
        # We prioritize size 6 as they are most numerous and likely to be violated.
        def find_violation(adj, n):
            # Check size 6
            for verts in itertools.combinations(range(n), 6):
                if is_regular_subset(verts, adj):
                    return verts
            # Check size 7
            for verts in itertools.combinations(range(n), 7):
                if is_regular_subset(verts, adj):
                    return verts
            # Check size 8
            for verts in itertools.combinations(range(n), 8):
                if is_regular_subset(verts, adj):
                    return verts
            # ... up to n
            # For n=22, checking all sizes is too slow.
            # However, if size 6 is clear, the probability of a larger regular subgraph is lower?
            # Not necessarily. But usually, local search fixes small violations first.
            # We'll check up to size 10 for speed, then assume valid if no small violation?
            # No, we must be sure. 
            # Strategy: If no violation of size 6, 7, 8, 9, 10, 11, 12 is found, 
            # we might still have a violation of size 13+.
            # But for the purpose of the search, if we clear all small ones, 
            # it's likely valid. 
            # To be safe, we do a final full check before returning.
            
            # For the search loop, we only check size 6 to keep it fast.
            # If size 6 is clear, we do a full check.
            return None

        # Local Search Loop
        # We will run multiple restarts for each n if we don't find a solution quickly.
        # But finding a valid graph for n=22 is hard.
        # Let's try a few random starts and local searches.
        
        found = False
        max_restarts = 50 # Adjust based on time
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Reset graph
            adj = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            # Local search steps
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                # Check for violations of size 6
                # We can sample some random 6-subsets to speed up finding a violation
                # Or iterate all? Iterating all 74k is ~0.1s in Python if optimized?
                # is_regular_subset is fast.
                # Let's try to find one violation.
                
                violation = None
                # To speed up, we can just pick a random 6-subset first.
                # If it's regular, we fix it.
                # If not, we might miss others.
                # Better: Iterate all 6-subsets until one is found.
                # If none found, check larger.
                
                # Optimization: Keep track of which 6-subsets are currently regular?
                # Too complex. Just scan.
                
                # Scan size 6
                for verts in itertools.combinations(range(n), 6):
                    if is_regular_subset(verts, adj):
                        violation = verts
                        break
                        
                if violation:
                    # Flip a random edge in the violation
                    u, v = random.sample(violation, 2)
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue # Re-check after flip
                
                # If no size 6 violation, check size 7, 8, etc.
                # We'll check up to size 10 to balance speed and coverage.
                violation_found_in_large = False
                for size in range(7, 11):
                    for verts in itertools.combinations(range(n), size):
                        if is_regular_subset(verts, adj):
                            violation = verts
                            violation_found_in_large = True
                            break
                    if violation_found_in_large:
                        break
                        
                if violation_found_in_large:
                    u, v = random.sample(violation, 2)
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    continue
                
                # If no violation found in size 6-10, do a full check
                # Full check is expensive, so only do it if we think we are close.
                # We'll do a full check every 50 steps or when no small violation is found.
                
                # Actually, if we didn't find a violation in 6-10, we should check the rest.
                # But checking 11..22 is very expensive.
                # Heuristic: If no violation in 6-10, it's likely valid.
                # Let's do a full check to be sure.
                
                is_valid = True
                for size in range(11, n+1):
                    for verts in itertools.combinations(range(n), size):
                        if is_regular_subset(verts, adj):
                            is_valid = False
                            violation = verts
                            break
                    if not is_valid:
                        break
                        
                if is_valid:
                    # Construct edge list
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                else:
                    # Fix the large violation
                    u, v = random.sample(violation, 2)
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
        # If we didn't find a valid graph for this n, try next smaller n

    # Fallback: Return a valid graph on n=5 (empty) or try to return best found?
    # The task says "Maximize n". If we fail to find 20, we should return something valid.
    # A random graph on 5 is likely valid (no 6 vertices).
    # Actually, any graph on n < 6 is valid.
    # Let's return a valid graph on n=20 if we found one in a previous iteration?
    # No, we return immediately when found.
    # If we loop through 22, 21, 20 and fail all, we return a small valid graph.
    
    # To ensure we return *something* valid, let's construct a trivial one.
    # n=5, no edges.
    return {'n': 5, 'edges': []}
