import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We will try to find a graph for n=20, then 19, etc.
    # If we find a valid graph, we return it.
    
    for target_n in [20, 19, 18, 17, 16]:
        if time.time() > deadline - 10:
            break
            
        n = target_n
        # Precompute 6-subsets masks
        masks6 = []
        mask_vertices = []
        for subset in combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            masks6.append(mask)
            mask_vertices.append(list(subset))
            
        max_restarts = 10 if n == 20 else 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random init
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            def count_violations_fast(adj):
                count = 0
                example_mask = 0
                for i in range(len(masks6)):
                    mask = masks6[i]
                    verts = mask_vertices[i]
                    d0 = (adj[verts[0]] & mask).bit_count()
                    # Unroll for speed
                    if (adj[verts[1]] & mask).bit_count() != d0: continue
                    if (adj[verts[2]] & mask).bit_count() != d0: continue
                    if (adj[verts[3]] & mask).bit_count() != d0: continue
                    if (adj[verts[4]] & mask).bit_count() != d0: continue
                    if (adj[verts[5]] & mask).bit_count() != d0: continue
                    count += 1
                    if count == 1:
                        example_mask = mask
                return count, example_mask

            current_count, current_example = count_violations_fast(adj)
            
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            iterations = 0
            max_iterations = 5000 # Increased iterations
            
            while T > min_T and iterations < max_iterations:
                if time.time() > deadline - 5:
                    break
                    
                # If we have a violation, try to fix it
                if current_count > 0:
                    # Flip an edge in the example violation
                    mask = current_example
                    verts = mask_vertices[masks6.index(mask)] # This is O(N), bad. 
                    # Better: store index in example? Or just re-derive?
                    # Let's modify count_violations to return index.
                    
                    # For now, let's just pick a random pair from the mask bits.
                    # We can extract verts from mask quickly if we precomputed.
                    # But we need the specific mask's verts.
                    # Let's change count_violations to return (count, example_idx)
                    
                    # Re-implementation of count_violations to return index
                    # ...
                    
                    # Actually, let's just use a simpler approach for the flip:
                    # Pick a random vertex u in mask, random vertex v in mask.
                    m = mask
                    verts_in_mask = []
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        verts_in_mask.append(idx)
                        m ^= lsb
                    
                    if len(verts_in_mask) < 2: continue
                    u, v = random.sample(verts_in_mask, 2)
                    
                    # Flip edge (u, v)
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    new_count, new_example = count_violations_fast(adj)
                    
                    delta = new_count - current_count
                    
                    if delta < 0 or (random.random() < math.exp(-delta / T)):
                        current_count = new_count
                        current_example = new_example
                    else:
                        # Revert
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                else:
                    # No size-6 violations. Check full validity.
                    break
                
                T *= cooling
                iterations += 1
                
            # Check if current graph is fully valid
            if current_count == 0:
                # Verify no regular subsets of size > 6
                valid = True
                for k in range(7, n + 1):
                    # For n=20, k=7 is 77520 subsets. k=20 is 1.
                    # This is expensive.
                    # Optimization: If we have no size 6, it's likely valid.
                    # But we must be sure.
                    # We can skip this check for the first few restarts to save time?
                    # No, if we return an invalid graph, score is 0.
                    # So we MUST check.
                    
                    # To speed up, we can break early.
                    for subset in combinations(range(n), k):
                        mask = 0
                        for v in subset:
                            mask |= (1 << v)
                        
                        # Check regularity
                        verts = list(subset)
                        d0 = (adj[verts[0]] & mask).bit_count()
                        regular = True
                        for v in verts[1:]:
                            if (adj[v] & mask).bit_count() != d0:
                                regular = False
                                break
                        if regular:
                            valid = False
                            break
                    if not valid:
                        break
                        
                if valid:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    # Fallback: return a small valid graph
    # n=5, empty graph is regular (0-regular). So n=5 empty is INVALID.
    # We need a graph with NO regular induced subgraph of size 6.
    # If n < 6, the condition "No set of 6 or more vertices..." is vacuously true?
    # "No set of 6 or more vertices may induce a regular subgraph".
    # If n=5, there are no sets of size 6. So it is valid.
    # The prompt says "1 <= n <= 22".
    # So n=5 is a valid graph, but small.
    # We should try to return the largest n we found.
    # If we didn't find any, return n=5.
    
    return {'n': 5, 'edges': []}
