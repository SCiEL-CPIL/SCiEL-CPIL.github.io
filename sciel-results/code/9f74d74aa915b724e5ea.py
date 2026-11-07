import time
import random

P = {'n': 12, 'm': 40}

def construct():
    deadline = time.time() + 85
    n = P['n']
    m = P['m']
    N = 1 << n
    
    # Precompute nothing heavy, just use bitwise ops.
    # However, to speed up the "Gain" calculation, we can precompute
    # the list of all masks that are comparable to a given mask?
    # No, N=4096 is small enough to just loop.
    
    # Greedy Initialization
    # Start with a maximal chain to ensure high initial connectivity
    chain = []
    cur = 0
    for b in range(n):
        chain.append(cur)
        cur |= (1 << b)
    chain.append(cur)
    
    family = chain[:m]
    fam_set = set(family)
    
    # Optimized Local Search using Sub/Sup Counts
    sub_count = [0] * N
    sup_count = [0] * N
    
    def update_counts(mask, delta):
        for M in range(N):
            if (mask & M) == mask:
                sub_count[M] += delta
            if (mask & M) == M:
                sup_count[M] += delta

    for mask in family:
        update_counts(mask, 1)
        
    # Greedily add remaining m - len(family) sets
    while len(family) < m:
        if time.time() > deadline - 5:
            break
            
        best_mask = -1
        best_score = -1
        
        # Find mask M not in family maximizing comparability with current family
        for M in range(N):
            if M in fam_set:
                continue
            
            # Score is number of existing sets comparable to M
            # Since M is not in family, no self-comparison to subtract
            score = sub_count[M] + sup_count[M]
            
            if score > best_score:
                best_score = score
                best_mask = M
        
        if best_mask == -1:
            break
            
        family.append(best_mask)
        fam_set.add(best_mask)
        update_counts(best_mask, 1)
        
    # Local Search (Hill Climbing) with O(N) updates
    improved = True
    passes = 0
    while improved and time.time() < deadline - 2:
        improved = False
        passes += 1
        
        for i in range(m):
            if time.time() > deadline - 2:
                break
                
            old_mask = family[i]
            
            # Calculate Loss of old_mask (comparables with others)
            # sub_count[old_mask] includes old_mask itself, so subtract 1
            # sup_count[old_mask] includes old_mask itself, so subtract 1
            loss = (sub_count[old_mask] - 1) + (sup_count[old_mask] - 1)
            
            best_mask = -1
            best_gain = -1
            
            # Try all candidates M not in family
            for M in range(N):
                if M in fam_set:
                    continue
                
                # Gain is comparables of M with (family \ {old_mask})
                # sub_count[M] counts A in family s.t. A subset M.
                # If old_mask subset M, old_mask is in this count, so subtract 1.
                # sup_count[M] counts A in family s.t. M subset A.
                # If M subset old_mask, old_mask is in this count, so subtract 1.
                
                gain = sub_count[M] + sup_count[M]
                if (old_mask & M) == old_mask:
                    gain -= 1
                elif (old_mask & M) == M:
                    gain -= 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_mask = M
            
            # If we found a better replacement
            if best_mask != -1 and best_gain > loss:
                # Update family and fam_set
                fam_set.remove(old_mask)
                fam_set.add(best_mask)
                family[i] = best_mask
                
                update_counts(old_mask, -1)
                update_counts(best_mask, 1)
                improved = True
                
    # Convert to lists of integers
    result = []
    for mask in family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
