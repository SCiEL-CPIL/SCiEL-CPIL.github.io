"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 12
    m = 40
    N = 1 << n # 4096
    
    # Precompute comparability sets
    # comps[i] is a set of all j != i such that i and j are comparable
    comps = [set() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                comps[i].add(j)

    # Greedy Construction
    # Start with 0 and N-1
    family = [0, N - 1]
    fam_set = set(family)
    
    # S[idx] = number of comparable sets in family for family[idx]
    # For 0 and N-1, they are comparable to each other.
    S = [1, 1]
    
    # Add m-2 more sets
    for _ in range(m - 2):
        if time.time() > deadline:
            break
            
        best_mask = -1
        best_score = -1
        
        # Find mask M not in family maximizing S_M
        # S_M = sum(1 for f in family if f in comps[M])
        
        for M in range(N):
            if M in fam_set:
                continue
            
            s_m = 0
            for f in family:
                if f in comps[M]:
                    s_m += 1
            
            if s_m > best_score:
                best_score = s_m
                best_mask = M
        
        if best_mask == -1:
            break
            
        family.append(best_mask)
        fam_set.add(best_mask)
        
        # Update S for existing elements
        for idx in range(len(family) - 1):
            if best_mask in comps[family[idx]]:
                S[idx] += 1
        
        S.append(best_score)
        
        if time.time() > deadline:
            break

    # Local Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        fam_set = set(family)
        
        for i in range(m):
            if time.time() > deadline:
                break
                
            old_mask = family[i]
            old_S = S[i]
            
            best_mask = -1
            best_s_m = -1
            
            # Find M not in family maximizing S_M_excl
            # S_M_excl = sum(1 for f in family if f != old_mask and f in comps[M])
            
            for M in range(N):
                if M in fam_set:
                    continue
                
                s_m = 0
                for j in range(m):
                    if j == i:
                        continue
                    if family[j] in comps[M]:
                        s_m += 1
                
                if s_m > best_s_m:
                    best_s_m = s_m
                    best_mask = M
            
            if best_mask != -1 and best_s_m > old_S:
                # Perform swap
                for j in range(m):
                    if j == i:
                        continue
                    delta_j = 0
                    if best_mask in comps[family[j]]:
                        delta_j += 1
                    if old_mask in comps[family[j]]:
                        delta_j -= 1
                    S[j] += delta_j
                
                S[i] = best_s_m
                
                fam_set.remove(old_mask)
                fam_set.add(best_mask)
                family[i] = best_mask
                
                improved = True

    # Convert to lists of integers
    result = []
    for mask in family:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    return result
# EVOLVE-BLOCK-END
