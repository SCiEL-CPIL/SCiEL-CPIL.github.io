import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def count_comparables(family_masks):
        """Count comparable pairs in a list of masks."""
        cnt = 0
        L = len(family_masks)
        for i in range(L):
            ai = family_masks[i]
            for j in range(i + 1, L):
                bj = family_masks[j]
                if (ai & ~bj) == 0 or (bj & ~ai) == 0:
                    cnt += 1
        return cnt

    def greedy_build(init_family):
        """Greedily add sets that maximize comparable pairs with current family."""
        family = init_family[:]
        fam_set = set(family)
        
        while len(family) < m:
            if time.time() > deadline:
                break
            best_gain = -1
            best_mask = None
            
            for s in range(total_masks):
                if s in fam_set:
                    continue
                gain = 0
                for f in family:
                    if (s & ~f) == 0 or (f & ~s) == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_mask = s
            
            if best_mask is None:
                break
            family.append(best_mask)
            fam_set.add(best_mask)
        
        return family

    def local_search(family, max_passes=10):
        """Local search: try replacing each element with best candidate."""
        best_family = family[:]
        best_score = count_comparables(best_family)
        fam_set = set(best_family)
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(len(best_family)):
                if time.time() > deadline:
                    break
                
                old_mask = best_family[i]
                
                # Calculate old comparability count (excluding self)
                old_comp = 0
                for j in range(len(best_family)):
                    if j == i:
                        continue
                    if (old_mask & ~best_family[j]) == 0 or (best_family[j] & ~old_mask) == 0:
                        old_comp += 1
                
                # Find best replacement
                best_gain = -1
                best_repl = None
                
                for s in range(total_masks):
                    if s in fam_set:
                        continue
                    
                    new_comp = 0
                    for j in range(len(best_family)):
                        if j == i:
                            continue
                        if (s & ~best_family[j]) == 0 or (best_family[j] & ~s) == 0:
                            new_comp += 1
                    
                    delta = new_comp - old_comp
                    if delta > best_gain:
                        best_gain = delta
                        best_repl = s
                
                if best_repl is not None and best_gain > 0:
                    best_family[i] = best_repl
                    fam_set.discard(old_mask)
                    fam_set.add(best_repl)
                    improved = True
            
            if not improved:
                break
        
        return best_family, count_comparables(best_family)

    # Strategy 1: Single chain initialization (length 13)
    chain1 = []
    cur = 0
    chain1.append(0)
    for i in range(n):
        cur |= (1 << i)
        chain1.append(cur)
    
    # Strategy 2: Multi-chain initialization
    # Two parallel chains on disjoint element sets
    chain_a = []
    cur_a = 0
    chain_a.append(0)  # empty set
    for i in range(6):
        cur_a |= (1 << i)
        chain_a.append(cur_a)
    
    chain_b = []
    cur_b = 0
    for i in range(6, 12):
        cur_b |= (1 << i)
        chain_b.append(cur_b)
    
    full_set = (1 << 12) - 1
    multi_init = list(set(chain_a + chain_b + [full_set]))
    
    # Greedily fill both strategies
    fam1 = greedy_build(chain1)
    score1 = count_comparables(fam1)
    
    fam2 = greedy_build(multi_init)
    score2 = count_comparables(fam2)
    
    # Pick better initialization
    if score1 >= score2:
        best_fam = fam1
    else:
        best_fam = fam2
    
    # Local search
    best_fam, best_score = local_search(best_fam, max_passes=15)
    
    # Try random perturbations and re-local-search
    for restart in range(5):
        if time.time() > deadline:
            break
        
        perturbed = best_fam[:]
        fam_set = set(perturbed)
        for _ in range(random.randint(3, 5)):
            idx = random.randint(0, len(perturbed) - 1)
            candidates = [s for s in range(total_masks) if s not in fam_set]
            if not candidates:
                break
            repl = random.choice(candidates)
            fam_set.discard(perturbed[idx])
            perturbed[idx] = repl
            fam_set.add(repl)
        
        perturbed, sc = local_search(perturbed, max_passes=8)
        if sc > best_score:
            best_fam = perturbed
            best_score = sc
    
    # Final local search pass
    best_fam, best_score = local_search(best_fam, max_passes=10)
    
    # Convert to list of lists
    result = []
    for mask in best_fam:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    return result
