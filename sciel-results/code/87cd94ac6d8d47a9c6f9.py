"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

P = {'n': 12, 'm': 40}

def _calc_score_bitmask(family):
    """Calculate score for a family of bitmasks."""
    m = len(family)
    count = 0
    for i in range(m):
        si = family[i]
        for j in range(i+1, m):
            sj = family[j]
            if (si & sj) == si and si != sj:
                count += 1
            elif (si & sj) == sj and si != sj:
                count += 1
    return count

def _greedy_chain_init(n, m):
    """Initialize by building chains to maximize comparability."""
    family = []
    used = set()
    max_mask = 1 << n
    
    # Precompute all masks
    all_masks = list(range(max_mask))
    
    # Strategy: Greedily add the set that maximizes comparability with existing sets.
    # To avoid getting stuck in a local minimum of just one chain, 
    # we can bias towards sets of varying sizes.
    
    # Start with a few random sets to seed the structure, then greedy.
    # Or just pure greedy from empty.
    
    for _ in range(m):
        best_set = -1
        best_gain = -1
        
        # To speed up, we can just iterate all 4096.
        # 40 * 4096 * 40 is ~6.5M ops, which is fine for init.
        
        for s in all_masks:
            if s in used:
                continue
            
            gain = 0
            for t in family:
                if (s & t) == s and s != t:
                    gain += 1
                elif (s & t) == t and s != t:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_set = s
        
        if best_set != -1:
            family.append(best_set)
            used.add(best_set)
        else:
            # Fallback
            for s in all_masks:
                if s not in used:
                    family.append(s)
                    used.add(s)
                    break
                    
    return family

def _local_search(family, n, m, deadline):
    """Vectorized local search."""
    max_mask = 1 << n
    current_family = np.array(family, dtype=np.int32)
    current_fs = set(current_family.tolist())
    
    # Precompute all masks
    all_masks = np.arange(max_mask, dtype=np.int32)
    
    current_score = _calc_score_bitmask(current_family.tolist())
    best_family = current_family.copy()
    best_score = current_score
    
    # Precompute comparability matrix for the current family? 
    # No, we recompute gains for replacements.
    
    no_improvement_count = 0
    max_no_improve = 10 # Allow more iterations
    
    # We will iterate through positions. 
    # For each position, find the best replacement from all 4096 masks.
    
    while time.time() < deadline:
        best_delta_global = 0
        best_idx_global = -1
        best_new_set_global = -1
        
        for idx in range(m):
            if time.time() >= deadline:
                break
            
            old_set = int(current_family[idx])
            
            # Compute old comparabilities with others
            # We can just compute the gain of the new set and subtract the gain of the old set?
            # No, the gain of the new set depends on the others.
            
            others = np.delete(current_family, idx) # shape (m-1,)
            
            # Vectorized evaluation of all 4096 candidates against 'others'
            # cand: (4096, 1)
            # oth: (1, 39)
            cand = all_masks[:, np.newaxis]
            oth = others[np.newaxis, :]
            
            # Check if cand is subset of oth
            # (cand & oth) == cand
            cand_subset_oth = (cand & oth) == cand
            
            # Check if oth is subset of cand
            # (oth & cand) == oth
            oth_subset_cand = (oth & cand) == oth
            
            # Strict subset: exclude equality
            eq_mask = cand == oth
            
            strict_cand_subset = cand_subset_oth & ~eq_mask
            strict_oth_subset = oth_subset_cand & ~eq_mask
            
            # Count comparabilities for each candidate with the others
            new_comps = strict_cand_subset.sum(axis=1) + strict_oth_subset.sum(axis=1)
            
            # We need to subtract the comparabilities that the OLD set had with the others.
            # old_comps is a scalar.
            # Let's compute old_comps efficiently.
            # Actually, we can just compute the delta directly if we know old_comps.
            
            # Compute old_comps
            # old_set vs others
            # (old_set & others) == old_set OR (old_set & others) == others
            # Exclude equality
            old_cand_subset = (old_set & others) == old_set
            old_oth_subset = (old_set & others) == others
            old_eq = others == old_set
            old_comps = np.sum((old_cand_subset | old_oth_subset) & ~old_eq)
            
            deltas = new_comps - old_comps
            
            # Mask out invalid candidates (already in family)
            valid_mask = np.ones(max_mask, dtype=bool)
            for s in current_fs:
                valid_mask[s] = False
            
            deltas[~valid_mask] = -999999
            
            max_idx = np.argmax(deltas)
            max_delta = int(deltas[max_idx])
            
            if max_delta > best_delta_global:
                best_delta_global = max_delta
                best_idx_global = idx
                best_new_set_global = int(all_masks[max_idx])
        
        if best_delta_global > 0 and best_idx_global >= 0:
            # Apply the best move
            old_set = int(current_family[best_idx_global])
            current_fs.remove(old_set)
            current_fs.add(best_new_set_global)
            current_family[best_idx_global] = best_new_set_global
            current_score += best_delta_global
            no_improvement_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_family = current_family.copy()
        else:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improve:
                # Perturbation: randomly replace 1-2 sets
                num_perturb = random.randint(1, 2)
                for _ in range(num_perturb):
                    if time.time() >= deadline:
                        break
                    idx = random.randint(0, m - 1)
                    old_set = int(current_family[idx])
                    
                    # Pick a random replacement
                    # To be smarter, pick from a random subset of all masks
                    candidates = random.sample(range(max_mask), 100)
                    best_new = -1
                    best_delta_perturb = -999999
                    
                    for new_set in candidates:
                        if new_set in current_fs:
                            continue
                        
                        # Calculate delta
                        others = np.delete(current_family, idx)
                        # Vectorized delta for this specific new_set
                        # new_set vs others
                        ns = new_set
                        oc1 = (ns & others) == ns
                        oc2 = (ns & others) == others
                        oe = others == ns
                        new_comps = np.sum((oc1 | oc2) & ~oe)
                        
                        old_c1 = (old_set & others) == old_set
                        old_c2 = (old_set & others) == others
                        old_e = others == old_set
                        old_comps = np.sum((old_c1 | old_c2) & ~old_e)
                        
                        delta = new_comps - old_comps
                        if delta > best_delta_perturb:
                            best_delta_perturb = delta
                            best_new = new_set
                    
                    if best_new >= 0:
                        current_fs.remove(old_set)
                        current_fs.add(best_new)
                        current_family[idx] = best_new
                        current_score += best_delta_perturb
                        if current_score > best_score:
                            best_score = current_score
                            best_family = current_family.copy()
                
                no_improvement_count = 0
                
    return best_family, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    best_family_overall = None
    best_score_overall = -1
    
    # Multiple restarts
    num_restarts = 5
    restart_time = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + (restart + 1) * restart_time
        if time.time() >= restart_deadline:
            break
        
        # Initial family with some randomness
        random.seed(restart * 12345 + 678)
        current_family = _greedy_chain_init(n, m)
        
        # Local search
        family, score = _local_search(current_family, n, m, restart_deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_family_overall = family
    
    # Final polish with remaining time
    if best_family_overall is not None and time.time() < deadline - 2:
        family, score = _local_search(best_family_overall, n, m, deadline)
        if score > best_score_overall:
            best_family_overall = family
            best_score_overall = score
    
    if best_family_overall is None:
        best_family_overall = _greedy_chain_init(n, m)
    
    # Convert back to lists of ints
    result = []
    for s in best_family_overall:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    return result
# EVOLVE-BLOCK-END
