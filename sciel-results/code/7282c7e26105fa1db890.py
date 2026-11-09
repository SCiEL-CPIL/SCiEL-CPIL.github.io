"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
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
            if (si & sj) == si:
                count += 1
            elif (si & sj) == sj:
                count += 1
    return count

def _build_initial_family(n, m):
    """Build initial family using chain decomposition."""
    max_mask = 1 << n
    
    # Precompute all masks by popcount for chain construction
    masks_by_popcount = [[] for _ in range(n+1)]
    for mask in range(max_mask):
        pc = bin(mask).count('1')
        masks_by_popcount[pc].append(mask)
    
    family = []
    used = set()
    
    # Strategy: Build chains greedily
    # A chain is a sequence where each set is a strict subset of the next
    # For n=12, max chain length is 13
    # We want to pack 40 sets into as few chains as possible
    
    # Try to build chains of length 13, 13, 14 (but 14>13, so 13,13,13,1)
    # Actually, let's just greedily add sets that maximize comparability
    
    # Phase 1: Build a backbone chain
    # Start with empty set, then add elements one by one
    backbone = []
    current = 0
    backbone.append(current)
    used.add(current)
    
    # Greedily extend: at each step, add the element that maximizes future comparability
    # For simplicity, just add elements in order to create a chain
    for i in range(n):
        current = current | (1 << i)
        backbone.append(current)
        used.add(current)
    
    # Now we have a chain of length n+1 = 13
    # We need 40-13 = 27 more sets
    # Add sets that are comparable with as many existing sets as possible
    
    for _ in range(m - len(backbone)):
        best_set = -1
        best_gain = -1
        
        # Try all unused masks, but sample for speed
        candidates = []
        for pc in range(n+1):
            for mask in masks_by_popcount[pc]:
                if mask not in used:
                    candidates.append(mask)
        
        if not candidates:
            break
        
        # Sample up to 500 candidates
        if len(candidates) > 500:
            candidates = random.sample(candidates, 500)
        
        for s in candidates:
            gain = 0
            for t in family:
                if (s & t) == s and s != t:
                    gain += 1
                elif (t & s) == t and t != s:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_set = s
        
        if best_set != -1:
            family.append(best_set)
            used.add(best_set)
        else:
            # Fallback
            for s in range(max_mask):
                if s not in used:
                    family.append(s)
                    used.add(s)
                    break
    
    return family

def _local_search(family, n, m, deadline):
    """Vectorized 2-opt local search."""
    max_mask = 1 << n
    current_family = np.array(family, dtype=np.int32)
    current_fs = set(current_family.tolist())
    current_score = _calc_score_bitmask(current_family.tolist())
    
    best_family = list(current_family)
    best_score = current_score
    
    # Precompute all masks as numpy array
    all_masks = np.arange(max_mask, dtype=np.int32)
    
    no_improvement_count = 0
    max_no_improve = 8
    
    while time.time() < deadline:
        best_delta = 0
        best_idx = -1
        best_new_set = -1
        
        # Try each position
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_set = int(current_family[idx])
            
            # Compute old comparabilities for old_set with all others
            old_comps = 0
            for j in range(m):
                if j == idx:
                    continue
                other = int(current_family[j])
                if (old_set & other) == old_set and old_set != other:
                    old_comps += 1
                elif (old_set & other) == other and old_set != other:
                    old_comps += 1
            
            # Vectorized evaluation of all 4096 candidates
            others = np.delete(current_family, idx)  # shape (m-1,)
            
            # (4096, 39) broadcast
            cand = all_masks[:, np.newaxis]  # (4096, 1)
            oth = others[np.newaxis, :]       # (1, 39)
            
            # new_set subset of other: (cand & oth) == cand
            cand_subset_oth = (cand & oth) == cand  # (4096, 39)
            
            # other subset of new_set: (oth & cand) == oth
            oth_subset_cand = (oth & cand) == oth   # (4096, 39)
            
            # Exclude equality
            eq_mask = cand == oth  # (4096, 39)
            
            strict_cand_subset = cand_subset_oth & ~eq_mask
            strict_oth_subset = oth_subset_cand & ~eq_mask
            
            # Count comparabilities for each candidate
            new_comps = strict_cand_subset.sum(axis=1) + strict_oth_subset.sum(axis=1)  # (4096,)
            
            # Delta for each candidate
            deltas = new_comps - old_comps  # (4096,)
            
            # Mask out invalid candidates
            valid_mask = np.ones(max_mask, dtype=bool)
            for s in current_fs:
                valid_mask[s] = False
            
            deltas[~valid_mask] = -999999
            
            # Find best
            max_idx = np.argmax(deltas)
            max_delta = int(deltas[max_idx])
            
            if max_delta > best_delta:
                best_delta = max_delta
                best_idx = idx
                best_new_set = int(all_masks[max_idx])
        
        if best_delta > 0 and best_idx >= 0:
            # Apply the best move
            old_set = int(current_family[best_idx])
            current_fs.remove(old_set)
            current_fs.add(best_new_set)
            current_family[best_idx] = best_new_set
            current_score += best_delta
            no_improvement_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_family = current_family.tolist()
        else:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improve:
                # Perturbation: randomly replace 3 sets
                num_perturb = random.randint(3, 5)
                for _ in range(num_perturb):
                    if time.time() >= deadline:
                        break
                    idx = random.randint(0, m - 1)
                    old_set = int(current_family[idx])
                    
                    # Find a good replacement
                    best_new = -1
                    best_delta_perturb = -999999
                    
                    # Try random candidates
                    for _ in range(300):
                        new_set = random.randint(0, max_mask - 1)
                        if new_set in current_fs:
                            continue
                        
                        delta = 0
                        for j in range(m):
                            if j == idx:
                                continue
                            other = int(current_family[j])
                            if (old_set & other) == old_set and old_set != other:
                                delta -= 1
                            elif (old_set & other) == other and old_set != other:
                                delta -= 1
                            if (new_set & other) == new_set and new_set != other:
                                delta += 1
                            elif (new_set & other) == other and new_set != other:
                                delta += 1
                        
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
                            best_family = current_family.tolist()
                
                no_improvement_count = 0
    
    return best_family, best_score

def _polish(family, n, m, deadline):
    """Final polish: multiple passes of local search with tighter convergence."""
    current_family = list(family)
    current_score = _calc_score_bitmask(current_family)
    
    best_family = list(current_family)
    best_score = current_score
    
    # Multiple short local search passes
    passes = 3
    pass_time = (deadline - time.time()) / passes
    
    for p in range(passes):
        pass_deadline = time.time() + pass_time
        if time.time() >= pass_deadline:
            break
        
        family, score = _local_search(current_family, n, m, pass_deadline)
        
        if score > best_score:
            best_score = score
            best_family = family
            current_family = family
    
    return best_family, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    best_family_overall = None
    best_score_overall = -1
    
    # Multiple restarts with different random seeds
    num_restarts = 4
    restart_time = (deadline - start_time) / (num_restarts + 1)
    
    for restart in range(num_restarts):
        restart_deadline = start_time + (restart + 1) * restart_time
        if time.time() >= restart_deadline:
            break
        
        random.seed(restart * 7919 + 123)
        
        # Build initial family
        current_family = _build_initial_family(n, m)
        
        # Local search
        family, score = _local_search(current_family, n, m, restart_deadline)
        
        if score > best_score_overall:
            best_score_overall = score
            best_family_overall = family
    
    # Final polish with remaining time
    if best_family_overall is not None and time.time() < deadline - 5:
        family, score = _polish(best_family_overall, n, m, deadline)
        if score > best_score_overall:
            best_family_overall = family
            best_score_overall = score
    
    if best_family_overall is None:
        best_family_overall = _build_initial_family(n, m)
    
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
