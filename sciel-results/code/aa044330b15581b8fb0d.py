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

def _greedy_chain_init(n, m):
    """Initialize by building chains to maximize comparability."""
    family = []
    used = set()
    max_mask = 1 << n
    
    # Build multiple chains
    # A chain is a sequence where each set is a strict subset of the next
    # We want to fill all 40 slots with sets that form as many chains as possible
    
    # Strategy: build chains of varying lengths
    # Each chain contributes C(k,2) comparabilities for k elements
    # We want to maximize total comparabilities across all chains
    
    # First, let's try to build long chains
    # Start with small sets and grow them
    
    # Precompute: for each mask, which masks are its immediate supersets?
    # Actually, let's just do greedy: repeatedly pick the set that maximizes gain
    
    for _ in range(m):
        best_set = -1
        best_gain = -1
        
        # Candidate: try to extend existing chains
        # For each existing set, try adding a subset or superset
        # Also try random candidates
        
        candidates = set()
        
        # For each existing set, find potential subsets and supersets
        for t in family:
            # Find all proper subsets of t that aren't used
            sub = t
            while sub > 0:
                candidates.add(sub)
                sub = (sub - 1) & t
            # Also 0 is a subset of everything
            candidates.add(0)
            
            # Find all proper supersets of t
            # Supersets of t are masks s where (s & t) == t and s != t
            # This is expensive to enumerate, so let's add some random ones
            for _ in range(5):
                # Random superset: add random bits to t
                extra = random.randint(1, max_mask - 1)
                candidates.add(t | extra)
        
        # Add some random candidates
        for _ in range(100):
            candidates.add(random.randint(0, max_mask - 1))
        
        # Ensure we have empty and full set
        candidates.add(0)
        candidates.add(max_mask - 1)
        
        # Evaluate each candidate
        for s in candidates:
            if s in used:
                continue
            
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
            for s in range(max_mask):
                if s not in used:
                    family.append(s)
                    used.add(s)
                    break
    
    return family

def _local_search(family, n, m, deadline):
    """Local search with vectorized delta computation."""
    max_mask = 1 << n
    current_family = np.array(family, dtype=np.int32)
    current_fs = set(current_family.tolist())
    current_score = _calc_score_bitmask(current_family.tolist())
    
    best_family = list(current_family)
    best_score = current_score
    
    # Precompute all masks as numpy array
    all_masks = np.arange(max_mask, dtype=np.int32)
    
    # Precompute comparability matrix for speed
    # comp[i][j] = 1 if family[i] and family[j] are comparable (strict subset)
    
    no_improvement_count = 0
    max_no_improve = 5
    
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
                # Perturbation: randomly replace 2-3 sets
                num_perturb = random.randint(2, 3)
                for _ in range(num_perturb):
                    if time.time() >= deadline:
                        break
                    idx = random.randint(0, m - 1)
                    old_set = int(current_family[idx])
                    
                    # Find a good replacement
                    best_new = -1
                    best_delta_perturb = -999999
                    
                    # Try random candidates
                    for _ in range(200):
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

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    best_family_overall = None
    best_score_overall = -1
    
    # Multiple restarts
    num_restarts = 3
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
    if best_family_overall is not None and time.time() < deadline - 5:
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
