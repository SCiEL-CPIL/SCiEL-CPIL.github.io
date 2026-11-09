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
    """Initialize by building chains: pick a chain of size k, then another disjoint-ish chain, etc."""
    family = []
    used = set()
    max_mask = 1 << n
    
    # Build multiple chains
    for _ in range(m):
        best_set = -1
        best_gain = -1
        candidates = list(range(max_mask))
        random.shuffle(candidates)
        candidates_to_check = candidates[:500]
        if 0 not in candidates_to_check:
            candidates_to_check.append(0)
        if (max_mask - 1) not in candidates_to_check:
            candidates_to_check.append(max_mask - 1)
        
        for s in candidates_to_check:
            if s in used:
                continue
            
            gain = 0
            for t in family:
                if (s & t) == s:
                    gain += 1
                elif (t & s) == t:
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

def _two_opt_local_search(family, n, m, deadline):
    """Vectorized 2-opt: evaluate all 4096 replacements simultaneously per position."""
    max_mask = 1 << n
    current_family = list(family)
    current_fs = set(current_family)
    current_score = _calc_score_bitmask(current_family)
    
    best_family = list(current_family)
    best_score = current_score
    
    # Precompute all masks as numpy array
    all_masks = np.arange(max_mask, dtype=np.int32)
    
    no_improvement_count = 0
    
    while time.time() < deadline:
        improved = False
        best_delta = 0
        best_idx = -1
        best_new_set = -1
        
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_set = current_family[idx]
            
            # Create mask of valid candidates (not already in family, not the old set)
            valid_mask = np.ones(max_mask, dtype=bool)
            valid_mask[old_set] = False
            for s in current_fs:
                if s != old_set:
                    valid_mask[s] = False
            
            # For each candidate, compute comparability with all other sets
            # other_sets is array of the m-1 other sets
            other_sets = np.array([current_family[j] for j in range(m) if j != idx], dtype=np.int32)
            
            # new_set & other == new_set: new_set is subset of other
            # new_set & other == other: other is subset of new_set
            # Vectorize: for each candidate new_set, check against all others
            # Shape: (max_mask, m-1)
            ns = all_masks[:, np.newaxis]  # (4096, 1)
            os_ = other_sets[np.newaxis, :]  # (1, m-1)
            
            # Comparability: (ns & os_) == ns OR (ns & os_) == os_
            # But we need to exclude invalid candidates
            intersect = ns & os_  # (4096, m-1)
            comp1 = (intersect == ns)  # new_set subset of other
            comp2 = (intersect == os_)  # other subset of new_set
            comparable = comp1 | comp2  # (4096, m-1)
            
            # Count comparabilities for each candidate
            gains = comparable.sum(axis=1)  # (4096,)
            
            # Subtract old contributions
            old_intersect = (old_set & other_sets)
            old_comp1 = (old_intersect == old_set)
            old_comp2 = (old_intersect == other_sets)
            old_gain = (old_comp1 | old_comp2).sum()
            
            # Delta = gains - old_gain
            deltas = gains - old_gain
            
            # Apply validity mask
            deltas[~valid_mask] = -np.inf
            
            # Find best
            best_local_delta = np.max(deltas)
            if best_local_delta > best_delta:
                best_delta = int(best_local_delta)
                best_idx = idx
                best_new_set = int(np.argmax(deltas))
        
        if best_delta > 0 and best_idx >= 0:
            # Apply the best move
            old_set = current_family[best_idx]
            current_fs.remove(old_set)
            current_fs.add(best_new_set)
            current_family[best_idx] = best_new_set
            current_score += best_delta
            improved = True
            no_improvement_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(current_family)
        else:
            no_improvement_count += 1
            # If stuck, do a random perturbation
            if no_improvement_count >= 5:
                for _ in range(10):
                    idx = random.randint(0, m - 1)
                    new_set = random.randint(0, max_mask - 1)
                    if new_set in current_fs:
                        continue
                    old_set = current_family[idx]
                    
                    delta = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        other = current_family[j]
                        if (old_set & other) == old_set:
                            delta -= 1
                        elif (old_set & other) == other:
                            delta -= 1
                        if (new_set & other) == new_set:
                            delta += 1
                        elif (new_set & other) == other:
                            delta += 1
                    
                    if delta >= 0 or random.random() < 0.3:
                        current_fs.remove(old_set)
                        current_fs.add(new_set)
                        current_family[idx] = new_set
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_family = list(current_family)
                        break
                no_improvement_count = 0
    
    return best_family, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = P['n']
    m = P['m']
    
    # Initial family
    current_family = _greedy_chain_init(n, m)
    
    # 2-opt local search
    best_family, best_score = _two_opt_local_search(current_family, n, m, deadline)
    
    # Convert back to lists of ints
    result = []
    for s in best_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    return result
# EVOLVE-BLOCK-END
