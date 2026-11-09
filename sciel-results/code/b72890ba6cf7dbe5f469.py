"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

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
    # A chain is a sequence of sets where each is a strict subset of the next
    # We want to fill m slots with sets that form as many comparable pairs as possible
    
    # Strategy: create several chains of varying lengths
    # For n=12, max chain length is 13 (empty set to full set)
    # If we make one chain of length 13, that gives C(13,2)=78 comparable pairs
    # But we need 40 sets. Multiple chains can share elements but sets must be distinct.
    
    # Better: greedily add sets that maximize gain, but with a bias toward chain structures
    
    for _ in range(m):
        best_set = -1
        best_gain = -1
        # Sample a subset of candidates to speed up
        candidates = list(range(max_mask))
        random.shuffle(candidates)
        # Only check first 500 candidates for speed, plus always check 0 and max_mask-1
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
            elif gain == best_gain and best_set != -1:
                # Tie-break: prefer sets that are "middle" sized (more room to extend chain)
                pass

        if best_set != -1:
            family.append(best_set)
            used.add(best_set)
        else:
            # Fallback: find any unused set
            for s in range(max_mask):
                if s not in used:
                    family.append(s)
                    used.add(s)
                    break
    
    return family

def _two_opt_local_search(family, n, m, deadline):
    """2-opt: try replacing each set with every other possible set, pick best improving move."""
    max_mask = 1 << n
    current_family = list(family)
    current_fs = set(current_family)
    current_score = _calc_score_bitmask(current_family)
    
    best_family = list(current_family)
    best_score = current_score
    
    # Precompute all masks
    all_masks = list(range(max_mask))
    
    no_improvement_count = 0
    
    while time.time() < deadline:
        improved = False
        best_delta = 0
        best_idx = -1
        best_new_set = -1
        
        # Try each position
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_set = current_family[idx]
            
            # Try candidate replacements
            # To speed up, don't try all 4096; try a smart subset
            # But for correctness, let's try all but break early if time runs out
            for new_set in all_masks:
                if new_set == old_set or new_set in current_fs:
                    continue
                
                # Calculate delta
                delta = 0
                for j in range(m):
                    if j == idx:
                        continue
                    other = current_family[j]
                    
                    # Old contribution
                    if (old_set & other) == old_set:
                        delta -= 1
                    elif (old_set & other) == other:
                        delta -= 1
                    
                    # New contribution
                    if (new_set & other) == new_set:
                        delta += 1
                    elif (new_set & other) == other:
                        delta += 1
                
                if delta > best_delta:
                    best_delta = delta
                    best_idx = idx
                    best_new_set = new_set
            
            if time.time() >= deadline:
                break
        
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
                # Random restart: pick a random set and replace with random other
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
