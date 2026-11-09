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
    """Initialize by building multiple chains to maximize comparable pairs."""
    family = []
    used = set()
    max_mask = 1 << n
    
    # Strategy: Build multiple chains of different lengths
    # For n=12, we can have chains of various sizes
    # A chain of length k gives C(k,2) comparable pairs
    # We want to partition 40 into chain lengths to maximize total comparable pairs
    
    # Optimal: use as many long chains as possible
    # Max chain length is 13 (sizes 0 to 12)
    # 40 = 13 + 13 + 14? No, max is 13. So 13+13+14 doesn't work.
    # 13+13+13 = 39, then 1 more. Or 12+12+12+4, etc.
    
    # Actually, let's just greedily build chains
    # Start with empty set, then add sets that are supersets of existing ones
    
    # Better approach: create several "levels" and pick sets from each level
    # Level k has C(12,k) sets
    # We want to pick sets such that many pairs are comparable
    
    # Simplest effective strategy: create 3-4 chains
    chains = []
    remaining = m
    
    # Create chains of length ~13, 13, 13, 1
    chain_lengths = []
    rem = m
    while rem > 0:
        if rem >= 13:
            chain_lengths.append(13)
            rem -= 13
        elif rem >= 12:
            chain_lengths.append(12)
            rem -= 12
        else:
            chain_lengths.append(rem)
            rem = 0
    
    # For each chain, pick a random starting point and build up/down
    for cl in chain_lengths:
        # Pick a random "spine" - a sequence of sets where each adds one element
        # Start with a random subset, then add elements one by one
        start_size = random.randint(0, max(0, n - cl))
        
        # Generate a random ordering of elements
        elements = list(range(n))
        random.shuffle(elements)
        
        # Start with first start_size elements
        current_mask = 0
        for i in range(start_size):
            current_mask |= (1 << elements[i])
        
        chain = [current_mask]
        
        # Add elements one by one
        for i in range(start_size, min(start_size + cl - 1, n)):
            current_mask |= (1 << elements[i])
            chain.append(current_mask)
        
        # If we need more, we can also go down from the top
        # But for simplicity, just use what we have
        # Pad with random sets if needed
        while len(chain) < cl:
            # Pick a random set that's comparable to at least one in the chain
            s = random.randint(0, max_mask - 1)
            if s not in used:
                chain.append(s)
        
        for s in chain:
            if s not in used:
                family.append(s)
                used.add(s)
    
    # Fill remaining slots with greedy selection
    while len(family) < m:
        best_set = -1
        best_gain = -1
        
        # Try all masks
        for s in range(max_mask):
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
            # Fallback
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
