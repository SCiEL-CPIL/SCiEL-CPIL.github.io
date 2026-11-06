<<<<<<< SEARCH
def _greedy_build(seed):
    """Greedy construction."""
    rng = random.Random(seed)
    S_mask = 0
    blocked = 0
    
    while True:
        candidates_mask = ALL_MASK & ~blocked & ~S_mask
        
        if not candidates_mask:
            break
            
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            m ^= lsb
            
        if not candidates:
            break
            
        # Score candidates
        curr_indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            curr_indices.append(idx)
            m ^= lsb
            
        best_score = -1
        best_candidate = None
        best_new_blocked = 0
        
        rng.shuffle(candidates)
        
        for p in candidates:
            # Check validity
            valid = True
            for idx in curr_indices:
                if S_mask & _C[p][idx]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new blocked
            new_blocked = blocked
            for idx in curr_indices:
                new_blocked |= _C[p][idx]
            
            score = bin(new_blocked & ~blocked & ~S_mask).count('1')
            
            if score > best_score:
                best_score = score
                best_candidate = p
                best_new_blocked = new_blocked
            elif score == best_score and best_candidate is not None:
                if rng.random() < 0.5:
                    best_candidate = p
                    best_new_blocked = new_blocked
                    
        if best_candidate is None:
            break
            
        S_mask |= (1 << best_candidate)
        blocked = best_new_blocked
        
    return S_mask
=======
def _greedy_build(seed):
    """Greedy construction with improved scoring."""
    rng = random.Random(seed)
    S_mask = 0
    blocked = 0
    
    while True:
        candidates_mask = ALL_MASK & ~blocked & ~S_mask
        
        if not candidates_mask:
            break
            
        candidates = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            candidates.append(idx)
            m ^= lsb
            
        if not candidates:
            break
            
        # Score candidates
        curr_indices = []
        m = S_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            curr_indices.append(idx)
            m ^= lsb
        
        best_score = -1
        best_candidate = None
        best_new_blocked = 0
        
        rng.shuffle(candidates)
        
        for p in candidates:
            # Check validity
            valid = True
            for idx in curr_indices:
                if S_mask & _C[p][idx]:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Compute new blocked
            new_blocked = blocked
            for idx in curr_indices:
                new_blocked |= _C[p][idx]
            
            # Primary score: newly blocked points (blocking power)
            newly_blocked = new_blocked & ~blocked & ~S_mask
            score = bin(newly_blocked).count('1')
            
            # Secondary: prefer points that are "far" from existing set (diversity)
            # This helps avoid clustering and may lead to smaller maximal sets
            if curr_indices:
                min_dist = float('inf')
                for idx in curr_indices:
                    d = _D[p, idx]
                    if d < min_dist:
                        min_dist = d
                score += 0.1 * min_dist  # Small bonus for diversity
            
            if score > best_score:
                best_score = score
                best_candidate = p
                best_new_blocked = new_blocked
            elif score == best_score and best_candidate is not None:
                if rng.random() < 0.5:
                    best_candidate = p
                    best_new_blocked = new_blocked
                    
        if best_candidate is None:
            break
            
        S_mask |= (1 << best_candidate)
        blocked = best_new_blocked
        
    return S_mask
>>>>>>> REPLACE

