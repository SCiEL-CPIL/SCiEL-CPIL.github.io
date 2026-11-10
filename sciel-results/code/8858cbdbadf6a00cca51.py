import time
import numpy as np

N = 256

# Precomputation
partner_masks = [[] for _ in range(N + 1)]
conf_masks = [[] for _ in range(N + 1)]

def _precompute():
    # 3-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                mask_bc = (1 << (b - 1)) | (1 << (c - 1))
                mask_ac = (1 << (a - 1)) | (1 << (c - 1))
                mask_ab = (1 << (a - 1)) | (1 << (b - 1))
                partner_masks[a].append(mask_bc)
                partner_masks[b].append(mask_ac)
                partner_masks[c].append(mask_ab)
    
    # 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            if dd <= N:
                mask_bcd = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (dd - 1))
                mask_acd = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (dd - 1))
                mask_abd = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (dd - 1))
                mask_abc = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
                
                conf_masks[a].append(mask_bcd)
                conf_masks[b].append(mask_acd)
                conf_masks[c].append(mask_abd)
                conf_masks[dd].append(mask_abc)

_precompute()

def _is_valid(mask, x):
    """Check if x can be added to set represented by mask without creating a 4-AP."""
    for m in conf_masks[x]:
        if (mask & m) == m:
            return False
    return True

def _calc_score(mask):
    """Calculate number of 3-APs in set represented by mask."""
    count = 0
    m = mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        x = idx + 1
        # Count partners of x in mask
        for p_mask in partner_masks[x]:
            if (mask & p_mask) == p_mask:
                count += 1
        m ^= lsb
    return count // 3

def _greedy_construct(seed):
    rng = np.random.default_rng(seed)
    mask = 0
    order = rng.permutation(N) + 1
    for x in order:
        if _is_valid(mask, x):
            mask |= (1 << (x - 1))
    return mask

def _local_search(mask, deadline):
    current_score = _calc_score(mask)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try adding elements
        for x in range(1, N + 1):
            if (mask >> (x - 1)) & 1:
                continue
            if _is_valid(mask, x):
                mask |= (1 << (x - 1))
                current_score = _calc_score(mask)
                improved = True
                break # Restart to check new additions
        
        if improved:
            continue

        # 2. Try swapping elements
        # Get list of elements in mask
        elements = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            elements.append(idx + 1)
            m ^= lsb
            
        for y in elements:
            if time.time() > deadline:
                break
            temp_mask = mask & ~(1 << (y - 1))
            # Try to add x to temp_mask
            for x in range(1, N + 1):
                if (temp_mask >> (x - 1)) & 1:
                    continue
                if _is_valid(temp_mask, x):
                    cand_mask = temp_mask | (1 << (x - 1))
                    cand_score = _calc_score(cand_mask)
                    if cand_score > current_score:
                        mask = cand_mask
                        current_score = cand_score
                        improved = True
                        break
            if improved:
                break
                
    return mask, current_score

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin
    
    best_mask = 0
    best_score = -1
    
    # Multiple restarts
    for seed in range(200):
        if time.time() > deadline - 5:
            break
            
        mask = _greedy_construct(seed)
        score = _calc_score(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask
            
        # Local search on this candidate
        if time.time() < deadline - 10:
            mask, score = _local_search(mask, deadline - 5)
            
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask == 0:
        return [1]
        
    # Convert mask to list
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(idx + 1)
        m ^= lsb
        
    return result
