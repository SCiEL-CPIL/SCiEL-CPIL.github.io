"""Optimized search for 3-AP-rich 4-AP-free subsets of [1,256]."""
import time
import random
import numpy as np

N = 256
DEADLINE = 85.0  # seconds; leave margin under 90s kill

def _count_3aps(mask):
    """Count distinct 3-term APs with positive difference in a boolean mask of length N+1 (index 0 unused)."""
    count = 0
    # For each difference d, count pairs (a, a+2d) where a, a+d, a+2d are all in set
    for d in range(1, (N + 1) // 2 + 1):
        # a ranges from 1 to N-2d
        # We need mask[a] & mask[a+d] & mask[a+2d]
        max_a = N - 2 * d
        if max_a < 1:
            break
        # Vectorized: mask[1:max_a+1] * mask[1+d:max_a+d+1] * mask[1+2*d:max_a+2*d+1]
        count += np.sum(mask[1:max_a+1] * mask[1+d:max_a+d+1] * mask[1+2*d:max_a+2*d+1])
    return count

def _has_4ap(mask, new_elem=None):
    """Check if adding new_elem (or the current mask) contains a nonconstant 4-AP.
    For efficiency during search, we check if the mask as-is has any 4-AP."""
    # Check all 4-APs: a, a+d, a+2d, a+3d all in mask
    for d in range(1, (N + 1) // 3 + 1):
        max_a = N - 3 * d
        if max_a < 1:
            break
        # mask[a] & mask[a+d] & mask[a+2d] & mask[a+3d]
        vals = mask[1:max_a+1] & mask[1+d:max_a+d+1] & mask[1+2*d:max_a+2*d+1] & mask[1+3*d:max_a+3*d+1]
        if np.any(vals):
            return True
    return False

def _add_4ap_conflicts(mask, candidate):
    """Check if adding 'candidate' to mask creates any 4-AP.
    A 4-AP involving candidate: candidate can be at position 0,1,2,3 of the AP.
    Returns True if conflict exists."""
    c = candidate
    for d in range(1, (N + 1) // 3 + 1):
        # candidate as a (first term): a=c, need c+d, c+2d, c+3d in mask
        if c + 3*d <= N:
            if mask[c+d] and mask[c+2*d] and mask[c+3*d]:
                return True
        # candidate as a+d (second term): a=c-d, need c-d, c+d, c+2d in mask
        if c - d >= 1 and c + 2*d <= N:
            if mask[c-d] and mask[c+d] and mask[c+2*d]:
                return True
        # candidate as a+2d (third term): a=c-2d, need c-2d, c-d, c+d in mask
        if c - 2*d >= 1 and c + d <= N:
            if mask[c-2*d] and mask[c-d] and mask[c+d]:
                return True
        # candidate as a+3d (fourth term): a=c-3d, need c-3d, c-2d, c-d in mask
        if c - 3*d >= 1:
            if mask[c-3*d] and mask[c-2*d] and mask[c-d]:
                return True
    return False

def _greedy_forward():
    """Forward greedy: add elements in order if no 4-AP is created."""
    mask = np.zeros(N+1, dtype=bool)
    for x in range(1, N+1):
        if not _add_4ap_conflicts(mask, x):
            mask[x] = True
    return mask

def _greedy_random_order(seed):
    """Forward greedy with random element order."""
    rng = random.Random(seed)
    order = list(range(1, N+1))
    rng.shuffle(order)
    mask = np.zeros(N+1, dtype=bool)
    for x in order:
        if not _add_4ap_conflicts(mask, x):
            mask[x] = True
    return mask

def _local_search(mask, max_iter=5000, seed=0):
    """Local search with add/remove/swap moves to maximize 3-APs while maintaining 4-AP-free."""
    rng = random.Random(seed)
    current_mask = mask.copy()
    current_score = _count_3aps(current_mask)
    best_mask = current_mask.copy()
    best_score = current_score
    
    for it in range(max_iter):
        improved = False
        # Try adding a random element
        candidates = [x for x in range(1, N+1) if not current_mask[x]]
        if candidates:
            for _ in range(5):  # try a few random additions
                x = rng.choice(candidates)
                if not _add_4ap_conflicts(current_mask, x):
                    current_mask[x] = True
                    new_score = _count_3aps(current_mask)
                    if new_score > current_score:
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask.copy()
                        improved = True
                        break
                    else:
                        current_mask[x] = False
        
        # Try removing an element that doesn't reduce score much, then re-adding
        if not improved:
            elems = [x for x in range(1, N+1) if current_mask[x]]
            if elems:
                x = rng.choice(elems)
                current_mask[x] = False
                # Try to add a different element
                candidates2 = [y for y in range(1, N+1) if not current_mask[y]]
                if candidates2:
                    for _ in range(3):
                        y = rng.choice(candidates2)
                        if not _add_4ap_conflicts(current_mask, y):
                            current_mask[y] = True
                            new_score = _count_3aps(current_mask)
                            if new_score >= current_score:
                                current_score = new_score
                                if current_score > best_score:
                                    best_score = current_score
                                    best_mask = current_mask.copy()
                                improved = True
                                break
                            else:
                                current_mask[y] = False
                    if not improved:
                        current_mask[x] = True  # restore
                        current_score = _count_3aps(current_mask)
        
        if time.time() > DEADLINE - 2:
            break
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    
    # Strategy 1: Forward greedy (deterministic)
    mask1 = _greedy_forward()
    score1 = _count_3aps(mask1)
    best_mask = mask1.copy()
    best_score = score1
    
    if time.time() > DEADLINE - 5:
        return sorted(np.where(best_mask)[0].tolist())
    
    # Strategy 2: Multiple random-order greedy + local search
    rng = random.Random(42)
    for trial in range(50):
        if time.time() > DEADLINE - 3:
            break
        seed = rng.randint(0, 10**9)
        mask = _greedy_random_order(seed)
        score = _count_3aps(mask)
        
        # Local search from this starting point
        mask_ls, score_ls = _local_search(mask, max_iter=2000, seed=seed)
        if score_ls > best_score:
            best_score = score_ls
            best_mask = mask_ls.copy()
    
    # Strategy 3: Start from best found, do more local search
    if time.time() > DEADLINE - 10:
        pass
    else:
        mask_final, score_final = _local_search(best_mask, max_iter=10000, seed=12345)
        if score_final > best_score:
            best_score = score_final
            best_mask = mask_final.copy()
    
    # Verify no 4-AP
    if _has_4ap(best_mask):
        # Fallback to deterministic greedy
        best_mask = _greedy_forward()
    
    result = sorted(np.where(best_mask)[0].tolist())
    if not result:
        result = [1]
    return result
