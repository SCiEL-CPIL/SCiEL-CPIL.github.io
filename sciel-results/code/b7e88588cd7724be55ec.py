import time
import numpy as np
import random

_BASE = 51975
_DEADLINE = 115.0
_START = time.time()

# Precompute divisors
def _get_divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors(_BASE)
_DIVS_SORTED = sorted(_DIVS)
_N = _BASE
_ARANGE_N = np.arange(_N)

def _get_best_residue(mask, m):
    """
    Find residue r in [0, m) that maximizes the number of uncovered positions
    in mask (where 0 is uncovered, 1 is covered) that would be covered by r.
    i.e., maximize sum( (1 - mask)[r::m] )
    """
    remaining = 1 - mask
    keys = _ARANGE_N % m
    counts = np.bincount(keys, weights=remaining, minlength=m)
    best_r = int(np.argmax(counts))
    return best_r

def _get_coverage(classes):
    """
    Compute the coverage mask (0/1 array of length _BASE) for the given classes.
    classes: list of [m, r]
    """
    mask = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        mask[r::m] = 1
    return mask

def _score(classes):
    """Exact uncovered count."""
    mask = _get_coverage(classes)
    return _BASE - np.sum(mask)

def _build_greedy(order=None):
    """Greedy construction: moduli in given order, each picks best residue."""
    if order is None:
        order = _DIVS_SORTED
    
    mask = np.zeros(_BASE, dtype=np.uint8)
    classes = []
    
    for m in order:
        if time.time() > _DEADLINE - 5:
            break
            
        best_r = _get_best_residue(mask, m)
        
        classes.append([m, best_r])
        mask[best_r::m] = 1
        
    return classes

def _local_search(classes, max_rounds=200):
    """Local search: for each modulus, find best residue given others fixed."""
    classes = [c[:] for c in classes]
    n = len(classes)
    if n == 0: return classes
    
    # Initialize count_mask
    count_mask = np.zeros(_BASE, dtype=np.uint16)
    for m, r in classes:
        count_mask[r::m] += 1
    
    for round_num in range(max_rounds):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        
        for i in range(n):
            m = classes[i][0]
            r_old = classes[i][1]
            
            # Temporarily remove class i from count_mask
            idx_old = np.arange(r_old, _BASE, m)
            count_mask[idx_old] -= 1
            
            # Compute mask_others: 1 if count > 0
            mask_others = (count_mask > 0).astype(np.uint8)
            
            # Find best residue for class i given mask_others
            best_r = _get_best_residue(mask_others, m)
            
            # Check if improvement
            # Gain of best_r vs r_old?
            # We can just check if best_r is different and if the coverage improves.
            # Actually, _get_best_residue maximizes coverage of 1-mask_others.
            # So if best_r != r_old, it's strictly better or equal.
            # If equal, we might keep old to avoid oscillation, or switch.
            # Let's check the score difference.
            
            # Coverage of r_old on mask_others:
            # sum(1 - mask_others[r_old::m])
            # Coverage of best_r on mask_others:
            # sum(1 - mask_others[best_r::m])
            
            # We can compute these quickly?
            # Or just trust argmax.
            # If best_r != r_old, we update.
            
            if best_r != r_old:
                # Restore old? No, we are replacing it.
                # We already decremented count_mask for old.
                # So we just increment for new.
                idx_new = np.arange(best_r, _BASE, m)
                count_mask[idx_new] += 1
                classes[i][1] = best_r
                improved = True
            else:
                # No change, restore count_mask
                count_mask[idx_old] += 1
        
        if not improved:
            break
    
    return classes

def construct():
    best_classes = None
    best_score = _BASE + 1
    
    # 1. Start with greedy in sorted order
    classes = _build_greedy(_DIVS_SORTED)
    best_score = _score(classes)
    best_classes = [c[:] for c in classes]
    
    # 2. Local search on greedy result
    classes_ls = _local_search(classes)
    score_ls = _score(classes_ls)
    if score_ls < best_score:
        best_score = score_ls
        best_classes = classes_ls
    
    # 3. Try greedy in reverse order
    if time.time() < _DEADLINE - 20:
        classes_rev = _build_greedy(sorted(_DIVS, reverse=True))
        score_rev = _score(classes_rev)
        if score_rev < best_score:
            best_score = score_rev
            best_classes = [c[:] for c in classes_rev]
        
        # Local search on reverse greedy
        classes_rev_ls = _local_search(classes_rev)
        score_rev_ls = _score(classes_rev_ls)
        if score_rev_ls < best_score:
            best_score = score_rev_ls
            best_classes = classes_rev_ls

    # 4. Random restarts with local search
    rng = random.Random(42)
    
    # We can afford many more restarts now due to speedup
    while time.time() < _DEADLINE - 10:
        random_order = rng.sample(_DIVS, len(_DIVS))
        classes_rand = _build_greedy(random_order)
        
        # Local search immediately
        classes_rand_ls = _local_search(classes_rand, max_rounds=50)
        score_rand = _score(classes_rand_ls)
        
        if score_rand < best_score:
            best_score = score_rand
            best_classes = [c[:] for c in classes_rand_ls]
            
    return best_classes
