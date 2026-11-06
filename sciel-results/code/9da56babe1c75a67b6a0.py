"""Optimized covering system search using NumPy vectorization."""
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

# Precompute indices for each modulus to speed up coverage updates
# For modulus m, the indices covered by residue r are np.arange(r, _BASE, m)
# We can't precompute all r for all m easily as it's large, but we can use slicing.

def _get_coverage(classes):
    """
    Compute the coverage mask (0/1 array of length _BASE) for the given classes.
    classes: list of [m, r]
    """
    mask = np.zeros(_BASE, dtype=np.uint8)
    for m, r in classes:
        # Use slicing for speed
        mask[r::_BASE//m if _BASE % m == 0 else m] = 1 
        # Wait, slicing `mask[r::m]` is the correct way to get r, r+m, r+2m...
        # But we must ensure we don't go out of bounds. Python slicing handles this automatically.
        mask[r::_BASE//m if False else m] = 1 # Correct syntax: mask[r::m] = 1
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
        best_r = 0
        best_count = -1
        
        # For each residue r, count how many new positions are covered
        # Vectorized approach:
        # For a fixed m, the positions covered by r are r, r+m, ...
        # We want to maximize sum(1 - mask[r::m])
        
        # We can compute this for all r at once?
        # Let's try a semi-vectorized approach for the inner loop.
        # For each r in range(m):
        #   idx = np.arange(r, _BASE, m)
        #   count = np.sum(1 - mask[idx])
        
        # This is still a loop over m residues.
        # Can we vectorize the loop over r?
        # The indices for different r are disjoint sets within the same modulus m.
        # Actually, for a fixed m, the sets {r, r+m, ...} for r=0..m-1 partition the set of indices 
        # that are congruent to 0..m-1 mod m? No, they partition the indices based on their residue mod m.
        
        # Let's stick to the loop over r but use numpy for the inner sum.
        for r in range(m):
            idx = np.arange(r, _BASE, m)
            count = np.sum(1 - mask[idx])
            if count > best_count:
                best_count = count
                best_r = r
        
        classes.append([m, best_r])
        mask[best_r::m] = 1
        
    return classes

def _local_search(classes, max_rounds=200):
    """Local search: for each modulus, find best residue given others fixed."""
    classes = [c[:] for c in classes]
    n = len(classes)
    
    for round_num in range(max_rounds):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        
        for i in range(n):
            m = classes[i][0]
            r_old = classes[i][1]
            
            # Compute coverage from all other classes
            other_classes = [classes[j] for j in range(n) if j != i]
            other_mask = np.zeros(_BASE, dtype=np.uint8)
            for mj, rj in other_classes:
                other_mask[rj::mj] = 1
            
            # For each residue r, compute how many newly covered
            best_r = r_old
            best_gain = -1
            
            for r in range(m):
                idx = np.arange(r, _BASE, m)
                gain = np.sum(1 - other_mask[idx])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != r_old:
                classes[i][1] = best_r
                improved = True
        
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
