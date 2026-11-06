"""Optimized odd covering system search using bitset-based local search."""
import time
import random
import numpy as np

_BASE = 51975
_DEADLINE = 118.0  # safety margin before 150s kill

def _get_divisors(n):
    """Get all divisors > 1 of n."""
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _get_divisors(_BASE)
_DIVS_SORTED = sorted(_DIVS)

# Precompute index arrays for each divisor/residue pair
# _IDX[m_index][r] = array of indices in range(r, _BASE, m)
_m_list = _DIVS_SORTED
_m_to_idx = {m: i for i, m in enumerate(_m_list)}
_IDX = {}
for m in _m_list:
    _IDX[m] = [np.arange(r, _BASE, m, dtype=np.int32) for r in range(m)]

def _score(classes):
    """Exact uncovered count using numpy."""
    covered = np.zeros(_BASE, dtype=np.bool_)
    for m, r in classes:
        covered[_IDX[m][r]] = True
    return int(_BASE - covered.sum())

def _greedy_build(order):
    """Greedy: process moduli in given order, pick best residue."""
    covered = np.zeros(_BASE, dtype=np.bool_)
    classes = []
    for m in order:
        best_r = 0
        best_gain = -1
        for r in range(m):
            idx = _IDX[m][r]
            gain = int((~covered[idx]).sum())
            if gain > best_gain:
                best_gain = gain
                best_r = r
        classes.append([m, best_r])
        covered[_IDX[m][best_r]] = True
    return classes

def _local_search(classes, max_rounds=50):
    """Local search: for each modulus, find best residue given others fixed.
    
    Uses the observation that changing residue r to r' for modulus m
    only affects positions where (r - r') mod m == 0 are lost and 
    (r' - r) mod m == 0 are gained, but only if not covered by others.
    """
    classes = [c[:] for c in classes]
    n = len(classes)
    
    for round_num in range(max_rounds):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            m = classes[i][0]
            r_cur = classes[i][1]
            
            # Compute coverage from all other classes
            other_covered = np.zeros(_BASE, dtype=np.bool_)
            for j in range(n):
                if j == i:
                    continue
                mj, rj = classes[j]
                other_covered[_IDX[mj][rj]] = True
            
            # For each candidate residue r, compute the gain
            # gain(r) = number of positions in class(m,r) not covered by others
            #           - number of positions in class(m,r_cur) not covered by others
            cur_uncovered = int((~other_covered[_IDX[m][r_cur]]).sum())
            best_r = r_cur
            best_gain = 0
            
            for r in range(m):
                if r == r_cur:
                    continue
                unc = int((~other_covered[_IDX[m][r]]).sum())
                gain = unc - cur_uncovered
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_gain > 0:
                classes[i][1] = best_r
                improved = True
        
        if not improved:
            break
    
    return classes

def _two_opt_swap(classes, max_rounds=20):
    """Try swapping residues of pairs of moduli to find better combinations."""
    classes = [c[:] for c in classes]
    n = len(classes)
    
    for round_num in range(max_rounds):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                mi, ri = classes[i]
                mj, rj = classes[j]
                
                # Compute coverage excluding i and j
                other_covered = np.zeros(_BASE, dtype=np.bool_)
                for k in range(n):
                    if k == i or k == j:
                        continue
                    mk, rk = classes[k]
                    other_covered[_IDX[mk][rk]] = True
                
                # Current uncovered count with (mi,ri) and (mj,rj)
                cur_mask = other_covered.copy()
                cur_mask[_IDX[mi][ri]] = True
                cur_mask[_IDX[mj][rj]] = True
                cur_uncovered = int(_BASE - cur_mask.sum())
                
                # Try a few alternative residues for i
                best_combo = None
                best_uncovered = cur_uncovered
                for ri_new in range(mi):
                    if ri_new == ri:
                        continue
                    # Check if this improves things when combined with rj
                    test_mask = other_covered.copy()
                    test_mask[_IDX[mi][ri_new]] = True
                    test_mask[_IDX[mj][rj]] = True
                    unc = int(_BASE - test_mask.sum())
                    if unc < best_uncovered:
                        best_uncovered = unc
                        best_combo = (ri_new, rj)
                
                # Try a few alternative residues for j
                for rj_new in range(mj):
                    if rj_new == rj:
                        continue
                    test_mask = other_covered.copy()
                    test_mask[_IDX[mi][ri]] = True
                    test_mask[_IDX[mj][rj_new]] = True
                    unc = int(_BASE - test_mask.sum())
                    if unc < best_uncovered:
                        best_uncovered = unc
                        best_combo = (ri, rj_new)
                
                if best_combo is not None:
                    classes[i][1] = best_combo[0]
                    classes[j][1] = best_combo[1]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return classes

def construct():
    _start = time.time()
    best_classes = None
    best_score = _BASE + 1
    
    # Multiple restarts with different orderings
    rng = random.Random(42)
    orderings = [
        _DIVS_SORTED,
        sorted(_DIVS, reverse=True),
    ]
    for _ in range(10):
        orderings.append(rng.sample(_DIVS, len(_DIVS)))
    
    for order in orderings:
        if time.time() > _DEADLINE - 10:
            break
        
        # Greedy construction
        classes = _greedy_build(order)
        score = _score(classes)
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
        
        # Local search (single-variable)
        classes_ls = _local_search(classes)
        score_ls = _score(classes_ls)
        if score_ls < best_score:
            best_score = score_ls
            best_classes = [c[:] for c in classes_ls]
        
        # Two-opt swap (pairwise)
        if time.time() < _DEADLINE - 15:
            classes_swap = _two_opt_swap(classes_ls)
            score_swap = _score(classes_swap)
            if score_swap < best_score:
                best_score = score_swap
                best_classes = [c[:] for c in classes_swap]
    
    # Final polish: run local search on the best solution found
    if best_classes is not None and time.time() < _DEADLINE - 5:
        classes_final = _local_search(best_classes, max_rounds=100)
        score_final = _score(classes_final)
        if score_final < best_score:
            best_score = score_final
            best_classes = classes_final
    
    if best_classes is None:
        best_classes = _greedy_build(_DIVS_SORTED)
    
    return best_classes
