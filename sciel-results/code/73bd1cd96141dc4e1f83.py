"""Odd covering systems with moduli dividing 51975.
Combines efficient greedy set cover with multi-stage local search and
residue-space randomization for robustness.
"""
import time
import numpy as np

BASE = 51975  # 3^3 * 5^2 * 7 * 11

# Precompute all divisors > 1 of BASE
_DIVISORS = None

def _get_divisors():
    global _DIVISORS
    if _DIVISORS is None:
        all_divs = set()
        for a in range(4):  # 3^0..3^3
            for b in range(3):  # 5^0..5^2
                for c in range(2):  # 7^0..7^1
                    for d in range(2):  # 11^0..11^1
                        val = (3**a) * (5**b) * (7**c) * (11**d)
                        if val > 1:
                            all_divs.add(val)
        _DIVISORS = sorted(all_divs)
    return _DIVISORS

def _best_residue_for_mod(m, uncovered_arr):
    """Find the residue r that covers the most uncovered residues for modulus m."""
    best_r = 0
    best_count = -1
    for r in range(m):
        cnt = int(np.sum(uncovered_arr[r::m]))
        if cnt > best_count:
            best_count = cnt
            best_r = r
    return best_r, best_count

def _greedy_construct(div_order, deadline):
    """Greedy construction with a given divisor ordering."""
    uncovered_arr = np.ones(BASE, dtype=np.uint8)
    selected_classes = []
    used_moduli = set()
    
    for m in div_order:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        r, gain = _best_residue_for_mod(m, uncovered_arr)
        if gain <= 0:
            continue
        
        used_moduli.add(m)
        selected_classes.append([m, r])
        uncovered_arr[r::m] = 0
    
    current_score = int(np.sum(uncovered_arr))
    return selected_classes, uncovered_arr, current_score

def _local_search(classes, uncovered_arr, current_score, deadline, max_iters=30):
    """Efficient local search: try changing each class's residue."""
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        for i in range(len(classes)):
            if time.time() > deadline:
                return classes, uncovered_arr, current_score
            
            m, r_old = classes[i]
            
            # Compute uncovered without class i
            unc_without = uncovered_arr.copy()
            unc_without[r_old::m] = 1  # restore what class i was covering
            
            # Find best residue for m given unc_without
            best_r = r_old
            best_gain = -1
            for r in range(m):
                gain = int(np.sum(unc_without[r::m]))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != r_old:
                # Apply the change
                classes[i] = [m, best_r]
                uncovered_arr = unc_without.copy()
                uncovered_arr[best_r::m] = 0
                new_score = int(np.sum(uncovered_arr))
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Revert
                    classes[i] = [m, r_old]
                    uncovered_arr[r_old::m] = 0
                    # uncovered_arr is now correct again
    
    return classes, uncovered_arr, current_score

def _add_unused_moduli(classes, uncovered_arr, current_score, divisors_sorted, deadline):
    """Try adding unused moduli that improve the score."""
    used_moduli = set(m for m, r in classes)
    improved = True
    
    while improved:
        improved = False
        for m in divisors_sorted:
            if time.time() > deadline:
                return classes, uncovered_arr, current_score
            if m in used_moduli:
                continue
            
            best_r, best_gain = _best_residue_for_mod(m, uncovered_arr)
            if best_gain > 0:
                new_unc = uncovered_arr.copy()
                new_unc[best_r::m] = 0
                new_score = int(np.sum(new_unc))
                if new_score < current_score:
                    used_moduli.add(m)
                    classes.append([m, best_r])
                    uncovered_arr = new_unc
                    current_score = new_score
                    improved = True
    
    return classes, uncovered_arr, current_score

def _pair_swap_search(classes, uncovered_arr, current_score, deadline, max_iters=10):
    """Try swapping residues between pairs of classes (limited candidate set)."""
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        for i in range(len(classes)):
            if time.time() > deadline:
                return classes, uncovered_arr, current_score
            for j in range(i + 1, len(classes)):
                if time.time() > deadline:
                    return classes, uncovered_arr, current_score
                
                mi, ri = classes[i]
                mj, rj = classes[j]
                
                # Compute uncovered without classes i and j
                unc_ij = uncovered_arr.copy()
                unc_ij[ri::mi] = 1
                unc_ij[rj::mj] = 1
                
                # Try a limited set of candidate residues
                best_ri = ri
                best_rj = rj
                best_score = current_score
                
                # For efficiency, only try a subset of residues
                ri_candidates = [ri]
                for r in range(min(mi, 30)):
                    if r != ri:
                        ri_candidates.append(r)
                rj_candidates = [rj]
                for r in range(min(mj, 30)):
                    if r != rj:
                        rj_candidates.append(r)
                
                for new_ri in ri_candidates:
                    if time.time() > deadline:
                        break
                    for new_rj in rj_candidates:
                        if new_ri == ri and new_rj == rj:
                            continue
                        
                        test_unc = unc_ij.copy()
                        test_unc[new_ri::mi] = 0
                        test_unc[new_rj::mj] = 0
                        test_score = int(np.sum(test_unc))
                        
                        if test_score < best_score:
                            best_score = test_score
                            best_ri = new_ri
                            best_rj = new_rj
                
                if best_ri != ri or best_rj != rj:
                    classes[i] = [mi, best_ri]
                    classes[j] = [mj, best_rj]
                    uncovered_arr = unc_ij.copy()
                    uncovered_arr[best_ri::mi] = 0
                    uncovered_arr[best_rj::mj] = 0
                    current_score = best_score
                    improved = True
    
    return classes, uncovered_arr, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    divisors = _get_divisors()
    divisors_sorted = sorted(divisors)
    
    # Multiple greedy starts with different orderings
    best_classes = None
    best_score = BASE
    best_uncovered = None
    
    # Strategy 1: Greedy by increasing modulus
    classes1, unc1, score1 = _greedy_construct(divisors_sorted, deadline)
    if score1 < best_score:
        best_score = score1
        best_classes = classes1[:]
        best_uncovered = unc1.copy()
    
    # Strategy 2: Greedy by decreasing modulus
    if time.time() < deadline:
        divs_desc = sorted(divisors, reverse=True)
        classes2, unc2, score2 = _greedy_construct(divs_desc, deadline)
        if score2 < best_score:
            best_score = score2
            best_classes = classes2[:]
            best_uncovered = unc2.copy()
    
    # Strategy 3: Randomized greedy (if time permits)
    if time.time() < deadline - 30:
        random_order = divisors_sorted[:]
        np.random.shuffle(random_order)
        classes3, unc3, score3 = _greedy_construct(random_order, deadline)
        if score3 < best_score:
            best_score = score3
            best_classes = classes3[:]
            best_uncovered = unc3.copy()
    
    if best_classes is None:
        return []
    
    # Local search refinement
    best_classes, best_uncovered, best_score = _local_search(
        best_classes, best_uncovered, best_score, deadline
    )
    
    # Add unused moduli
    best_classes, best_uncovered, best_score = _add_unused_moduli(
        best_classes, best_uncovered, best_score, divisors_sorted, deadline
    )
    
    # Another round of local search
    best_classes, best_uncovered, best_score = _local_search(
        best_classes, best_uncovered, best_score, deadline, max_iters=20
    )
    
    # Pair swap search
    if time.time() < deadline - 10:
        best_classes, best_uncovered, best_score = _pair_swap_search(
            best_classes, best_uncovered, best_score, deadline
        )
    
    # Final local search polish
    if time.time() < deadline:
        best_classes, best_uncovered, best_score = _local_search(
            best_classes, best_uncovered, best_score, deadline, max_iters=10
        )
    
    return best_classes
