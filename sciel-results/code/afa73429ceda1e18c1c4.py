"""Construct a finite set U to maximize sumset/difference-set exponent lower bound."""
import time
import numpy as np

_DEADLINE = 110.0  # safety margin before 150s kill
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _sumset(A, B=None):
    if B is None:
        B = A
    A = sorted(A)
    B = sorted(B)
    result = set()
    for a in A:
        for b in B:
            result.add(a + b)
    return result

def _diffset(A, B=None):
    if B is None:
        B = A
    A = sorted(A)
    B = sorted(B)
    result = set()
    for a in A:
        for b in B:
            result.add(a - b)
    return result

def _score_set(U):
    if not U or 0 not in U:
        return 0.0
    U = list(set(U))
    if len(U) < 2:
        return 0.0
    UU = _sumset(U)
    UminusU = _diffset(U)
    ratio = len(UminusU) / len(UU)
    if ratio <= 1.0:
        return 1.0
    max_val = max(U)
    if max_val == 0:
        return 1.0
    return 1.0 + np.log(ratio) / np.log(2 * max_val + 1)

def _product_set(base, depth):
    """Compute product set: elements are sum of b_i * M_i where M_i = product of max of previous bases."""
    if depth <= 0:
        return [0]
    if depth == 1:
        return list(base)
    # For product construction, use mixed-radix:
    # U_k = { sum_{i=1}^{k} s_i * P_i : s_i in S_i }
    # where P_i = prod_{j=1}^{i-1} (max(S_j) + 1)
    # This ensures unique representation if bases are "efficient"
    # Simpler: just use the base with appropriate scaling
    # For identical base S with max m:
    # P_1 = 1, P_2 = m+1, P_3 = (m+1)^2, ...
    m = max(base)
    M = m + 1
    U = [0]
    for i in range(depth):
        scale = M ** i
        new_U = []
        for u in U:
            for s in base:
                new_U.append(u + s * scale)
        U = new_U
    return U

def _find_best_base(max_size=8, max_val=15):
    """Search for small base sets with high ratio."""
    from itertools import combinations
    best = None
    best_score = 0.0
    # Try all subsets of {0,1,...,max_val} of size up to max_size that contain 0
    for size in range(2, max_size + 1):
        for combo in combinations(range(1, max_val + 1), size - 1):
            S = sorted([0] + list(combo))
            UU = _sumset(S)
            DU = _diffset(S)
            ratio = len(DU) / len(UU)
            max_v = max(S)
            if max_v == 0:
                continue
            score = 1.0 + np.log(ratio) / np.log(2 * max_v + 1) if ratio > 1 else 1.0
            if score > best_score:
                best_score = score
                best = S
    return best, best_score

def _local_search(U, max_iters=500):
    """Try to improve U by adding/removing elements."""
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    best_score = _score_set(U)
    best_U = list(U)
    max_val = max(U)
    candidates = set()
    # Candidates to try adding
    for i in range(1, min(max_val + 20, 500)):
        if i not in U:
            candidates.add(i)
    # Also try removing (not 0)
    for u in U:
        if u != 0:
            candidates.add(-u)  # negative means remove
    
    for _ in range(max_iters):
        if _time_left() < 5:
            break
        improved = False
        for c in list(candidates):
            if _time_left() < 5:
                break
            if c > 0:
                new_U = sorted(U + [c])
                s = _score_set(new_U)
                if s > best_score + 1e-10:
                    best_score = s
                    best_U = new_U
                    U = new_U
                    candidates.discard(c)
                    # Add new candidates
                    new_max = max(U)
                    for i in range(1, min(new_max + 15, 600)):
                        if i not in U:
                            candidates.add(i)
                    improved = True
                    break
            else:
                # Remove element -c
                elem = -c
                if elem in U and elem != 0:
                    new_U = [x for x in U if x != elem]
                    s = _score_set(new_U)
                    if s > best_score + 1e-10:
                        best_score = s
                        best_U = new_U
                        U = new_U
                        candidates.discard(c)
                        improved = True
                        break
        if not improved:
            break
    return best_U, best_score

def construct():
    global _START
    _START = time.time()
    
    # Start with a known good construction
    # Product of {0,1,2,5} with depth 3
    base = [0, 1, 2, 5]
    depth = 3
    U = _product_set(base, depth)
    best_U = list(U)
    best_score = _score_set(U)
    
    # Try different bases and depths
    bases_to_try = [
        ([0, 1, 2, 5], 3),
        ([0, 1, 2, 5], 4),
        ([0, 1, 3], 3),
        ([0, 1, 3], 4),
        ([0, 1, 2, 4, 6], 2),
        ([0, 1, 2, 4, 6], 3),
    ]
    
    for b, d in bases_to_try:
        if _time_left() < 10:
            break
        try:
            U_test = _product_set(b, d)
            s = _score_set(U_test)
            if s > best_score:
                best_score = s
                best_U = U_test
        except:
            continue
    
    # Local search to improve
    if _time_left() > 15:
        best_U, best_score = _local_search(best_U, max_iters=200)
    
    # Ensure 0 is in the set and it's a list of nonnegative integers
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.append(0)
        best_U = sorted(best_U)
    
    return list(best_U)

# EVOLVE-BLOCK-START
# The construct() function above is the entry point
# EVOLVE-BLOCK-END
