import time
import random
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

# Precompute all points and pairwise squared distances
_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]
_POINT_INDEX = {p: i for i, p in enumerate(_ALL_POINTS)}
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute: for each point p, which other points q form an isosceles triangle
# with p and some third point. We'll build the "isosceles partner" structure.
# ISO_PARTNERS[p] = set of q such that there exists r with (p,q,r) isosceles
# More useful: for a set S, a point q is "blocked" if there exist a,b in S with (q,a,b) isosceles.
# PRECOMP[p][a] = bitmask or set of q such that (q,p,a) is isosceles
# Let's use: ISO_WITH[p][a] = frozenset of q where (q,p,a) is isosceles
# That's N_TOTAL^3 which is too large. Instead:
# For each pair (p,a), compute the set of q such that (q,p,a) is isosceles.
# Store as numpy boolean array: ISO_PAIR[p,a] = boolean array of length N_TOTAL

# Actually, let's precompute: for each p and each a, which q make (q,p,a) isosceles.
# iso_pair_mask[p, a, q] = True if (q, p, a) is isosceles
# This is N_TOTAL^3 booleans = 256^3 = 16.7M booleans ~ 16.7 MB, feasible.

_iso_pair_mask = np.zeros((N_TOTAL, N_TOTAL, N_TOTAL), dtype=bool)
for p in range(N_TOTAL):
    for a in range(N_TOTAL):
        if p == a:
            continue
        # For each q, check if (q, p, a) is isosceles
        d_qp = _D2[q_idx, p] if False else None  # placeholder
        # Vectorized: for all q
        d_qp = _D2[:, p]  # distance from q to p, shape (N_TOTAL,)
        d_qa = _D2[:, a]  # distance from q to a
        d_pa = _D2[p, a]  # scalar
        # Isosceles if d_qp == d_qa OR d_qp == d_pa OR d_qa == d_pa
        mask = (d_qp == d_qa) | (d_qp == d_pa) | (d_qa == d_pa)
        _iso_pair_mask[p, a, :] = mask

def _is_isosceles_triple(a, b, c):
    """Check if points a, b, c form an isosceles triangle."""
    d1 = _D2[a, b]
    d2 = _D2[a, c]
    d3 = _D2[b, c]
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add_fast(idx_p, chosen_indices):
    """Check if point idx_p can be added to chosen without creating isosceles triangle.
    Uses precomputed masks for speed."""
    if len(chosen_indices) < 2:
        return True
    ch = chosen_indices
    for i in range(len(ch)):
        for j in range(i + 1, len(ch)):
            a, b = ch[i], ch[j]
            if _is_isosceles_triple(idx_p, a, b):
                return False
    return True

def _is_no_isosceles(indices):
    """Check if set of point indices forms no-isosceles set."""
    n = len(indices)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_isosceles_triple(indices[i], indices[j], indices[k]):
                    return False
    return True

def _compute_blocked_mask(chosen_indices):
    """Compute which points are blocked by the chosen set.
    A point q is blocked if there exist a, b in chosen such that (q, a, b) is isosceles.
    Returns a boolean numpy array of length N_TOTAL."""
    blocked = np.zeros(N_TOTAL, dtype=bool)
    ch = chosen_indices
    n = len(ch)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ch[i], ch[j]
            blocked |= _iso_pair_mask[a, b, :]
    return blocked

def _is_maximal_fast(chosen_indices):
    """Check if chosen set is maximal using precomputed masks."""
    chosen_set = set(chosen_indices)
    blocked = _compute_blocked_mask(chosen_indices)
    for i in range(N_TOTAL):
        if i in chosen_set:
            continue
        if not blocked[i]:
            return False
    return True

def _reduce_to_minimal(chosen_indices):
    """Iteratively remove points if remaining set is still maximal."""
    S = list(chosen_indices)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            if len(S) - 1 < 2:
                continue
            S_test = S[:i] + S[i + 1:]
            if _is_maximal_fast(S_test):
                S = S_test
                changed = True
                break
    return S

def _greedy_hitting_set(deadline_margin=15):
    """
    Greedy hitting set: build a maximal no-isosceles set by greedily adding
    points that cover the most uncovered points (for maximality).
    """
    start_time = time.time()
    
    chosen = []
    chosen_set = set()
    blocked = np.zeros(N_TOTAL, dtype=bool)
    
    while True:
        if time.time() - start_time > DEADLINE - deadline_margin:
            return None
        
        # Find all addable points (vectorized-ish)
        addable = []
        for i in range(N_TOTAL):
            if i in chosen_set:
                continue
            if _can_add_fast(i, chosen):
                addable.append(i)
        
        if not addable:
            break
        
        # Check if already maximal
        if _is_maximal_fast(chosen):
            return chosen
        
        # For each addable point, compute coverage gain
        # Adding point p blocks all q such that exists a in chosen with (q,p,a) isosceles
        best_p = None
        best_gain = -1
        
        for p in addable:
            # Compute new blocked points if we add p
            new_blocked_count = 0
            for a in chosen:
                mask = _iso_pair_mask[p, a, :]
                new_blocked_count += np.sum(mask & ~blocked & ~np.isin(np.arange(N_TOTAL), list(chosen_set)))
            # Simpler: just count how many currently unblocked, unchosen points would be blocked
            for a in chosen:
                pass
            # Recompute properly
            gain = 0
            for q in range(N_TOTAL):
                if q in chosen_set or blocked[q]:
                    continue
                for a in chosen:
                    if _iso_pair_mask[p, a, q]:
                        gain += 1
                        break
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        chosen.append(best_p)
        chosen_set.add(best_p)
        
        # Update blocked
        for a in chosen[:-1]:
            blocked |= _iso_pair_mask[best_p, a, :]
    
    return chosen

def _local_search_improve(S, deadline_margin=10):
    """Try to improve set S by removing one point and adding another."""
    start_time = time.time()
    S_list = list(S)
    n_S = len(S_list)
    
    for i in range(n_S):
        if time.time() - start_time > DEADLINE - deadline_margin:
            break
        
        S_without = S_list[:i] + S_list[i + 1:]
        S_without_set = set(S_without)
        
        best_new = None
        best_size = n_S
        
        for p_add in range(N_TOTAL):
            if p_add in S_without_set:
                continue
            
            if not _can_add_fast(p_add, S_without):
                continue
            
            S_new = S_without + [p_add]
            
            if not _is_maximal_fast(S_new):
                continue
            
            reduced = _reduce_to_minimal(S_new)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_new = reduced
        
        if best_new is not None:
            return best_new
    
    return S_list

def _greedy_from_core(core_indices, deadline_margin=15):
    """Build a maximal no-isosceles set starting from a given core."""
    start_time = time.time()
    
    if not _is_no_isosceles(core_indices):
        return None
    
    chosen = list(core_indices)
    chosen_set = set(chosen)
    blocked = _compute_blocked_mask(chosen)
    
    while True:
        if time.time() - start_time > DEADLINE - deadline_margin:
            return None
        
        # Find addable points
        addable = []
        for i in range(N_TOTAL):
            if i in chosen_set:
                continue
            if _can_add_fast(i, chosen):
                addable.append(i)
        
        if not addable:
            break
        
        if _is_maximal_fast(chosen):
            break
        
        # Pick point with max coverage gain
        best_p = None
        best_gain = -1
        for p in addable:
            gain = 0
            for q in range(N_TOTAL):
                if q in chosen_set or blocked[q]:
                    continue
                for a in chosen:
                    if _iso_pair_mask[p, a, q]:
                        gain += 1
                        break
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        chosen.append(best_p)
        chosen_set.add(best_p)
        for a in chosen[:-1]:
            blocked |= _iso_pair_mask[best_p, a, :]
    
    return chosen

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy from empty
    result = _greedy_hitting_set(deadline_margin=20)
    if result is not None:
        reduced = _reduce_to_minimal(result)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
    
    # Strategy 2: Random cores + greedy build
    for core_size in [2, 3, 4, 5]:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        n_attempts = 200
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            core_indices = rng.sample(range(N_TOTAL), core_size)
            
            result = _greedy_from_core(core_indices, deadline_margin=10)
            if result is not None and _is_maximal_fast(result):
                reduced = _reduce_to_minimal(result)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 8:
                break
    
    # Strategy 3: Local search improvement
    if best_set is not None and best_size > 8:
        for _ in range(30):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            improved = _local_search_improve(best_set, deadline_margin=10)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            else:
                break
    
    # Fallback
    if best_set is None:
        chosen = []
        for i in range(N_TOTAL):
            if _can_add_fast(i, chosen):
                chosen.append(i)
        best_set = _reduce_to_minimal(chosen)
    
    return [_ALL_POINTS[i] for i in best_set]
