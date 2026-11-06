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

# Precompute isosceles masks:
# iso_c_mask[i][j] = bitmask of points k such that (i, j, k) is isosceles
# This allows fast checking: S is no-isosceles iff for all i,j in S, (iso_c_mask[i][j] & S_mask) == 0
# p is blocked by S iff there exist i,j in S such that p is in iso_c_mask[i][j]
# Equivalently: for each p, define block_mask[p] = OR over all pairs (i,j) of (1<<p) if (i,j,p) is isosceles
# But we need: p is blocked iff there exists pair (i,j) subset S such that (i,j,p) is isosceles.

# Let's precompute:
# 1. iso_c_mask[i][j] as a Python int bitmask (256 bits)
iso_c_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]
for i in range(N_TOTAL):
    for j in range(i + 1, N_TOTAL):
        di_j = _D2[i, j]
        mask = 0
        for k in range(N_TOTAL):
            if k == i or k == j:
                continue
            di_k = _D2[i, k]
            dj_k = _D2[j, k]
            if di_k == dj_k or di_k == di_j or dj_k == di_j:
                mask |= (1 << k)
        iso_c_mask[i][j] = mask
        iso_c_mask[j][i] = mask

# For each point p, precompute which pairs (i, j) make (i, j, p) isosceles
# blocked_pairs[p] = list of (i, j) such that (i, j, p) is isosceles, i < j
# Then p is blocked by S iff there exists (i, j) in blocked_pairs[p] with i in S and j in S
blocked_pairs = [[] for _ in range(N_TOTAL)]
for p in range(N_TOTAL):
    bit_p = 1 << p
    for i in range(N_TOTAL):
        for j in range(i + 1, N_TOTAL):
            if i == p or j == p:
                continue
            if iso_c_mask[i][j] & bit_p:
                blocked_pairs[p].append((i, j))

def _is_no_isosceles(S_list):
    """Check if set S is no-isosceles using precomputed masks."""
    n = len(S_list)
    if n < 3:
        return True
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            if iso_c_mask[a][b] & S_mask:
                return False
    return True

def _can_add(p, S_list):
    """Check if point p can be added to S without creating isosceles triangle."""
    n = len(S_list)
    if n < 2:
        return True
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    # p can be added iff for all a, b in S, (p, a, b) is NOT isosceles
    # i.e., p is not in iso_c_mask[a][b] for any a, b in S
    # Equivalently: for all a in S, (iso_c_mask[p][a] & S_mask) == 0
    for a in S_list:
        if iso_c_mask[p][a] & S_mask:
            return False
    return True

def _is_blocked(p, S_list):
    """Check if point p is blocked by set S (forms isosceles with some pair in S)."""
    n = len(S_list)
    if n < 2:
        return False
    S_mask = 0
    for idx in S_list:
        S_mask |= (1 << idx)
    for a, b in blocked_pairs[p]:
        if (S_mask & (1 << a)) and (S_mask & (1 << b)):
            return True
    return False

def _is_maximal(S_list):
    """Check if S is maximal: no point can be added."""
    S_set = set(S_list)
    for p in range(N_TOTAL):
        if p in S_set:
            continue
        if not _is_blocked(p, S_list):
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
            if _is_maximal(S_test):
                S = S_test
                changed = True
                break
    return S

def _greedy_cover(start_time, deadline_margin=15):
    """
    Greedy set-cover approach to build a maximal no-isosceles set.
    
    We want to find a small set S that is:
    1. No-isosceles
    2. Maximal (blocks all other points)
    
    Strategy: Start with a small core, then greedily add points that block the most
    currently unblocked points while maintaining the no-isosceles property.
    """
    chosen = []
    chosen_set = set()
    
    # Track which points are currently blocked
    # A point q is blocked if there exist a, b in chosen such that (a, b, q) is isosceles
    # i.e., q is in iso_c_mask[a][b]
    blocked = [False] * N_TOTAL
    
    while True:
        if time.time() - start_time > DEADLINE - deadline_margin:
            return None
        
        # Check if already maximal
        if _is_maximal(chosen):
            return chosen
        
        # Find all unblocked, non-chosen points
        unblocked = []
        for q in range(N_TOTAL):
            if q in chosen_set:
                continue
            if not blocked[q]:
                unblocked.append(q)
        
        if not unblocked:
            break
        
        # Find addable points
        addable = []
        for p in range(N_TOTAL):
            if p in chosen_set:
                continue
            if _can_add(p, chosen):
                addable.append(p)
        
        if not addable:
            break
        
        # For each addable point, compute how many unblocked points it would block
        best_p = None
        best_gain = -1
        
        for p in addable:
            gain = 0
            # If we add p, it blocks q if there exists a in chosen such that (p, a, q) is isosceles
            # i.e., q is in iso_c_mask[p][a] for some a in chosen
            for q in unblocked:
                for a in chosen:
                    if iso_c_mask[p][a] & (1 << q):
                        gain += 1
                        break
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        chosen.append(best_p)
        chosen_set.add(best_p)
        
        # Update blocked array
        for q in range(N_TOTAL):
            if q in chosen_set:
                continue
            if not blocked[q]:
                for a in chosen:
                    if iso_c_mask[best_p][a] & (1 << q):
                        blocked[q] = True
                        break
    
    return chosen

def _local_search(S, start_time, deadline_margin=10):
    """Try to improve set S by swapping one point for another."""
    S_list = list(S)
    n_S = len(S_list)
    best = S_list[:]
    best_size = n_S
    
    for i in range(n_S):
        if time.time() - start_time > DEADLINE - deadline_margin:
            break
        
        S_without = S_list[:i] + S_list[i + 1:]
        
        # Try adding each possible point
        for p_add in range(N_TOTAL):
            if p_add in set(S_without):
                continue
            
            if not _can_add(p_add, S_without):
                continue
            
            S_new = S_without + [p_add]
            
            if not _is_maximal(S_new):
                continue
            
            reduced = _reduce_to_minimal(S_new)
            
            if len(reduced) < best_size:
                best_size = len(reduced)
                best = reduced[:]
    
    return best

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy cover from empty
    result = _greedy_cover(start_time, deadline_margin=20)
    if result is not None:
        reduced = _reduce_to_minimal(result)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
    
    # Strategy 2: Random cores + greedy build
    for core_size in [2, 3, 4, 5]:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        n_attempts = 100
        for attempt in range(n_attempts):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            core_indices = rng.sample(range(N_TOTAL), core_size)
            
            if not _is_no_isosceles(core_indices):
                continue
            
            # Greedy build from core
            chosen = list(core_indices)
            chosen_set = set(chosen)
            
            blocked = [False] * N_TOTAL
            for q in range(N_TOTAL):
                if q in chosen_set:
                    continue
                for a in chosen:
                    for b in chosen:
                        if a < b and (iso_c_mask[a][b] & (1 << q)):
                            blocked[q] = True
                            break
                    if blocked[q]:
                        break
            
            while True:
                if time.time() - start_time > DEADLINE - 10:
                    break
                
                # Check if already maximal
                if _is_maximal(chosen):
                    break
                
                # Find unblocked points
                unblocked = []
                for q in range(N_TOTAL):
                    if q not in chosen_set and not blocked[q]:
                        unblocked.append(q)
                
                if not unblocked:
                    break
                
                # Find addable points
                addable = []
                for p in range(N_TOTAL):
                    if p not in chosen_set and _can_add(p, chosen):
                        addable.append(p)
                
                if not addable:
                    break
                
                # Pick addable point that blocks most unblocked
                best_p = None
                best_gain = -1
                for p in addable:
                    gain = 0
                    for q in unblocked:
                        for a in chosen:
                            if iso_c_mask[p][a] & (1 << q):
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
                for q in range(N_TOTAL):
                    if q in chosen_set:
                        continue
                    if not blocked[q]:
                        for a in chosen:
                            if iso_c_mask[best_p][a] & (1 << q):
                                blocked[q] = True
                                break
            
            if _is_maximal(chosen):
                reduced = _reduce_to_minimal(chosen)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    best_set = reduced[:]
            
            if best_size <= 8:
                break
    
    # Strategy 3: Local search improvement
    if best_set is not None and best_size > 8:
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            improved = _local_search(best_set, start_time, deadline_margin=10)
            if improved is not None and len(improved) < best_size:
                best_size = len(improved)
                best_set = improved[:]
            else:
                break
    
    # Fallback
    if best_set is None:
        chosen = []
        for i in range(N_TOTAL):
            if _can_add(i, chosen):
                chosen.append(i)
        best_set = _reduce_to_minimal(chosen)
    
    return [_ALL_POINTS[i] for i in best_set]
