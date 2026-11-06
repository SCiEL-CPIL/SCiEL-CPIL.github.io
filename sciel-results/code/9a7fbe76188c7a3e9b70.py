import time
import random
import numpy as np

N = 16
DEADLINE = 325.0
N_TOTAL = N * N

_ALL_POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
_D2 = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
for i, p1 in enumerate(_ALL_POINTS):
    for j, p2 in enumerate(_ALL_POINTS):
        _D2[i, j] = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute iso_c_mask[i][j] = bitmask of points k such that (i,j,k) is isosceles
# We use Python ints for bitmasks since N_TOTAL=256
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

def _popcount(x):
    return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

def _is_no_isosceles(S_mask, S_list):
    """Check if set S is no-isosceles using precomputed masks."""
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            if iso_c_mask[a][b] & S_mask:
                return False
    return True

def _is_blocked_mask(p, S_mask):
    """
    Returns a bitmask of points q that are blocked by p in conjunction with S?
    No, we need to know if p is blocked by S.
    p is blocked by S iff there exist a,b in S such that (p,a,b) is isosceles.
    This is equivalent to: for some a in S, there exists b in S such that b is in iso_c_mask[p][a].
    So, for a fixed p, let M_a = iso_c_mask[p][a].
    p is blocked if (S_mask & M_a) has any bit set for some a in S.
    Actually, we can precompute for each p a list of (a, mask_b) pairs.
    But for the greedy step, we need the reverse: given S, which points are blocked?
    """
    pass

def _get_blocked_mask(S_mask, S_list):
    """
    Returns a bitmask of all points q that are blocked by S.
    q is blocked if there exist a,b in S such that (q,a,b) is isosceles.
    This is equivalent to: for some pair a,b in S, q is in iso_c_mask[a][b].
    So blocked_mask = union over all pairs a,b in S of iso_c_mask[a][b].
    """
    blocked = 0
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            blocked |= iso_c_mask[a][b]
    return blocked

def _is_maximal(S_mask, S_list):
    """Check if S is maximal: no point can be added."""
    blocked_mask = _get_blocked_mask(S_mask, S_list)
    # All points should be either in S or blocked
    # Unblocked points are those not in S and not in blocked_mask
    # If there are any unblocked points, it's not maximal
    # Note: blocked_mask might include points in S, that's fine.
    # We just need to check if (all_points_mask ^ S_mask) & ~blocked_mask is 0
    all_mask = (1 << N_TOTAL) - 1
    unblocked = (all_mask ^ S_mask) & ~blocked_mask
    return unblocked == 0

def _greedy_build(deadline_margin=15):
    """
    Greedy construction using bitmask coverage.
    """
    S_list = []
    S_mask = 0
    all_mask = (1 << N_TOTAL) - 1
    
    while True:
        if time.time() > DEADLINE - deadline_margin:
            return None
        
        # Calculate currently blocked points
        blocked_mask = _get_blocked_mask(S_mask, S_list)
        
        # Check if maximal
        unblocked_mask = (all_mask ^ S_mask) & ~blocked_mask
        if unblocked_mask == 0:
            break
            
        # Find addable points
        # A point p is addable if it is not in S and adding it doesn't create an isosceles triangle with existing points.
        # Adding p creates isosceles if for some a,b in S, (p,a,b) is isosceles.
        # This is equivalent to: p is in iso_c_mask[a][b] for some a,b in S.
        # So p is addable iff p is NOT in (union over a,b in S of iso_c_mask[a][b])?
        # Wait. The condition for S to be no-isosceles is that no triple in S is isosceles.
        # If we add p, we must ensure (p,a,b) is not isosceles for any a,b in S.
        # This means p must not be in iso_c_mask[a][b] for any a,b in S.
        # So the set of "forbidden" points for addition is exactly the `blocked_mask` computed above?
        # No. `blocked_mask` contains points q that form isosceles with *some pair* in S.
        # If p is in `blocked_mask`, it means there exist a,b in S such that (p,a,b) is isosceles.
        # So yes, p is addable iff p is not in S and p is not in blocked_mask.
        
        addable_mask = (all_mask ^ S_mask) & ~blocked_mask
        
        if addable_mask == 0:
            break
            
        # Score each addable point by how many *unblocked* points it would block
        # If we add p, the new blocked mask will include:
        # 1. Old blocked points
        # 2. Points q such that (q,p,a) is isosceles for some a in S.
        #    This is union over a in S of iso_c_mask[p][a].
        # 3. Points q such that (q,p,b) is isosceles... wait, the pair is {p,a}.
        #    So the new points blocked by pairs involving p are:
        #    Union over a in S of iso_c_mask[p][a].
        #    Let's call this `new_blocked_by_p`.
        #    The points *newly* blocked are `new_blocked_by_p & unblocked_mask`.
        #    Note: p itself is removed from unblocked (it becomes part of S).
        #    So gain = popcount(new_blocked_by_p & unblocked_mask).
        #    We should also count p itself as "covered" (it leaves the unblocked set).
        #    So gain = 1 + popcount(new_blocked_by_p & unblocked_mask).
        
        best_p = -1
        best_gain = -1
        
        # Iterate through set bits in addable_mask
        temp_mask = addable_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            p = lsb.bit_length() - 1
            temp_mask ^= lsb
            
            # Calculate new blocked points due to p
            new_blocked_by_p = 0
            for a in S_list:
                new_blocked_by_p |= iso_c_mask[p][a]
                
            # Points newly blocked that were previously unblocked
            newly_blocked = new_blocked_by_p & unblocked_mask
            gain = 1 + _popcount(newly_blocked)
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
                
        if best_p == -1:
            break
            
        S_list.append(best_p)
        S_mask |= (1 << best_p)
        
    return S_list

def _reduce_to_minimal(S_list, deadline_margin=10):
    """Iteratively remove points if remaining set is still maximal and no-isosceles."""
    S = list(S_list)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            if len(S) - 1 < 2:
                continue
            S_test = S[:i] + S[i+1:]
            S_mask_test = 0
            for x in S_test:
                S_mask_test |= (1 << x)
            if _is_no_isosceles(S_mask_test, S_test) and _is_maximal(S_mask_test, S_test):
                S = S_test
                changed = True
                break
        if time.time() > DEADLINE - deadline_margin:
            break
    return S

def _build_from_core(core, deadline_margin=15):
    """Start with a core and extend greedily."""
    S_list = list(core)
    S_mask = 0
    for x in core:
        S_mask |= (1 << x)
        
    if not _is_no_isosceles(S_mask, S_list):
        return None
        
    all_mask = (1 << N_TOTAL) - 1
    
    while True:
        if time.time() > DEADLINE - deadline_margin:
            return None
            
        blocked_mask = _get_blocked_mask(S_mask, S_list)
        unblocked_mask = (all_mask ^ S_mask) & ~blocked_mask
        
        if unblocked_mask == 0:
            break
            
        addable_mask = (all_mask ^ S_mask) & ~blocked_mask
        
        if addable_mask == 0:
            break
            
        best_p = -1
        best_gain = -1
        
        temp_mask = addable_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            p = lsb.bit_length() - 1
            temp_mask ^= lsb
            
            new_blocked_by_p = 0
            for a in S_list:
                new_blocked_by_p |= iso_c_mask[p][a]
                
            newly_blocked = new_blocked_by_p & unblocked_mask
            gain = 1 + _popcount(newly_blocked)
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
                
        if best_p == -1:
            break
            
        S_list.append(best_p)
        S_mask |= (1 << best_p)
        
    return S_list

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Pure greedy
    for attempt in range(20):
        if time.time() - start_time > DEADLINE - 30:
            break
        result = _greedy_build(deadline_margin=15)
        if result is None:
            continue
        reduced = _reduce_to_minimal(result, deadline_margin=10)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        if best_size <= 8:
            break

    # Strategy 2: Greedy from random cores
    for attempt in range(100):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        core_size = rng.choice([2, 3, 4])
        core = rng.sample(range(N_TOTAL), core_size)
        core_mask = 0
        for x in core:
            core_mask |= (1 << x)
            
        if not _is_no_isosceles(core_mask, core):
            continue
            
        result = _build_from_core(core, deadline_margin=15)
        if result is None:
            continue
            
        reduced = _reduce_to_minimal(result, deadline_margin=10)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
            
        if best_size <= 8:
            break
            
    if best_set is None:
        # Fallback
        chosen = []
        chosen_mask = 0
        for i in range(N_TOTAL):
            if _is_no_isosceles(chosen_mask | (1<<i), chosen + [i]):
                chosen.append(i)
                chosen_mask |= (1 << i)
        best_set = _reduce_to_minimal(chosen, deadline_margin=5)
        
    return [_ALL_POINTS[i] for i in best_set]
