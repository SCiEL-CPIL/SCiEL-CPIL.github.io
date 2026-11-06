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

# Precompute: for each pair (i,j), bitmask of points k such that (i,j,k) is isosceles
# Using Python ints as bitmasks (256 bits needed)
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

# Precompute conflict matrix: conflict[i][j] = True if there exists k such that (i,j,k) is isosceles
# i.e., if adding both i and j to a set would require checking k
# Actually: i and j "conflict" if iso_c_mask[i][j] != 0 (there exists some k that forms isosceles with i,j)
# But for the no-isosceles constraint on set S: for all pairs (a,b) in S, no c in S should be in iso_c_mask[a][b]

# For vectorized checking: 
# can_add(p, S) = for all a in S: (iso_c_mask[p][a] & S_mask) == 0
# blocked(p, S) = exists a in S such that (iso_c_mask[p][a] & S_mask) != 0 ... wait no
# blocked(p, S) = exists a,b in S such that (p,a,b) is isosceles
# = exists a in S: (1<<p) is in iso_c_mask[a][b] for some b in S
# = exists a in S: (iso_c_mask[a][p] & S_mask) != 0  ... where iso_c_mask[a][p] contains b's such that (a,p,b) isosceles
# Wait, iso_c_mask[a][p] = set of k such that (a,p,k) is isosceles. So (p,a,b) isosceles iff b in iso_c_mask[a][p].
# So blocked(p, S) = exists a in S: (iso_c_mask[a][p] & S_mask) != 0

# Precompute iso_c_mask as a numpy array of uint64 pairs (256 bits = 4 x uint64)
# Actually let's use Python ints but create a lookup structure for vectorized access

# For vectorized can_add: 
#   For candidate p, check: for all a in S, (iso_c_mask[p][a] & S_mask) == 0
#   This is equivalent to: (OR over a in S of iso_c_mask[p][a]) & S_mask == 0
#   Let union_iso[p] = OR of iso_c_mask[p][a] for all a... no, it depends on S.
#   
# Alternative vectorized approach:
#   Precompute a 2D array: pair_iso_mask[i][j] = iso_c_mask[i][j] as a tuple of 4 uint64s
#   Then for a set S (as bitmask), can_add(p) means:
#     For all a where S has bit a: (pair_iso_mask[p][a] & S_mask) == 0
#   We can compute: for each p, the "bad" set = OR over a in S of pair_iso_mask[p][a]
#   Then can_add(p) iff bad_set & S_mask == 0

# Let's store iso_c_mask as numpy array of shape (N_TOTAL, N_TOTAL, 4) of uint64
iso_c_arr = np.zeros((N_TOTAL, N_TOTAL, 4), dtype=np.uint64)
for i in range(N_TOTAL):
    for j in range(N_TOTAL):
        m = iso_c_mask[i][j]
        iso_c_arr[i, j, 0] = m & 0xFFFFFFFFFFFFFFFF
        iso_c_arr[i, j, 1] = (m >> 64) & 0xFFFFFFFFFFFFFFFF
        iso_c_arr[i, j, 2] = (m >> 128) & 0xFFFFFFFFFFFFFFFF
        iso_c_arr[i, j, 3] = (m >> 192) & 0xFFFFFFFFFFFFFFFF

def _mask_to_arr(m):
    """Convert Python int bitmask to numpy array of 4 uint64."""
    return np.array([m & 0xFFFFFFFFFFFFFFFF,
                     (m >> 64) & 0xFFFFFFFFFFFFFFFF,
                     (m >> 128) & 0xFFFFFFFFFFFFFFFF,
                     (m >> 192) & 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)

def _arr_and_check(arr_a, arr_b):
    """Check if any bit is set in both arr_a and arr_b (element-wise AND, then OR)."""
    return np.any(np.any(arr_a & arr_b))

def _can_add_vec(p, S_mask, S_indices):
    """Vectorized check: can point p be added to set S (indices in S_indices, mask S_mask)?"""
    if len(S_indices) < 2:
        return True
    # For all a in S: (iso_c_mask[p][a] & S_mask) == 0
    # Get all iso_c_mask[p][a] for a in S_indices
    masks = iso_c_arr[p, S_indices, :]  # shape (|S|, 4)
    S_mask_arr = _mask_to_arr(S_mask)  # shape (4,)
    # Check: for each row, does (row & S_mask_arr) have any nonzero?
    and_result = masks & S_mask_arr  # shape (|S|, 4)
    return not np.any(and_result)

def _blocked_by_vec(p, S_mask, S_indices):
    """Vectorized check: is point p blocked by set S?"""
    if len(S_indices) < 2:
        return False
    # p is blocked iff exists a in S: (iso_c_mask[a][p] & S_mask) != 0
    # Note: iso_c_mask[a][p] contains b such that (a,p,b) is isosceles
    # We need b in S, b != a, b != p. Since p not in S, b != p is automatic.
    # But b could equal a? No, because in precompute we excluded k==i and k==j.
    masks = iso_c_arr[S_indices, p, :]  # shape (|S|, 4)
    S_mask_arr = _mask_to_arr(S_mask)
    and_result = masks & S_mask_arr
    return np.any(and_result)

def _is_no_isosceles_vec(S_mask, S_indices):
    """Check if set S is no-isosceles."""
    n = len(S_indices)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            a = S_indices[i]
            b = S_indices[j]
            # Check if any c in S is in iso_c_mask[a][b]
            mask_ab = iso_c_arr[a, b, :]
            S_mask_arr = _mask_to_arr(S_mask)
            if np.any(mask_ab & S_mask_arr):
                return False
    return True

def _is_maximal_vec(S_mask, S_indices):
    """Check if S is maximal: no point outside S is unblocked."""
    S_set = set(S_indices)
    for p in range(N_TOTAL):
        if p in S_set:
            continue
        if not _blocked_by_vec(p, S_mask, S_indices):
            return False
    return True

def _greedy_vectorized(deadline_margin=15):
    """
    Vectorized greedy: at each step, compute for ALL addable points how many
    unblocked points they would block, pick the best.
    """
    S_indices = []
    S_mask = 0
    S_set = set()
    
    while True:
        # Find all addable points
        addable = []
        for p in range(N_TOTAL):
            if p in S_set:
                continue
            if _can_add_vec(p, S_mask, S_indices):
                addable.append(p)
        
        if not addable:
            break
        
        # Check if already maximal
        if _is_maximal_vec(S_mask, S_indices):
            break
        
        # Find unblocked points
        unblocked = []
        for p in range(N_TOTAL):
            if p in S_set:
                continue
            if not _blocked_by_vec(p, S_mask, S_indices):
                unblocked.append(p)
        
        if not unblocked:
            break
        
        # For each addable point, compute how many unblocked points it blocks
        best_p = None
        best_gain = -1
        for p in addable:
            gain = 1  # blocks itself
            # For each unblocked q, check if q would be blocked by S ∪ {p}
            # q blocked by S∪{p} iff:
            #   (q already blocked by S) OR (exists a in S: (q,p,a) isosceles) OR (exists a in S: (q,a,p) isosceles)
            # Since q is unblocked by S, the first condition is false.
            # (q,p,a) isosceles iff a in iso_c_mask[q][p] (i.e., q in iso_c_mask[a][p]... wait)
            # (q,p,a) isosceles iff there exists a such that the triple is isosceles.
            # From precompute: iso_c_mask[q][p] = set of k such that (q,p,k) is isosceles.
            # So (q,p,a) isosceles iff a in iso_c_mask[q][p].
            # Also (q,a,p) isosceles iff p in iso_c_mask[q][a], i.e., a such that iso_c_mask[q][a] contains p.
            # But iso_c_mask is symmetric in the sense that (q,a,p) isosceles iff (q,p,a) isosceles (same triple).
            # So we just need: exists a in S such that a in iso_c_mask[q][p].
            # i.e., (S_mask & iso_c_mask[q][p]) != 0
            
            q_mask_arr = _mask_to_arr(S_mask)
            for q in unblocked:
                if q == p:
                    continue
                # Check if q would be blocked by adding p
                # q blocked iff exists a in S: (q,p,a) isosceles iff a in iso_c_mask[q][p]
                mask_qp = iso_c_arr[q, p, :]
                if np.any(mask_qp & q_mask_arr):
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        S_indices.append(best_p)
        S_mask |= (1 << best_p)
        S_set.add(best_p)
        
        if time.time() > DEADLINE - deadline_margin:
            return None
    
    return S_indices

def _reduce_to_minimal_vec(S_indices, deadline_margin=10):
    """Iteratively remove points if remaining set is still maximal and no-isosceles."""
    S = list(S_indices)
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
            if _is_no_isosceles_vec(S_mask_test, S_test) and _is_maximal_vec(S_mask_test, S_test):
                S = S_test
                changed = True
                break
        if time.time() > DEADLINE - deadline_margin:
            break
    return S

def _local_search_vec(S_indices, deadline_margin=5):
    """Try to reduce S by swap operations."""
    S = list(S_indices)
    best_size = len(S)
    
    for iteration in range(100):
        if time.time() > DEADLINE - deadline_margin:
            break
        
        improved = False
        n_S = len(S)
        
        for i in range(n_S):
            S_without = S[:i] + S[i+1:]
            S_mask_without = 0
            for x in S_without:
                S_mask_without |= (1 << x)
            
            if not _is_no_isosceles_vec(S_mask_without, S_without):
                continue
            
            # Try to add a different point
            for p_add in range(N_TOTAL):
                if (S_mask_without >> p_add) & 1:
                    continue
                if not _can_add_vec(p_add, S_mask_without, S_without):
                    continue
                S_new = S_without + [p_add]
                S_mask_new = S_mask_without | (1 << p_add)
                if not _is_no_isosceles_vec(S_mask_new, S_new):
                    continue
                if not _is_maximal_vec(S_mask_new, S_new):
                    continue
                reduced = _reduce_to_minimal_vec(S_new, deadline_margin=3)
                if len(reduced) < best_size:
                    best_size = len(reduced)
                    S = reduced
                    improved = True
                    break
                if time.time() > DEADLINE - deadline_margin:
                    break
            if improved or time.time() > DEADLINE - deadline_margin:
                break
        
        if not improved:
            break
    
    return S

def construct():
    start_time = time.time()
    
    best_set = None
    best_size = N_TOTAL + 1
    
    rng = random.Random(42)
    
    # Strategy 1: Multiple greedy restarts with vectorized checking
    for attempt in range(200):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        result = _greedy_vectorized(deadline_margin=10)
        if result is None:
            continue
        
        reduced = _reduce_to_minimal_vec(result, deadline_margin=8)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 8:
            return [_ALL_POINTS[i] for i in best_set]
    
    # Strategy 2: Local search on best
    if best_set is not None and best_size > 8:
        improved_set = _local_search_vec(best_set, deadline_margin=10)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    # Strategy 3: Random core + greedy extension
    for attempt in range(100):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        core_size = rng.choice([2, 3, 4])
        core = rng.sample(range(N_TOTAL), core_size)
        core_mask = 0
        for x in core:
            core_mask |= (1 << x)
        
        if not _is_no_isosceles_vec(core_mask, core):
            continue
        
        S_indices = list(core)
        S_mask = core_mask
        S_set = set(core)
        
        while True:
            if time.time() - start_time > DEADLINE - 15:
                break
            
            addable = []
            for p in range(N_TOTAL):
                if p in S_set:
                    continue
                if _can_add_vec(p, S_mask, S_indices):
                    addable.append(p)
            
            if not addable:
                break
            
            if _is_maximal_vec(S_mask, S_indices):
                break
            
            unblocked = []
            for p in range(N_TOTAL):
                if p in S_set:
                    continue
                if not _blocked_by_vec(p, S_mask, S_indices):
                    unblocked.append(p)
            
            if not unblocked:
                break
            
            best_p = None
            best_gain = -1
            S_mask_arr = _mask_to_arr(S_mask)
            for p in addable:
                gain = 1
                for q in unblocked:
                    if q == p:
                        continue
                    mask_qp = iso_c_arr[q, p, :]
                    if np.any(mask_qp & S_mask_arr):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            
            if best_p is None:
                break
            
            S_indices.append(best_p)
            S_mask |= (1 << best_p)
            S_set.add(best_p)
        
        if len(S_indices) < best_size:
            reduced = _reduce_to_minimal_vec(S_indices, deadline_margin=8)
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced[:]
        
        if best_size <= 8:
            break
    
    # Strategy 4: Final local search
    if best_set is not None and best_size > 8:
        improved_set = _local_search_vec(best_set, deadline_margin=5)
        if len(improved_set) < best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]
    
    if best_set is None:
        # Fallback: simple greedy
        chosen = []
        chosen_mask = 0
        chosen_set = set()
        for i in range(N_TOTAL):
            if i in chosen_set:
                continue
            if _can_add_vec(i, chosen_mask, chosen):
                chosen.append(i)
                chosen_mask |= (1 << i)
                chosen_set.add(i)
        best_set = _reduce_to_minimal_vec(chosen, deadline_margin=5)
    
    return [_ALL_POINTS[i] for i in best_set]
