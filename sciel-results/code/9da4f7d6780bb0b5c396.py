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

# Precompute isosceles triples
# A triple (a,b,c) is isosceles if any two distances are equal
# For each pair (a,b), precompute which c make (a,b,c) isosceles
# iso_mask[a][b] = bitmask of c such that (a,b,c) is isosceles (c != a, c != b)
iso_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]

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
        iso_mask[i][j] = mask
        iso_mask[j][i] = mask

# For maximality: p is blocked by S iff ∃ a,b ∈ S such that (p,a,b) is isosceles
# i.e., p ∈ iso_mask[a][b] for some a,b ∈ S
# Equivalently: for each p, the set of pairs (a,b) that block p is:
#   block_pairs[p] = {(a,b) : p ∈ iso_mask[a][b]}
# p is blocked by S iff S contains at least one pair from block_pairs[p]

# Precompute: for each p, for each a ≠ p, the set of b ≠ p,a such that (p,a,b) is isosceles
# block_mask[p][a] = bitmask of b such that p ∈ iso_mask[a][b]
block_mask = [[0] * N_TOTAL for _ in range(N_TOTAL)]

for p in range(N_TOTAL):
    for a in range(N_TOTAL):
        if a == p:
            continue
        mask = 0
        for b in range(N_TOTAL):
            if b == p or b == a:
                continue
            if iso_mask[a][b] & (1 << p):
                mask |= (1 << b)
        block_mask[p][a] = mask

def _is_no_isosceles(S_mask, S_list):
    """Check if set S has no isosceles triple."""
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        for j in range(i + 1, n):
            a = S_list[i]
            b = S_list[j]
            # If any c in S (other than a,b) makes (a,b,c) isosceles, fail
            if iso_mask[a][b] & S_mask:
                return False
    return True

def _is_blocked(p, S_mask, S_list):
    """Check if point p is blocked by set S."""
    if len(S_list) < 2:
        return False
    for a in S_list:
        if S_mask & (1 << a):
            if S_mask & block_mask[p][a]:
                return True
    return False

def _is_maximal(S_mask, S_list):
    """Check if S is maximal."""
    for p in range(N_TOTAL):
        if S_mask & (1 << p):
            continue
        if not _is_blocked(p, S_mask, S_list):
            return False
    return True

def _can_add(p, S_mask, S_list):
    """Check if p can be added to S without creating an isosceles triple."""
    if len(S_list) < 2:
        return True
    # Adding p creates isosceles iff ∃ a,b in S: (p,a,b) is isosceles
    # i.e., p is blocked by S
    return not _is_blocked(p, S_mask, S_list)

def _greedy_maximal():
    """
    Build a maximal no-isosceles set by greedily adding points that block the most unblocked points.
    """
    S_list = []
    S_mask = 0
    
    while True:
        # Check if already maximal
        if _is_maximal(S_mask, S_list):
            break
        
        # Find all addable points
        addable = []
        for p in range(N_TOTAL):
            if S_mask & (1 << p):
                continue
            if _can_add(p, S_mask, S_list):
                addable.append(p)
        
        if not addable:
            break
        
        # Find unblocked points
        unblocked = []
        for p in range(N_TOTAL):
            if S_mask & (1 << p):
                continue
            if not _is_blocked(p, S_mask, S_list):
                unblocked.append(p)
        
        if not unblocked:
            break
        
        # Score each addable point by how many unblocked points it blocks
        best_p = None
        best_gain = -1
        for p in addable:
            gain = 1  # blocks itself
            for q in unblocked:
                if q == p:
                    continue
                # Check if q is blocked by S ∪ {p}
                # q blocked iff ∃ a,b in S∪{p}: (q,a,b) isosceles
                # Since q is currently unblocked, no pair in S blocks it.
                # So we need: ∃ a in S: (q,p,a) isosceles
                blocked = False
                for a in S_list:
                    if iso_mask[p][a] & (1 << q):
                        blocked = True
                        break
                    if iso_mask[q][a] & (1 << p):
                        blocked = True
                        break
                if blocked:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_p = p
        
        if best_p is None:
            break
        
        S_list.append(best_p)
        S_mask |= (1 << best_p)
    
    return S_list

def _reduce(S_list, deadline_margin=10):
    """Try to remove points while maintaining maximality and no-isosceles."""
    S = list(S_list)
    changed = True
    while changed:
        changed = False
        for i in range(len(S)):
            if time.time() > DEADLINE - deadline_margin:
                return S
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
    return S

def _local_search(S_list, deadline_margin=5):
    """Local search: try removing a point and adding a different one."""
    S = list(S_list)
    best_size = len(S)
    
    for _ in range(100):
        if time.time() > DEADLINE - deadline_margin:
            break
        
        improved = False
        for i in range(len(S)):
            S_without = S[:i] + S[i+1:]
            S_mask_without = 0
            for x in S_without:
                S_mask_without |= (1 << x)
            
            if not _is_no_isosceles(S_mask_without, S_without):
                continue
            
            # Try adding a different point
            for p_add in range(N_TOTAL):
                if S_mask_without & (1 << p_add):
                    continue
                if not _can_add(p_add, S_mask_without, S_without):
                    continue
                S_new = S_without + [p_add]
                S_mask_new = S_mask_without | (1 << p_add)
                if _is_no_isosceles(S_mask_new, S_new) and _is_maximal(S_mask_new, S_new):
                    reduced = _reduce(S_new, deadline_margin=3)
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

def _random_greedy(rng, deadline_margin=15):
    """Greedy with randomized tie-breaking."""
    S_list = []
    S_mask = 0
    
    while True:
        if time.time() > DEADLINE - deadline_margin:
            return S_list
        
        if _is_maximal(S_mask, S_list):
            break
        
        addable = []
        for p in range(N_TOTAL):
            if S_mask & (1 << p):
                continue
            if _can_add(p, S_mask, S_list):
                addable.append(p)
        
        if not addable:
            break
        
        unblocked = []
        for p in range(N_TOTAL):
            if S_mask & (1 << p):
                continue
            if not _is_blocked(p, S_mask, S_list):
                unblocked.append(p)
        
        if not unblocked:
            break
        
        # Compute gains
        gains = []
        for p in addable:
            gain = 1
            for q in unblocked:
                if q == p:
                    continue
                blocked = False
                for a in S_list:
                    if iso_mask[p][a] & (1 << q) or iso_mask[q][a] & (1 << p):
                        blocked = True
                        break
                if blocked:
                    gain += 1
            gains.append((gain, p))
        
        if not gains:
            break
        
        # Pick from top 3 with some randomness
        gains.sort(reverse=True)
        top_k = min(3, len(gains))
        best_gain, best_p = rng.choice(gains[:top_k])
        
        S_list.append(best_p)
        S_mask |= (1 << best_p)
    
    return S_list

def construct():
    start_time = time.time()
    best_set = None
    best_size = N_TOTAL + 1
    rng = random.Random(42)
    
    # Phase 1: Greedy constructions with randomization
    for attempt in range(30):
        if time.time() - start_time > DEADLINE - 40:
            break
        
        result = _random_greedy(rng, deadline_margin=20)
        reduced = _reduce(result, deadline_margin=10)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 10:
            break
    
    # Phase 2: Local search on best
    if best_set is not None and best_size > 10:
        improved = _local_search(best_set, deadline_margin=20)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    # Phase 3: More random restarts
    for attempt in range(50):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        result = _random_greedy(rng, deadline_margin=15)
        reduced = _reduce(result, deadline_margin=5)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced[:]
        
        if best_size <= 10:
            break
    
    # Phase 4: Final local search
    if best_set is not None and best_size > 10:
        improved = _local_search(best_set, deadline_margin=5)
        if len(improved) < best_size:
            best_size = len(improved)
            best_set = improved[:]
    
    if best_set is None:
        # Fallback: simple greedy
        S_list = _greedy_maximal()
        best_set = _reduce(S_list, deadline_margin=2)
    
    return [_ALL_POINTS[i] for i in best_set]
