import time
import random

N = 128
MAX_TIME = 85.0  # safety margin for 90s wall-clock limit

# Precompute useful structures
# sum_pairs[x] = list of (a,b) with a<b, a+b=x, 1<=a<b<=128, x<=128
_sum_pairs = [[] for _ in range(129)]
for _a in range(1, 129):
    for _b in range(_a + 1, 129):
        _s = _a + _b
        if _s <= 128:
            _sum_pairs[_s].append((_a, _b))

def _make_bitset(s_set):
    """Build a bitset where bit (x-1) is set if x is in s_set."""
    bitset = 0
    for x in s_set:
        bitset |= (1 << (x - 1))
    return bitset

def _in_bitset(bitset, val):
    """Check if val is in the set represented by bitset."""
    if val < 1 or val > 128:
        return False
    return (bitset >> (val - 1)) & 1

def _fast_validity_check(s):
    """Check if the set s is valid (no forbidden triples)."""
    s_set = set(s)
    bitset = _make_bitset(s_set)
    s_list = sorted(s_set)
    n = len(s_list)
    
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n: break
        if a + s_list[i+1] > 128: break
        for j in range(i+1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128: break
            if not _in_bitset(bitset, ab):
                continue
            for k in range(j+1, n):
                c = s_list[k]
                if b + c > 128: break
                if _in_bitset(bitset, a + c) and _in_bitset(bitset, b + c):
                    return False
    return True

def _count_conflicts(s_set):
    """
    Returns a dictionary mapping each element in s_set to the number of 
    forbidden triples (a,b,c) in s_set that contain that element.
    """
    bitset = _make_bitset(s_set)
    s_list = sorted(s_set)
    n = len(s_list)
    counts = {x: 0 for x in s_list}
    
    for i in range(n):
        a = s_list[i]
        if i + 1 >= n:
            break
        if a + s_list[i+1] > 128:
            break
            
        for j in range(i + 1, n):
            b = s_list[j]
            ab = a + b
            if ab > 128:
                break
            if not _in_bitset(bitset, ab):
                continue
            for k in range(j + 1, n):
                c = s_list[k]
                bc = b + c
                if bc > 128:
                    break
                ac = a + c
                if _in_bitset(bitset, ac) and _in_bitset(bitset, bc):
                    counts[a] += 1
                    counts[b] += 1
                    counts[c] += 1
    return counts

def _can_add(bitset, x):
    """
    Check if adding x to the set represented by bitset keeps it valid.
    Uses the efficient bitmask approach from inspiration 1.
    """
    # Case 1: x is a member of the triple (a = x).
    # Need distinct b, c in S such that x+b, x+c, b+c are all in S U {x}.
    # Since x+b, x+c > x, they must be in S. b+c in S or b+c == x.
    # V = { y in S | x+y in S }
    V_mask = bitset & (bitset >> x)
    if V_mask.bit_count() >= 2:
        v_list = []
        temp = V_mask
        while temp:
            lsb = temp & -temp
            v_list.append(lsb.bit_length())  # 1-indexed
            temp -= lsb
        n_v = len(v_list)
        for i in range(n_v):
            b = v_list[i]
            for j in range(i + 1, n_v):
                c = v_list[j]
                s = b + c
                if s == x or ((bitset >> (s - 1)) & 1):
                    return False

    # Case 2: x is a sum of two members (a+b = x).
    # Need distinct a, b, c in S with a+b=x, a+c in S, b+c in S.
    for a, b in _sum_pairs[x]:
        if ((bitset >> (a - 1)) & 1) == 0:
            continue
        if ((bitset >> (b - 1)) & 1) == 0:
            continue
        # c in S \ {a,b} such that a+c in S and b+c in S
        U_mask = (bitset & (bitset >> a)) & (bitset & (bitset >> b))
        U_mask &= ~(1 << (a - 1))
        U_mask &= ~(1 << (b - 1))
        if U_mask != 0:
            return False

    return True

def _greedy_remove_until_valid(s_set, start_time, deadline_margin=5.0):
    """
    Start with s_set and remove elements with highest conflict count until valid.
    """
    s_set = set(s_set)
    while True:
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        counts = _count_conflicts(s_set)
        if not counts or all(v == 0 for v in counts.values()):
            break  # Valid
        
        # Find element with max count
        max_count = max(counts.values())
        candidates = [x for x, v in counts.items() if v == max_count]
        # Tie-break: remove smallest to preserve top-half density
        remove_el = min(candidates)
        s_set.remove(remove_el)
        
    return s_set

def _try_greedy_add(s_set, start_time, deadline_margin=5.0):
    """Try to greedily add elements to improve the set using _can_add."""
    s_set = set(s_set)
    bitset = _make_bitset(s_set)
    candidates = [x for x in range(1, N+1) if x not in s_set]
    candidates.sort()
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - deadline_margin:
        improved = False
        for cand in candidates[:]:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            if _can_add(bitset, cand):
                bitset |= (1 << (cand - 1))
                s_set.add(cand)
                candidates.remove(cand)
                improved = True
    
    return sorted(s_set)

def _try_remove_add(s_set, start_time, deadline_margin=5.0):
    """Try removing one element and adding one or more others."""
    s_set = set(s_set)
    best_s = set(s_set)
    best_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        remaining_set = s_set - {remove_el}
        bitset = _make_bitset(remaining_set)
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        # Try to add as many as possible from remaining_set
        current = set(remaining_set)
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            if _can_add(bitset, cand):
                bitset |= (1 << (cand - 1))
                current.add(cand)
        
        if len(current) > best_len:
            best_s = current
            best_len = len(current)
    
    return sorted(best_s)

def _local_search(S_set, start_time, max_passes=30, swap_limit=15, deadline_margin=5.0):
    """Improve S via additions and systematic swaps using _can_add."""
    S_set = set(S_set)
    bitset = _make_bitset(S_set)
    not_in_S = [x for x in range(1, 129) if x not in S_set]
    
    for _ in range(max_passes):
        if time.time() > start_time + MAX_TIME - deadline_margin:
            break

        improved = False
        random.shuffle(not_in_S)

        for x in not_in_S[:]:
            if time.time() > start_time + MAX_TIME - deadline_margin:
                break

            # Try to add x directly
            if _can_add(bitset, x):
                bitset |= (1 << (x - 1))
                S_set.add(x)
                not_in_S.remove(x)
                improved = True
                continue

            # Try swaps: remove y from S, add x
            S_list = list(S_set)
            random.shuffle(S_list)

            found_swap = False
            for y in S_list[:swap_limit]:
                if time.time() > start_time + MAX_TIME - deadline_margin:
                    break
                S_temp_mask = bitset ^ (1 << (y - 1))
                if _can_add(S_temp_mask, x):
                    bitset = S_temp_mask | (1 << (x - 1))
                    S_set.remove(y)
                    S_set.add(x)
                    not_in_S.append(y)
                    if x in not_in_S:
                        not_in_S.remove(x)
                    found_swap = True
                    improved = True
                    break

            if found_swap:
                break

        if not improved:
            break

    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Strategy: Start with full set, greedily remove conflicts, then add back.
    
    # 1. Greedy Removal Phase
    full_set = set(range(1, N+1))
    s_set = _greedy_remove_until_valid(full_set, start_time)
    
    # 2. Greedy Add Phase
    if time.time() - start_time < deadline - 5.0:
        s_set = set(_try_greedy_add(s_set, start_time))
        
    # 3. Local Search (Remove-Add) Phase
    if time.time() - start_time < deadline - 5.0:
        s_set = set(_try_remove_add(s_set, start_time))
        
    # 4. Local Search with swaps
    if time.time() - start_time < deadline - 5.0:
        s_set = _local_search(s_set, start_time, max_passes=30, swap_limit=15)
        
    best_s = sorted(s_set)
    best_len = len(best_s)
    
    # 5. Random Perturbation / Restart if time permits
    if time.time() - start_time < deadline - 10.0:
        for _ in range(20):  # More restarts possible with faster bitset ops
            if time.time() - start_time > deadline - 5.0:
                break
            # Randomly remove 1-3 elements before starting greedy removal
            perturbed_set = set(full_set)
            for _ in range(random.randint(1, 3)):
                if perturbed_set:
                    random_remove = random.choice(list(perturbed_set))
                    perturbed_set.remove(random_remove)
            
            s_set = _greedy_remove_until_valid(perturbed_set, start_time)
            
            if time.time() - start_time < deadline - 5.0:
                s_set = set(_try_greedy_add(s_set, start_time))
                
            if time.time() - start_time < deadline - 5.0:
                s_set = set(_try_remove_add(s_set, start_time))
                
            if time.time() - start_time < deadline - 5.0:
                s_set = _local_search(s_set, start_time, max_passes=20, swap_limit=10)
                
            if len(s_set) > best_len:
                best_s = sorted(s_set)
                best_len = len(s_set)
                
    # 6. Final polish: try to add any remaining elements
    if time.time() - start_time < deadline - 2.0:
        bitset = _make_bitset(best_s)
        for x in range(1, 129):
            if (bitset >> (x - 1)) & 1:
                continue
            if _can_add(bitset, x):
                bitset |= (1 << (x - 1))
                best_s.append(x)
        best_s = sorted(set(best_s))
        
    # 7. One more local search pass
    if time.time() - start_time < deadline - 2.0 and len(best_s) < 128:
        best_s = _local_search(best_s, start_time, max_passes=15, swap_limit=10)
        best_s = sorted(best_s)
        
    # Final validation
    if not _fast_validity_check(set(best_s)):
        # Fallback to known valid set (top half)
        best_s = list(range(63, 129))
        
    return [int(x) for x in best_s]
