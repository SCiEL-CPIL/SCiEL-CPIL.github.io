"""Search-based solver using multiplicative-structure candidates and local search for mean-free subset problem."""
import time
import random

def _valid_mean_free(selected):
    """Returns True if the list 'selected' satisfies the condition that no element
    is the exact arithmetic mean of any subset of at least two OTHER elements.
    Uses bitset DP for speed."""
    n = len(selected)
    if n < 3:
        return True
    
    selected = list(set(selected))
    n = len(selected)
    if n < 3:
        return True
    
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        if m < 2:
            continue
        
        max_sum = sum(others)
        dp = [0] * (m + 1)
        dp[0] = 1
        
        for x in others:
            for k in range(m, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        
        for k in range(2, m + 1):
            target = k * a
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    return True

def _generate_23_candidates(max_val=512):
    """Generate all numbers of form 2^i * 3^j <= max_val."""
    candidates = set()
    i = 0
    while (1 << i) <= max_val:
        val_i = 1 << i
        j = 0
        while val_i * (3 ** j) <= max_val:
            candidates.add(val_i * (3 ** j))
            j += 1
        i += 1
    return sorted(candidates)

def _build_dp(existing):
    """Build DP bitsets for the existing set. Returns list of bitsets dp[k]."""
    n = len(existing)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in existing:
        for k in range(n, 0, -1):
            if dp[k-1]:
                dp[k] |= (dp[k-1] << x)
    return dp

def _fast_validity_check_incr(existing, new_elem):
    """Fast check: can we add new_elem to existing while maintaining validity?"""
    n = len(existing)
    existing_list = list(existing)
    
    # Check 1: new_elem not mean of subset of existing
    if n >= 2:
        dp = _build_dp(existing_list)
        max_sum = sum(existing_list)
        for k in range(2, n + 1):
            target = k * new_elem
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    
    # Check 2: For each existing elem a, check subsets of (existing \ {a}) ∪ {new_elem}
    for i, a in enumerate(existing_list):
        others = [x for j, x in enumerate(existing_list) if j != i]
        m = len(others)
        if m < 1:
            continue
        
        dp_others = _build_dp(others)
        max_sum_others = sum(others)
        
        for t in range(1, m + 1):
            k = t + 1
            target = k * a - new_elem
            if target < 0:
                continue
            if target > max_sum_others:
                break
            if dp_others[t] & (1 << target):
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    best_set = []
    best_len = 0
    
    candidates_23 = _generate_23_candidates(512)
    c23_set = set(candidates_23)
    
    # Strategy 1: Greedy over 2^i * 3^j candidates
    orders = [
        candidates_23,
        sorted(candidates_23, reverse=True),
    ]
    for _ in range(4):
        lst = candidates_23[:]
        random.shuffle(lst)
        orders.append(lst)
    
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        current_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if _fast_validity_check_incr(current, x):
                current.append(x)
                current_set.add(x)
                if len(current) > best_len:
                    best_set = current[:]
                    best_len = len(current)
    
    # Strategy 2: Expand with remaining numbers
    if best_set:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        remaining.sort(key=lambda x: (0 if x in c23_set else 1, x))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in remaining[:]:
                if time.time() > deadline:
                    break
                if x in best_set_set:
                    continue
                if _fast_validity_check_incr(best_set, x):
                    best_set.append(x)
                    best_set_set.add(x)
                    remaining.remove(x)
                    best_len = len(best_set)
                    improved = True
    
    # Strategy 3: Local Search - Swap and Add
    if best_set and time.time() < deadline:
        best_set_set = set(best_set)
        missing = [x for x in range(1, 513) if x not in best_set_set]
        
        # Try removing one element and adding up to 2
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            temp_set = best_set[:i] + best_set[i+1:]
            temp_set_set = set(temp_set)
            
            added_count = 0
            for x in missing:
                if time.time() > deadline:
                    break
                if x in temp_set_set:
                    continue
                if _fast_validity_check_incr(temp_set, x):
                    temp_set.append(x)
                    temp_set_set.add(x)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if len(temp_set) > best_len:
                best_set = temp_set
                best_len = len(best_set)
                best_set_set = set(best_set)
                missing = [x for x in range(1, 513) if x not in best_set_set]
        
        # Try removing two elements and adding up to 3
        if time.time() < deadline:
            for i in range(len(best_set)):
                if time.time() > deadline:
                    break
                for j in range(i+1, len(best_set)):
                    if time.time() > deadline:
                        break
                    temp_set = best_set[:i] + best_set[i+1:j] + best_set[j+1:]
                    temp_set_set = set(temp_set)
                    
                    added_count = 0
                    for x in missing:
                        if time.time() > deadline:
                            break
                        if x in temp_set_set:
                            continue
                        if _fast_validity_check_incr(temp_set, x):
                            temp_set.append(x)
                            temp_set_set.add(x)
                            added_count += 1
                            if added_count >= 3:
                                break
                    
                    if len(temp_set) > best_len:
                        best_set = temp_set
                        best_len = len(best_set)
                        best_set_set = set(best_set)
                        missing = [x for x in range(1, 513) if x not in best_set_set]
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
