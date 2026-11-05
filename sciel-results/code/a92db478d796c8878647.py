import time
import random
from collections import defaultdict

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

def _generate_candidates(max_val=512):
    """Generate a diverse set of candidates."""
    candidates = set()
    # Powers of 2
    i = 0
    while (1 << i) <= max_val:
        candidates.add(1 << i)
        i += 1
    # Powers of 3
    i = 0
    while (3 ** i) <= max_val:
        candidates.add(3 ** i)
        i += 1
    # 2^i * 3^j
    i = 0
    while (1 << i) <= max_val:
        val_i = 1 << i
        j = 0
        while val_i * (3 ** j) <= max_val:
            candidates.add(val_i * (3 ** j))
            j += 1
        i += 1
    # Add some random numbers to break structure
    for _ in range(100):
        candidates.add(random.randint(1, max_val))
    
    return sorted(candidates)

def _fast_check_k2(existing, x):
    """Check if x is the mean of any pair in existing.
    i.e., 2*x = a + b for some a, b in existing, a != b."""
    if len(existing) < 2:
        return True
    target = 2 * x
    # Use set for O(1) lookup
    ex_set = set(existing)
    for a in existing:
        b = target - a
        if b != a and b in ex_set:
            return False
    return True

def _fast_check_k3(existing, x):
    """Check if x is the mean of any triple in existing.
    i.e., 3*x = a + b + c for some distinct a, b, c in existing."""
    if len(existing) < 3:
        return True
    target = 3 * x
    n = len(existing)
    # O(n^2) check
    for i in range(n):
        for j in range(i + 1, n):
            a = existing[i]
            b = existing[j]
            c = target - a - b
            # c must be distinct from a and b
            if c == a or c == b:
                continue
            # Check if c is in existing
            # Optimization: if c > max(existing) or c < min(existing), skip
            if c > max(existing) or c < min(existing):
                continue
            # Linear search or set? Set is O(1)
            if c in set(existing):
                return False
    return True

def _full_validity_check_incr(existing, new_elem):
    """Full check: can we add new_elem to existing while maintaining validity?
    1. new_elem not mean of subset of existing (k>=2)
    2. For each existing a, a not mean of subset of (existing\{a})+{new_elem} (k>=2)
    """
    n = len(existing)
    existing_list = list(existing)
    
    # Check 1: new_elem not mean of subset of existing
    if n >= 2:
        # Build DP for existing
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in existing_list:
            for k in range(n, 0, -1):
                if dp[k-1]:
                    dp[k] |= (dp[k-1] << x)
        
        max_sum = sum(existing_list)
        for k in range(2, n + 1):
            target = k * new_elem
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    
    # Check 2: For each existing elem a
    for i, a in enumerate(existing_list):
        others = [x for j, x in enumerate(existing_list) if j != i]
        m = len(others)
        if m < 1:
            continue
        
        # Build DP for others
        dp_others = [0] * (m + 1)
        dp_others[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                if dp_others[k-1]:
                    dp_others[k] |= (dp_others[k-1] << x)
        
        max_sum_others = sum(others)
        # Subset T of others, size t, sum(T) + new_elem = (t+1)*a
        # sum(T) = (t+1)*a - new_elem
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
    deadline = start_time + 55.0  # Safety margin
    best_set = []
    best_len = 0
    
    candidates = _generate_candidates(512)
    
    # Strategy: Multiple Greedy Passes with different orderings
    # Pre-filter candidates using k=2 and k=3 checks against the growing set
    
    orders = [
        candidates,
        sorted(candidates, reverse=True),
    ]
    for _ in range(10):
        lst = candidates[:]
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
            
            # Fast pre-checks
            if not _fast_check_k2(current, x):
                continue
            if not _fast_check_k3(current, x):
                continue
            
            # Full check
            if _full_validity_check_incr(current, x):
                current.append(x)
                current_set.add(x)
        
        if len(current) > best_len:
            best_set = current[:]
            best_len = len(current)
    
    # Local Search: Try to expand best_set
    if best_set and time.time() < deadline:
        best_set_set = set(best_set)
        remaining = [x for x in range(1, 513) if x not in best_set_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in remaining:
                if time.time() > deadline:
                    break
                if x in best_set_set:
                    continue
                if _fast_check_k2(best_set, x) and _fast_check_k3(best_set, x):
                    if _full_validity_check_incr(best_set, x):
                        best_set.append(x)
                        best_set_set.add(x)
                        remaining.remove(x)
                        best_len = len(best_set)
                        improved = True
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
