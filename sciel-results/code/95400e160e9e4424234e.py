import time
import random
import sys

_DEADLINE = 85.0
_MAX_SIZE = 48
_N = 512

def _check_violation(C, idx):
    """
    Check if C[idx] is the mean of any subset of >=2 OTHER elements.
    Uses bitset DP.
    """
    a = C[idx]
    n = len(C)
    if n < 3:
        return False
    
    diffs = [C[j] - a for j in range(n) if j != idx]
    m = len(diffs)
    if m < 2:
        return False
    
    # Calculate bounds
    max_pos = sum(d for d in diffs if d > 0)
    max_neg = -sum(d for d in diffs if d < 0)
    offset = max_neg
    total_span = max_pos + max_neg + 1
    
    # If span is too large, fall back to small subset checks
    if total_span > 200000:
        # Check size 2
        for i in range(m):
            for j in range(i+1, m):
                if diffs[i] + diffs[j] == 0:
                    return True
        # Check size 3
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    if diffs[i] + diffs[j] + diffs[k] == 0:
                        return True
        # Check size 4
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    for l in range(k+1, m):
                        if diffs[i] + diffs[j] + diffs[k] + diffs[l] == 0:
                            return True
        return False
        
    target_bit = 1 << offset
    mask = (1 << total_span) - 1
    
    dp = [0] * (m + 1)
    dp[0] = target_bit
    
    for d in diffs:
        for k in range(m, 0, -1):
            if dp[k-1] == 0:
                continue
            if d >= 0:
                dp[k] |= (dp[k-1] << d) & mask
            else:
                dp[k] |= (dp[k-1] >> (-d)) & mask
            # Early exit
            if dp[k] & target_bit:
                return True
                
    return False

def _is_valid(C):
    for i in range(len(C)):
        if _check_violation(C, i):
            return False
    return True

def _fast_incremental_check(C, new_elem):
    """
    Fast check for whether adding new_elem to C creates a violation.
    Checks:
    1. new_elem is mean of subset of C (size 2, 3, 4)
    2. Any C[i] is mean of subset containing new_elem (size 2, 3)
    """
    a = new_elem
    n = len(C)
    C_set = set(C)
    
    # 1. new_elem is mean of 2 others: C[i] + C[j] = 2*a
    for i in range(n):
        for j in range(i+1, n):
            if C[i] + C[j] == 2 * a:
                return False
                
    # 2. new_elem is mean of 3 others: C[i] + C[j] + C[k] = 3*a
    for i in range(n):
        for j in range(i+1, n):
            rem = 3 * a - C[i] - C[j]
            if rem in C_set and rem != C[i] and rem != C[j]:
                return False
                    
    # 3. new_elem is mean of 4 others: C[i]+C[j]+C[k]+C[l] = 4*a
    # Use meet-in-the-middle: split into two pairs
    pair_sums = {}
    for i in range(n):
        for j in range(i+1, n):
            s = C[i] + C[j]
            if s not in pair_sums:
                pair_sums[s] = []
            pair_sums[s].append((i, j))
    
    for i in range(n):
        for j in range(i+1, n):
            target = 4 * a - C[i] - C[j]
            if target in pair_sums:
                for (k, l) in pair_sums[target]:
                    # Ensure all 4 indices are distinct
                    if len({i, j, k, l}) == 4:
                        return False
                        
    # 4. C[i] is mean of {new_elem, y}: C[i] = (a + y) / 2 => y = 2*C[i] - a
    for i in range(n):
        y = 2 * C[i] - a
        if y in C_set and y != C[i]:
            return False
            
    # 5. C[i] is mean of {new_elem, y, z}: 3*C[i] = a + y + z => y + z = 3*C[i] - a
    for i in range(n):
        target_sum = 3 * C[i] - a
        for j in range(n):
            if C[j] == C[i]:
                continue
            z = target_sum - C[j]
            if z in C_set and z != C[i] and z != C[j]:
                return False
                
    # 6. C[i] is mean of {new_elem, y, z, w}: 4*C[i] = a + y + z + w => y+z+w = 4*C[i]-a
    for i in range(n):
        target_sum = 4 * C[i] - a
        ci = C[i]
        others = [C[j] for j in range(n) if C[j] != ci]
        mo = len(others)
        if mo < 3:
            continue
        for j in range(mo):
            for k in range(j+1, mo):
                w = target_sum - others[j] - others[k]
                if w in C_set and w != ci and w != others[j] and w != others[k]:
                    return False
                    
    return True

def _generate_sidon_sequence(limit, target_size):
    """
    Generate a Sidon sequence (B2 sequence) within [1, limit].
    A Sidon sequence ensures all pairwise sums are distinct.
    This guarantees no element is the mean of two others.
    """
    seq = []
    for x in range(1, limit + 1):
        if len(seq) >= target_size:
            break
        
        valid = True
        # Check if adding x violates Sidon property
        # We need all sums a+b (a<=b) to be distinct.
        # Existing sums from seq
        existing_sums = set()
        for i in range(len(seq)):
            for j in range(i, len(seq)):
                existing_sums.add(seq[i] + seq[j])
                
        # Check new sums involving x
        # Sums: x+x, x+seq[i]
        if 2 * x in existing_sums:
            valid = False
        else:
            for y in seq:
                s = x + y
                if s in existing_sums:
                    valid = False
                    break
                # Note: We add to a temporary set? No, we just check against existing.
                # But wait, if we add x, the new set of sums will include x+x and x+y.
                # We must ensure x+x and x+y are not in existing_sums.
                # We also must ensure x+y1 != x+y2 which is true since y1!=y2.
                # And x+y != x+x which is true since y!=x (distinct elements).
                # So checking against existing_sums is sufficient.
        
        if valid:
            seq.append(x)
            
    return seq

def _greedy_construct(order, time_limit, initial_set=None):
    selected = initial_set[:] if initial_set else []
    start = time.time()
    
    # If initial set is provided, we assume it's valid. 
    # But to be safe, if it's large, we might want to verify once.
    # However, the Sidon generator is fast, and we can trust it for size 2.
    # We still need to check size >= 3 for the initial set.
    if initial_set and len(initial_set) > 0:
        if not _is_valid(initial_set):
            # Fallback to empty if initial set is invalid (shouldn't happen for small Sidon)
            selected = []
            
    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= _MAX_SIZE:
            break
        if x in selected:
            continue
            
        if _fast_incremental_check(selected, x):
            C = selected + [x]
            # Full check
            if _is_valid(C):
                selected.append(x)
    
    return selected

def _local_search_improve(C, time_limit):
    """
    Try to improve C by swapping elements or adding/removing.
    Returns improved set.
    """
    start = time.time()
    C = C[:]
    n = len(C)
    
    # Try to add elements
    for x in range(1, _N + 1):
        if time.time() - start > time_limit:
            break
        if x in C:
            continue
        if len(C) >= _MAX_SIZE:
            break
        if _fast_incremental_check(C, x):
            C_new = C + [x]
            if _is_valid(C_new):
                C = C_new
    
    # Try swaps: remove one, try to add another (iterative)
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        C_set = set(C)
        for i in range(len(C)):
            if time.time() - start > time_limit:
                break
            C_without_i = C[:i] + C[i+1:]
            C_without_i_set = set(C_without_i)
            for x in range(1, _N + 1):
                if time.time() - start > time_limit:
                    break
                if x in C_without_i_set:
                    continue
                if _fast_incremental_check(C_without_i, x):
                    C_candidate = C_without_i + [x]
                    if _is_valid(C_candidate):
                        C = C_candidate
                        improved = True
                        break
                        
    return C

def construct():
    start_time = time.time()
    best_set = []
    best_size = 0
    
    # Generate a strong Sidon sequence to use as a seed/core
    # Size 15-20 is a good balance. Larger Sidon sequences are harder to pack with others.
    sidon_core = _generate_sidon_sequence(_N, 18)
    
    # Pre-defined good orders
    orders = []
    
    # 1. Sidon core + Random rest
    remaining = [x for x in range(1, _N + 1) if x not in sidon_core]
    for seed in range(20):
        rng = random.Random(seed)
        rest = remaining[:]
        rng.shuffle(rest)
        orders.append(sidon_core + rest)
        
    # 2. Sidon core + Spread rest
    base = list(range(1, _N + 1, 10))
    random.Random(123).shuffle(base)
    remaining_spread = [x for x in range(1, _N + 1) if x not in set(base) and x not in sidon_core]
    random.Random(456).shuffle(remaining_spread)
    orders.append(sidon_core + base + remaining_spread)
    
    # 3. Lo-Hi alternating (without Sidon bias, to see if it beats Sidon seed)
    lo, hi = 1, _N
    oh = []
    while lo <= hi:
        oh.append(lo)
        if lo != hi: oh.append(hi)
        lo += 1
        hi -= 1
    orders.append(oh)
    
    # 4. Hi-Lo alternating
    lo, hi = 1, _N
    oh = []
    while lo <= hi:
        oh.append(hi)
        if lo != hi: oh.append(lo)
        lo += 1
        hi -= 1
    orders.append(oh)
    
    # 5. Random orders (no Sidon)
    for seed in range(10, 40):
        rng = random.Random(seed)
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        orders.append(order)
        
    # 6. Primes first
    def _is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i+2) == 0:
                return False
            i += 6
        return True
    primes = [x for x in range(1, _N + 1) if _is_prime(x)]
    rest = [x for x in range(1, _N + 1) if x not in set(primes)]
    random.Random(2468).shuffle(rest)
    orders.append(primes + rest)
    
    # Run greedy
    for i, order in enumerate(orders):
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 10:
            break
            
        time_limit = min(2.0, _DEADLINE - elapsed - 8)
        
        # If the order starts with Sidon core, we can pass it as initial_set to save checks?
        # The greedy function handles initial_set. 
        # But for random orders, initial_set is None.
        initial = None
        if i < 20: # First 20 orders are Sidon-based
             # We pass the Sidon core as initial set only if the order starts with it.
             # My order construction puts sidon_core at the front.
             # So we can pass it to skip the first 18 greedy steps' validation?
             # Actually, _greedy_construct will just add them one by one.
             # It's faster to just let it run, or pass initial_set=sidon_core if we are sure.
             # Let's pass initial_set=sidon_core for the first 20 orders to save time on validation of the core.
             initial = sidon_core
             
        result = _greedy_construct(order, time_limit, initial)
        
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
            
        if best_size == _MAX_SIZE:
            break
            
    # If we have time, try local search to improve
    if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 5:
        remaining_time = _DEADLINE - time.time() - 3
        if remaining_time > 2:
            improved = _local_search_improve(best_set, remaining_time)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
    
    # Try more random orders if time permits
    if best_size < _MAX_SIZE and time.time() - start_time < _DEADLINE - 5:
        for seed in range(100, 200):
            if time.time() - start_time > _DEADLINE - 3:
                break
            rng = random.Random(seed)
            order = list(range(1, _N + 1))
            rng.shuffle(order)
            result = _greedy_construct(order, 1.0)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result[:]
                if best_size == _MAX_SIZE:
                    break
    
    if not best_set:
        return [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    return best_set
