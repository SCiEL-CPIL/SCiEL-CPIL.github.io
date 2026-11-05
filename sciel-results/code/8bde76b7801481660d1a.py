import time
import random

def _is_valid_set(S):
    """
    Check if S is a valid set.
    Returns True if valid.
    """
    if len(S) < 3:
        return True
        
    S = list(S)
    N = len(S)
    
    for i in range(N):
        a = S[i]
        others = S[:i] + S[i+1:]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        max_sum = sum(others)
        
        dp = [0] * (n_others + 1)
        dp[0] = 1
        
        for val in others:
            for k in range(n_others - 1, -1, -1):
                if dp[k] == 0:
                    continue
                shifted = dp[k] << val
                dp[k+1] |= shifted
                
        for k in range(2, n_others + 1):
            if dp[k] == 0:
                continue
            target_sum = k * a
            if target_sum <= max_sum:
                if (dp[k] >> target_sum) & 1:
                    return False
            else:
                break
                
    return True

def _incremental_check(S, x):
    """
    Check if adding x to set S creates a violation.
    S is the current valid set, x is the new candidate.
    Returns True if S ∪ {x} is valid.
    """
    N = len(S)
    
    if N < 2:
        return True
    
    # Check 1: Is x the mean of any subset of S?
    # Need: sum of subset of S = k * x for some k >= 2
    max_sum_S = sum(S)
    dp = [0] * (N + 1)
    dp[0] = 1
    
    for val in S:
        for k in range(N - 1, -1, -1):
            if dp[k] == 0:
                continue
            shifted = dp[k] << val
            dp[k+1] |= shifted
    
    for k in range(2, N + 1):
        if dp[k] == 0:
            continue
        target_sum = k * x
        if target_sum <= max_sum_S:
            if (dp[k] >> target_sum) & 1:
                return False
        else:
            break
    
    # Check 2: Is any y in S the mean of a subset of S∪{x} that includes x?
    # Subset includes x and some elements from S\{y}
    # Sum = x + sum(subset of S\{y}) = k * y where k = 1 + |subset|
    # So sum(subset of S\{y}) = k*y - x for k >= 2 (since subset has at least 1 element, k>=2 means subset size >=1, but we need at least 2 OTHER elements total, so subset of S\{y} must have at least 1 element, making k>=2)
    
    for y in S:
        others_without_y = [v for v in S if v != y]
        n_others = len(others_without_y)
        
        if n_others < 1:
            continue
        
        # We need: x + sum(subset of others_without_y) = k * y
        # where k >= 2 (at least 2 other elements: x plus at least 1 from others_without_y)
        # So sum(subset) = k*y - x for k = 2, 3, ..., n_others+1
        # Subset size ranges from 1 to n_others
        
        max_sum_others = sum(others_without_y)
        
        dp2 = [0] * (n_others + 1)
        dp2[0] = 1
        
        for val in others_without_y:
            for k in range(n_others - 1, -1, -1):
                if dp2[k] == 0:
                    continue
                shifted = dp2[k] << val
                dp2[k+1] |= shifted
        
        for k in range(2, n_others + 2):  # k = 2 to n_others+1
            if k - 1 > n_others:
                break
            subset_size = k - 1  # number of elements from others_without_y
            if subset_size < 1 or subset_size > n_others:
                continue
            if dp2[subset_size] == 0:
                continue
            target_sum = k * y - x
            if target_sum < 0:
                continue
            if target_sum <= max_sum_others:
                if (dp2[subset_size] >> target_sum) & 1:
                    return False
    
    return True

def _fast_pair_check(S, x):
    """
    Quick check for pair violations involving x and S.
    Returns True if a pair violation is found (i.e., invalid).
    """
    S_set = set(S)
    
    # 1. x is mean of a, b in S: 2x = a + b
    for a in S:
        b = 2 * x - a
        if b in S_set and b != a:
            return True
            
    # 2. Some a in S is mean of x, b in S: 2a = x + b => b = 2a - x
    for a in S:
        b = 2 * a - x
        if b in S_set and b != a and b != x:
            return True
            
    return False

def _generate_structured_orders():
    """Generate various structured orderings."""
    candidates = list(range(1, 513))
    orders = []
    
    # Descending
    orders.append(candidates[::-1])
    
    # Ascending
    orders.append(candidates[:])
    
    # Interleaved hi-lo
    inter = []
    lo, hi = 1, 512
    while lo <= hi:
        inter.append(hi)
        if lo != hi:
            inter.append(lo)
        lo += 1
        hi -= 1
    orders.append(inter)
    
    # Interleaved lo-hi
    inter2 = []
    lo, hi = 1, 512
    while lo <= hi:
        inter2.append(lo)
        if lo != hi:
            inter2.append(hi)
        lo += 1
        hi -= 1
    orders.append(inter2)
    
    # Evens then odds
    evens = [i for i in range(1, 513) if i % 2 == 0]
    odds = [i for i in range(1, 513) if i % 2 == 1]
    orders.append(evens + odds)
    orders.append(odds + evens)
    
    # Multiples of 3, then non-multiples
    mult3 = [i for i in range(1, 513) if i % 3 == 0]
    non3 = [i for i in range(1, 513) if i % 3 != 0]
    orders.append(mult3 + non3)
    orders.append(non3 + mult3)
    
    # Large primes first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [i for i in range(1, 513) if is_prime(i)]
    non_primes = [i for i in range(1, 513) if not is_prime(i)]
    orders.append(primes + non_primes)
    
    # Powers of 2 and nearby
    powers = []
    p = 1
    while p <= 512:
        powers.append(p)
        p *= 2
    others = [i for i in range(1, 513) if i not in set(powers)]
    orders.append(powers + others)
    
    return orders

def construct():
    start_time = time.time()
    deadline = 85.0
    
    best_set = []
    best_len = 0
    
    candidates = list(range(1, 513))
    
    # Generate orders
    orders = _generate_structured_orders()
    
    # Add random orders
    rng = random.Random(42)
    num_random = 30
    for i in range(num_random):
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
    
    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
            
        current_set = []
        
        for x in order:
            if time.time() - start_time > deadline - 2:
                break
                
            if len(current_set) >= 48:
                break
                
            # Fast pair check first
            if _fast_pair_check(current_set, x):
                continue
                
            # Incremental check
            if _incremental_check(current_set, x):
                current_set.append(x)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
            
        if best_len >= 48:
            break
            
    # Local search: try to swap elements to improve
    if best_len < 48 and time.time() - start_time < deadline - 10:
        for _ in range(50):
            if time.time() - start_time > deadline - 5:
                break
            
            improved = False
            for i in range(len(best_set)):
                if time.time() - start_time > deadline - 5:
                    break
                for x in candidates:
                    if x in best_set:
                        continue
                    # Try swapping best_set[i] with x
                    new_set = best_set[:i] + [x] + best_set[i+1:]
                    if _incremental_check(best_set[:i] + best_set[i+1:], x) and _is_valid_set(new_set):
                        best_set = new_set
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    # Final verification
    if _is_valid_set(best_set):
        return best_set
    else:
        # Fallback: try to fix
        fixed = []
        for x in best_set:
            test = fixed + [x]
            if _is_valid_set(test):
                fixed.append(x)
        if fixed:
            return fixed
        return [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
