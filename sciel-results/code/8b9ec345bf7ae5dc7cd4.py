import math
import time

T12 = math.factorial(12)

def get_divisors(n):
    divisors = [1]
    # Prime factorization of 12!
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    for p, e in zip(primes, exponents):
        new_divs = []
        p_power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_power)
            p_power *= p
        divisors = new_divs
    return divisors

ALL_DIVISORS = get_divisors(T12)
VALID_DIVISORS = sorted([d for d in ALL_DIVISORS if d > 12])

def backtrack(target, cands):
    """
    Returns a list of factors from cands that multiply to target, or None.
    cands is a sorted list of unique integers.
    """
    if target == 1:
        return []
    if not cands:
        return None
    
    # Pruning: If the smallest candidate is greater than target, 
    # and target != 1, we can't form it with single factors? 
    # No, we can form it with multiple factors if they are smaller.
    # But if ALL candidates are > target, then no candidate can divide target 
    # (except if target is 1, handled). 
    # Wait, if c < target, c can divide target.
    # If c > target, c cannot divide target (since c is integer > 1).
    # So we can filter cands to only those <= target.
    
    # Since cands is sorted, we can break early if c > target?
    # If we are iterating, yes.
    
    for i in range(len(cands)):
        c = cands[i]
        if c > target:
            # Since cands is sorted ascending, all subsequent are > target.
            # They cannot divide target.
            break
        
        if target % c == 0:
            # Recurse with remaining candidates (ensuring distinctness)
            res = backtrack(target // c, cands[i+1:])
            if res is not None:
                return [c] + res
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # We want to minimize the largest factor M.
    # Iterate M from smallest valid divisor to largest.
    for M in VALID_DIVISORS:
        if time.time() > deadline:
            # Fallback: return a valid solution found earlier or a trivial one
            # Since we haven't returned, we haven't found a better M.
            # [T12] is always valid.
            return [T12]
        
        rem = T12 // M
        
        if rem == 1:
            return [M]
            
        # Candidates for the other factors must be:
        # 1. Divisors of T12
        # 2. > 12
        # 3. < M (distinct from M, and M is max)
        # 4. Must divide rem
        
        # Filter VALID_DIVISORS for these conditions
        # VALID_DIVISORS is sorted ascending.
        # We can stop the list comprehension when d >= M.
        candidates = []
        for d in VALID_DIVISORS:
            if d >= M:
                break
            if rem % d == 0:
                candidates.append(d)
                
        # If rem > 1, we need at least one factor.
        # If candidates is empty, we can't form rem.
        
        # To optimize backtrack, we can sort candidates in descending order?
        # If we pick larger factors first, we reduce 'rem' faster.
        # Let's try sorting candidates in descending order for the backtrack call.
        candidates.sort(reverse=True)
        
        decomposition = backtrack(rem, candidates)
        
        if decomposition is not None:
            # We found a valid set: decomposition + [M]
            # All distinct, all > 12, product T12.
            # M is the largest because all d in decomposition are < M.
            return decomposition + [M]
            
    return [T12]
