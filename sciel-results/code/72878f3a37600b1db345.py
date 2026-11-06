import math
import time

# Precompute 12!
T12 = math.factorial(12)

# Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
primes = [2, 3, 5, 7, 11]
exponents = [10, 5, 2, 1, 1]

def generate_divisors():
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        p_power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * p_power)
            p_power *= p
        divisors = new_divisors
    return divisors

ALL_DIVISORS = generate_divisors()
# Filter divisors > 12 and sort
VALID_DIVISORS = sorted([d for d in ALL_DIVISORS if d > 12])

# Precompute divisors of each valid divisor for faster lookup?
# Or just check modulo.
# Since the number of divisors is small, modulo is fine.

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # We iterate M from smallest to largest
    # M must be a divisor of 12! and > 12
    # We want the smallest M such that 12!/M can be decomposed
    # into distinct factors d where 12 < d < M.
    
    # To allow for the case where M is the only factor (k=1), 
    # the condition "distinct factors" is trivially satisfied for a single factor.
    # The problem says "1..12 DISTINCT integers".
    # If k=1, the list is [M]. M > 12. Product is M. 
    # If M = 12!, product is 12!. This is always a valid solution.
    # So the worst case score is 12!.
    
    # We want to minimize M.
    # Let's iterate M in VALID_DIVISORS.
    
    for M in VALID_DIVISORS:
        if time.time() > deadline:
            return [T12]
            
        rem = T12 // M
        
        if rem == 1:
            # Only one factor M
            return [M]
            
        # We need to decompose rem into distinct factors d_i such that:
        # 1. d_i > 12
        # 2. d_i < M
        # 3. d_i are distinct
        # 4. product(d_i) == rem
        
        # Candidates are divisors of rem that are > 12 and < M.
        # Since we want increasing order d_1 < d_2 < ... < d_{k-1} < M,
        # we can use the sorted VALID_DIVISORS and pick from indices.
        
        # Optimization: 
        # If rem < 13, we can't decompose it into factors > 12 (unless rem==1, handled above).
        if rem < 13:
            continue
            
        # DFS to find a decomposition of rem
        # State: (current_rem, start_idx)
        # start_idx: index in VALID_DIVISORS from which we can pick the next factor.
        # This ensures distinctness and increasing order.
        
        # We also need to ensure we don't exceed 12 factors total.
        # M is one factor. So we can pick at most 11 more factors.
        
        def can_decompose(current_rem, start_idx, depth):
            if current_rem == 1:
                return True
            if depth > 11: # Max 11 more factors
                return False
            if time.time() > deadline:
                return False
                
            # Pruning: 
            # The smallest available factor is VALID_DIVISORS[start_idx] if start_idx < len(VALID_DIVISORS)
            # If current_rem < VALID_DIVISORS[start_idx], we can't pick any factor.
            # Actually, we need a factor d such that d | current_rem.
            
            for i in range(start_idx, len(VALID_DIVISORS)):
                d = VALID_DIVISORS[i]
                if d >= M:
                    break
                if d > current_rem:
                    break
                if current_rem % d == 0:
                    if can_decompose(current_rem // d, i + 1, depth + 1):
                        return True
            return False

        if can_decompose(rem, 0, 0):
            # We found a valid M. Now we need to retrieve the factors.
            # We can modify the DFS to return the list of factors.
            
            def find_decomp(current_rem, start_idx, depth):
                if current_rem == 1:
                    return []
                if depth > 11:
                    return None
                if time.time() > deadline:
                    return None
                    
                for i in range(start_idx, len(VALID_DIVISORS)):
                    d = VALID_DIVISORS[i]
                    if d >= M:
                        break
                    if d > current_rem:
                        break
                    if current_rem % d == 0:
                        res = find_decomp(current_rem // d, i + 1, depth + 1)
                        if res is not None:
                            return [d] + res
                return None

            decomp = find_decomp(rem, 0, 0)
            if decomp is not None:
                return decomp + [M]
                
    return [T12]
