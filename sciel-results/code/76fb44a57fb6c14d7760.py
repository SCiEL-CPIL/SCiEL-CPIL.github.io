"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    N = 210
    MAX_VAL = 10210
    
    # Precompute squarefree kernels for all numbers up to MAX_VAL
    sf = [0] * (MAX_VAL + 1)
    for i in range(1, MAX_VAL + 1):
        x = i
        kernel = 1
        d = 2
        while d * d <= x:
            count = 0
            while x % d == 0:
                x //= d
                count += 1
            if count % 2 == 1:
                kernel *= d
            d += 1
        if x > 1:
            kernel *= x
        sf[i] = kernel

    # We need product of selected numbers * 210 to be a perfect square
    # This means the squarefree kernel of (product * 210) must be 1
    # Equivalently, the XOR (mod 2) of the prime exponent vectors must cancel out
    # 210 = 2 * 3 * 5 * 7
    # So we need the product of selected numbers to have squarefree kernel that is a divisor of 210
    # and specifically, kernel(product) * 210 must be a perfect square
    # This means kernel(product) must equal 210 (since 210*210 = 210^2 is a square)
    # Wait: kernel(product) * 210 is a square iff kernel(product) = 210
    # Because kernel(a*b) = kernel(a) * kernel(b) / gcd(kernel(a), kernel(b))^2 ... 
    # Actually, sf(a*b) = sf(a) * sf(b) / gcd(sf(a), sf(b))^2 is not quite right.
    # Let me think again. sf(n) is the product of primes with odd exponent.
    # sf(a*b) = sf(a) * sf(b) / (product of primes appearing in both sf(a) and sf(b))^2
    # No, that's not right either. Let's use the vector representation.
    # Represent each number by a vector in F2^4 for primes 2,3,5,7 (and other primes must have even exponent)
    # For a number x to be usable, sf(x) must divide 210 (i.e., only primes 2,3,5,7 can appear with odd exponent)
    # The vector for x is the parity of exponents of 2,3,5,7 in x
    # The vector for 210 is (1,1,1,1)
    # We need the sum (XOR) of vectors of selected numbers to equal (1,1,1,1)
    
    # Map divisors of 210 to 4-bit vectors
    # Divisors: 1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210
    divisors = [1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210]
    div_to_vec = {}
    for d in divisors:
        vec = 0
        if d % 2 == 0:
            vec |= 1
            d2 = d // 2
            if d2 % 2 == 0:
                # This shouldn't happen for divisors of 210
                pass
        # Better approach: check each prime
        v = 0
        if d % 2 == 0:
            v |= 1
        if d % 3 == 0:
            v |= 2
        if d % 5 == 0:
            v |= 4
        if d % 7 == 0:
            v |= 8
        div_to_vec[d] = v
    
    # Target vector for 210
    target_vec = 15  # 1111 in binary
    
    # For each number x in (210, MAX_VAL], compute its vector
    # Only numbers whose sf(x) divides 210 are usable
    # Group numbers by their vector, keeping the smallest number for each vector
    # But we need distinct numbers, and we want to minimize the maximum
    
    # Actually, we want to find a set of distinct numbers > 210 such that:
    # 1. Each number's sf divides 210
    # 2. The XOR of their vectors equals 15
    # 3. Minimize the maximum number in the set
    
    # Strategy: iterate through numbers from 211 upward. For each number, if it's usable,
    # try to see if adding it to some combination of previously seen numbers can reach the target.
    # Use DP: reachable states (vectors) and the path taken.
    
    # Since there are only 16 possible vectors, we can do a BFS/DP over vectors.
    # For each vector, track the minimum possible maximum value to achieve it.
    
    # dp[v] = (min_max_value, list_of_numbers)
    # Start: dp[0] = (0, [])
    
    INF = float('inf')
    dp_val = [INF] * 16
    dp_path = [None] * 16
    dp_val[0] = 0
    dp_path[0] = []
    
    for x in range(N + 1, MAX_VAL + 1):
        k = sf[x]
        if N % k != 0:
            continue
        v = div_to_vec[k]
        
        # Try adding x to any reachable state
        new_dp_val = list(dp_val)
        new_dp_path = list(dp_path)
        
        for prev_v in range(16):
            if dp_val[prev_v] == INF:
                continue
            new_v = prev_v ^ v
            new_max = max(dp_val[prev_v], x)
            if new_max < new_dp_val[new_v]:
                new_dp_val[new_v] = new_max
                new_dp_path[new_v] = dp_path[prev_v] + [x]
        
        dp_val = new_dp_val
        dp_path = new_dp_path
        
        # Early termination if we've reached the target
        if dp_val[target_vec] != INF:
            result = dp_path[target_vec]
            # Verify
            prod = 1
            for num in result:
                prod *= num
            total = prod * N
            root = math.isqrt(total)
            if root * root == total:
                # Ensure all are distinct and > 210
                if len(result) == len(set(result)) and all(n > 210 for n in result):
                    return list(result)
    
    # Fallback
    return [4 * N]
# EVOLVE-BLOCK-END
