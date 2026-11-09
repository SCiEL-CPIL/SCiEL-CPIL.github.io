import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Primes up to 256
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, 257) if is_prime[i]]
    prime_set = set(primes)
    
    # We assume A is odd, B is even for 9 distinct odd sums.
    # a_i are odd, b_j are even.
    # s_ij = a_i + b_j are odd primes.
    # a_i = p_1i - b_i? No.
    # Let's stick to the d1, d2 approach.
    # p_11, p_12, p_13 are primes.
    # p_2j = p_1j - d1
    # p_3j = p_1j - d2
    
    # d1, d2 must be even.
    # a1 - a2 = d1, a1 - a3 = d2.
    # a1, a2, a3 distinct => d1, d2 non-zero and distinct (if d1=d2, a2=a3).
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5
    
    # Try to find a 9-sum solution
    found_9 = False
    
    # Limit d1, d2 to reasonable range
    # Max sum is 256. Min sum 1+2=3.
    # d1, d2 < 256.
    # But a_i in [1, 128].
    
    # Iterate d1, d2
    possible_d = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    
    for d1 in possible_d:
        for d2 in possible_d:
            if d1 == d2:
                continue
            
            # Iterate p11
            for p11 in primes:
                # Check p21, p31
                p21 = p11 - d1
                p31 = p11 - d2
                if p21 < 2 or p31 < 2 or not is_prime[p21] or not is_prime[p31]:
                    continue
                
                # Iterate p12
                for p12 in primes:
                    if p12 in [p11, p21, p31]:
                        continue
                    p22 = p12 - d1
                    p32 = p12 - d2
                    if p22 < 2 or p32 < 2 or not is_prime[p22] or not is_prime[p32]:
                        continue
                    # Check distinctness so far
                    current_primes = {p11, p21, p31, p12, p22, p32}
                    if len(current_primes) < 6:
                        continue
                    
                    # Iterate p13
                    for p13 in primes:
                        if p13 in current_primes:
                            continue
                        p23 = p13 - d1
                        p33 = p13 - d2
                        if p23 < 2 or p33 < 2 or not is_prime[p23] or not is_prime[p33]:
                            continue
                        
                        all_primes = current_primes | {p13, p23, p33}
                        if len(all_primes) < 9:
                            continue
                        
                        # All 9 are distinct primes.
                        # Now determine A and B.
                        # p11 = a1 + b1
                        # p12 = a1 + b2
                        # p13 = a1 + b3
                        # p21 = a2 + b1
                        # ...
                        
                        # b1 = p11 - a1
                        # b2 = p12 - a1
                        # b3 = p13 - a1
                        # a2 = p21 - b1 = p21 - (p11 - a1) = a1 + (p21 - p11) = a1 - d1. Correct.
                        # a3 = a1 - d2.
                        
                        # We need to find a1 such that:
                        # a1 is odd, in [1, 128]
                        # a1 - d1 is odd, in [1, 128]
                        # a1 - d2 is odd, in [1, 128]
                        # b1 = p11 - a1 is even, in [1, 128]
                        # b2 = p12 - a1 is even, in [1, 128]
                        # b3 = p13 - a1 is even, in [1, 128]
                        # b1, b2, b3 distinct.
                        # a1, a1-d1, a1-d2 distinct.
                        
                        # Since p11 is odd and a1 is odd, b1 is even.
                        # b1 >= 1 => a1 <= p11 - 1
                        # b1 <= 128 => a1 >= p11 - 128
                        # a1 >= 1
                        # a1 <= 128
                        
                        # Also a1 - d1 >= 1 => a1 >= d1 + 1
                        # a1 - d2 >= 1 => a1 >= d2 + 1
                        # a1 - d1 <= 128 => a1 <= 128 + d1
                        # a1 - d2 <= 128 => a1 <= 128 + d2
                        
                        # So a1 must be in [max(1, d1+1, d2+1, p11-128, p12-128, p13-128), min(128, p11-1, p12-1, p13-1, 128+d1, 128+d2)]
                        # And a1 must be odd.
                        
                        low = max(1, d1+1, d2+1, p11-128, p12-128, p13-128)
                        high = min(128, p11-1, p12-1, p13-1)
                        
                        if low > high:
                            continue
                        
                        # Find an odd a1 in [low, high]
                        if low % 2 == 0:
                            low += 1
                            
                        for a1 in range(low, high + 1, 2):
                            a2 = a1 - d1
                            a3 = a1 - d2
                            b1 = p11 - a1
                            b2 = p12 - a1
                            b3 = p13 - a1
                            
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            
                            # Check distinctness
                            if len(set(A)) < 3 or len(set(B)) < 3:
                                continue
                            
                            # Check bounds (already mostly checked, but explicit)
                            if not all(1 <= x <= 128 for x in A) or not all(1 <= x <= 128 for x in B):
                                continue
                            
                            # Check parity (A odd, B even)
                            if not all(x % 2 == 1 for x in A) or not all(x % 2 == 0 for x in B):
                                continue
                            
                            # Verify sums are the primes
                            sums = set()
                            for ai in A:
                                for bj in B:
                                    sums.add(ai + bj)
                            
                            if len(sums) == 9:
                                # Verify pairwise coprime (they are distinct primes, so yes)
                                return A, B
                            
                    if time.time() > deadline:
                        return best_A, best_B

    # If 9 not found, try to find 8 or better?
    # Or just return the seed.
    return best_A, best_B
