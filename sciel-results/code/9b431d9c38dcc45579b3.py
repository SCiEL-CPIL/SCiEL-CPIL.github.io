"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import Counter

def construct():
    """
    Find 1..100 distinct integers strictly larger than 210 and at most 10210.
    Their product multiplied by 210 must be a perfect square.
    Minimize max(selected integers) - 210.
    """
    N = 210  # 2 * 3 * 5 * 7
    MAX_VAL = 10210
    
    # Precompute prime factorization for numbers up to MAX_VAL
    # We only need primes up to MAX_VAL
    primes = []
    is_prime = [True] * (MAX_VAL + 1)
    for i in range(2, MAX_VAL + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, MAX_VAL + 1, i):
                if j <= MAX_VAL:
                    is_prime[j] = False
    
    # factorize(n) returns a dict of prime -> exponent
    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors[p] = exp
        if temp > 1:
            factors[temp] = 1
        return factors
    
    # We want to select a set S such that product(S) * 210 is a square.
    # Let P = product(S).
    # P * 210 = square.
    # This implies that the parity of exponents in P * 210 is all even.
    # Let v_p(210) be the exponent of prime p in 210.
    # v_p(210) = 1 for p in {2,3,5,7}, 0 otherwise.
    # So for p in {2,3,5,7}, sum_{x in S} v_p(x) must be odd.
    # For p not in {2,3,5,7}, sum_{x in S} v_p(x) must be even.
    
    # Strategy:
    # Try to find small sets S.
    # k=1: x * 210 = square => x = 210 * k^2 / 210? No.
    # x * 210 = y^2 => x = y^2 / 210.
    # x must be integer > 210.
    # y^2 must be divisible by 210.
    # Let y = 210 * z. Then x = 210 * z^2.
    # Min x > 210 => z=1 => x = 210. Not > 210.
    # z=2 => x = 210 * 4 = 840.
    # Check: 840 * 210 = 176400 = 420^2. Yes.
    # Max - 210 = 840 - 210 = 630.
    
    # k=2: x * y * 210 = square.
    # x * y = 210 * k^2.
    # We want to minimize max(x, y).
    # Let x, y be factors of 210 * k^2.
    # x, y > 210, x != y.
    # To minimize max(x, y), we want x and y to be close to sqrt(210 * k^2) = 14.49 * k.
    # But x, y > 210.
    # So 14.49 * k > 210 => k > 14.49 / 14.49 ... wait.
    # sqrt(210) * k > 210 => k > sqrt(210) approx 14.49.
    # So k >= 15.
    
    # Let's iterate k from 15 upwards.
    # For each k, compute M = 210 * k^2.
    # Find factors d of M such that d > 210 and M/d > 210 and d != M/d.
    # We want to minimize max(d, M/d).
    # This is minimized when d is closest to sqrt(M).
    
    best_max = MAX_VAL
    best_set = []
    
    # Try k=1 solution first as a fallback
    # x = 840
    # max = 840, diff = 630
    current_best_diff = 630
    current_best_set = [840]
    
    # Iterate k
    for k in range(15, 200):  # Reasonable upper bound
        M = 210 * k * k
        # Find factors of M
        # We only need factors > 210
        # Iterate d from 1 to sqrt(M)
        # If M % d == 0:
        #   d1 = d
        #   d2 = M // d
        #   Check if d1 > 210 and d2 > 210 and d1 != d2
        #   If so, candidate max is max(d1, d2)
        
        # To be efficient, iterate d from 1 to int(sqrt(M))
        # But M can be large. k=100 => M = 2,100,000. sqrt ~ 1449.
        # This is fast.
        
        root = int(math.isqrt(M))
        for d in range(1, root + 1):
            if M % d == 0:
                d1 = d
                d2 = M // d
                if d1 == d2:
                    continue
                if d1 > 210 and d2 > 210:
                    if d1 <= MAX_VAL and d2 <= MAX_VAL:
                        m = max(d1, d2)
                        if m < best_max:
                            best_max = m
                            best_set = [d1, d2]
    
    # Check if we found a better k=2 solution
    if best_set:
        diff = best_max - 210
        if diff < current_best_diff:
            current_best_diff = diff
            current_best_set = best_set
    
    # Try k=3 solutions?
    # x * y * z * 210 = square.
    # x * y * z = 210 * k^2.
    # This is harder to search efficiently.
    # Given k=2 solution (224, 240) has diff 30, which is quite small.
    # 840 has diff 630.
    # Let's verify (224, 240) again.
    # 224 * 240 * 210 = 224 * 50400 = 11289600.
    # sqrt(11289600) = 3360.
    # 3360^2 = 11289600. Correct.
    # 224 > 210, 240 > 210. Distinct.
    # Max = 240. Diff = 30.
    
    # Can we do better than 30?
    # k=2: min max(x,y) for x*y = 210*k^2, x,y > 210.
    # We found (224, 240) for k=16.
    # Is there a k < 16?
    # k=15: M = 210 * 225 = 47250.
    # sqrt(47250) ~ 217.
    # Factors > 210:
    # 210 * 225 = 2 * 3 * 5 * 7 * 3^2 * 5^2 = 2 * 3^3 * 5^3 * 7.
    # Let's check divisors of 47250 near 217.
    # 47250 / 210 = 225. (210, 225) -> 210 not > 210.
    # 47250 / 215? No.
    # 47250 / 216? No.
    # 47250 / 217? No.
    # 47250 / 218? No.
    # 47250 / 219? No.
    # 47250 / 220? 47250/220 = 214.77 No.
    # 47250 / 224? No.
    # 47250 / 225 = 210. (225, 210) -> 210 not > 210.
    # 47250 / 226? No.
    # 47250 / 227? No.
    # 47250 / 228? No.
    # 47250 / 230? 47250/230 = 205.4 No.
    # 47250 / 231? 47250/231 = 204.5 No.
    # 47250 / 232? No.
    # 47250 / 233? No.
    # 47250 / 234? No.
    # 47250 / 235? 47250/235 = 201 No.
    # 47250 / 236? No.
    # 47250 / 237? No.
    # 47250 / 238? No.
    # 47250 / 239? No.
    # 47250 / 240? 47250/240 = 196.8 No.
    # 47250 / 241? No.
    # 47250 / 242? No.
    # 47250 / 243? No.
    # 47250 / 244? No.
    # 47250 / 245? 47250/245 = 192.8 No.
    # 47250 / 246? No.
    # 47250 / 247? No.
    # 47250 / 248? No.
    # 47250 / 249? No.
    # 47250 / 250 = 189 No.
    # So no k=15 pair.
    
    # k=16: M = 210 * 256 = 53760.
    # sqrt(53760) ~ 231.8.
    # Factors > 210:
    # 53760 / 224 = 240. (224, 240). Both > 210.
    # Max = 240. Diff = 30.
    # Any closer to 231.8?
    # 53760 / 230? No.
    # 53760 / 231? No.
    # 53760 / 232? 53760/232 = 231.7 No.
    # 53760 / 233? No.
    # 53760 / 234? No.
    # 53760 / 235? No.
    # 53760 / 236? No.
    # 53760 / 237? No.
    # 53760 / 238? No.
    # 53760 / 239? No.
    # 53760 / 240 = 224.
    # So (224, 240) is the best for k=16.
    
    # k=17: M = 210 * 289 = 60690.
    # sqrt ~ 246.
    # Factors > 210:
    # 60690 / 238 = 255. (238, 255). Max 255. Diff 45.
    # 60690 / 255 = 238.
    # Any closer?
    # 60690 / 240? No.
    # 60690 / 241? No.
    # 60690 / 242? No.
    # 60690 / 243? No.
    # 60690 / 244? No.
    # 60690 / 245? No.
    # 60690 / 246? No.
    # 60690 / 247? No.
    # 60690 / 248? No.
    # 60690 / 249? No.
    # 60690 / 250? No.
    # 60690 / 251? No.
    # 60690 / 252? No.
    # 60690 / 253? No.
    # 60690 / 254? No.
    # 60690 / 255 = 238.
    # So (238, 255) is best for k=17.
    
    # It seems (224, 240) is a strong candidate.
    # Is there a k=1 solution with max < 240?
    # x = 210 * z^2.
    # z=1 => 210 (not > 210).
    # z=2 => 840.
    # So no.
    
    # Is there a k=3 solution with max < 240?
    # x, y, z > 210.
    # x*y*z = 210 * k^2.
    # Min x*y*z = 211 * 212 * 213 = 9505356.
    # 210 * k^2 >= 9505356 => k^2 >= 45263 => k >= 212.
    # This is a large search space.
    # However, if max(x,y,z) < 240, then x,y,z in {211..239}.
    # There are 29 numbers.
    # We need to pick 3 distinct such that product * 210 is square.
    # Let's check if any such triple exists.
    
    # We can brute force triples from 211 to 239.
    # C(29, 3) = 29*28*27 / 6 = 3654 triples.
    # This is very fast.
    
    found_k3 = False
    best_k3_max = 240
    best_k3_set = []
    
    for i in range(211, 240):
        for j in range(i + 1, 240):
            for k in range(j + 1, 240):
                prod = i * j * k * 210
                root = math.isqrt(prod)
                if root * root == prod:
                    if k < best_k3_max:
                        best_k3_max = k
                        best_k3_set = [i, j, k]
                        found_k3 = True
    
    if found_k3:
        diff = best_k3_max - 210
        if diff < current_best_diff:
            current_best_diff = diff
            current_best_set = best_k3_set
    
    # Return the best set found
    return current_best_set

# EVOLVE-BLOCK-END
