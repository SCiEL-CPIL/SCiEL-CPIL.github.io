import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Primes up to 59
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
    
    def get_divisors(n):
        divs = [1]
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                new_divs = []
                power = 1
                while n % p == 0:
                    n //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            if n == 1:
                break
        if n > 1:
            for d in divs[:]:
                divs.append(d * n)
        return sorted(divs)
    
    def calc_score(n):
        if n < 2:
            return 0
        divs = get_divisors(n)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    candidates = []
    
    # Candidate 1: Product of first 15 primes (Square-free)
    n_15 = 1
    for p in primes[:15]:
        n_15 *= p
    if n_15 <= 10**18:
        candidates.append(n_15)
        
    # Candidate 2: Product of first 14 primes, with 2 squared
    n_14_sq2 = 2 * 1
    for p in primes[1:14]: # 3..43
        n_14_sq2 *= p
    n_14_sq2 *= 2 # So 2^2 * 3 * ... * 43
    if n_14_sq2 <= 10**18:
        candidates.append(n_14_sq2)
        
    # Candidate 3: Product of first 14 primes, with 3 squared
    n_14_sq3 = 2 * 1
    for p in primes[1:14]:
        n_14_sq3 *= p
    n_14_sq3 *= 3 # So 2 * 3^2 * ... * 43
    if n_14_sq3 <= 10**18:
        candidates.append(n_14_sq3)
        
    # Candidate 4: Product of first 13 primes, with 2 and 3 squared
    n_13_sq23 = 4 * 9 * 1
    for p in primes[2:13]: # 5..41
        n_13_sq23 *= p
    if n_13_sq23 <= 10**18:
        candidates.append(n_13_sq23)
        
    # Candidate 5: Product of first 13 primes, with 2 cubed
    n_13_cub2 = 8 * 1
    for p in primes[1:13]:
        n_13_cub2 *= p
    if n_13_cub2 <= 10**18:
        candidates.append(n_13_cub2)

    # Candidate 6: Product of first 13 primes, with 3 cubed
    n_13_cub3 = 2 * 27 * 1
    for p in primes[2:13]:
        n_13_cub3 *= p
    if n_13_cub3 <= 10**18:
        candidates.append(n_13_cub3)

    # Candidate 7: Product of first 14 primes, with 5 squared
    n_14_sq5 = 2 * 3 * 25 * 1
    for p in primes[3:14]: # 7..43
        n_14_sq5 *= p
    if n_14_sq5 <= 10**18:
        candidates.append(n_14_sq5)

    # Candidate 8: Product of first 14 primes, with 2^2 * 3^2
    n_14_sq23 = 4 * 9 * 1
    for p in primes[2:14]: # 5..43
        n_14_sq23 *= p
    if n_14_sq23 <= 10**18:
        candidates.append(n_14_sq23)

    # Candidate 9: Product of first 13 primes, with 2^2 * 3^2 * 5^2
    n_13_sq235 = 4 * 9 * 25 * 1
    for p in primes[3:13]: # 7..41
        n_13_sq235 *= p
    if n_13_sq235 <= 10**18:
        candidates.append(n_13_sq235)

    # Candidate 10: Product of first 12 primes, with 2^2 * 3^2 * 5^2 * 7^2
    n_12_sq2357 = 4 * 9 * 25 * 49 * 1
    for p in primes[4:12]: # 11..37
        n_12_sq2357 *= p
    if n_12_sq2357 <= 10**18:
        candidates.append(n_12_sq2357)

    # Candidate 11: Product of first 11 primes, with 2^2 * 3^2 * 5^2 * 7^2 * 11^2
    n_11_sq2357_11 = 4 * 9 * 25 * 49 * 121 * 1
    for p in primes[5:11]: # 13..31
        n_11_sq2357_11 *= p
    if n_11_sq2357_11 <= 10**18:
        candidates.append(n_11_sq2357_11)

    # Candidate 12: Product of first 15 primes, with 2^2
    n_15_sq2 = 4 * 1
    for p in primes[1:15]: # 3..47
        n_15_sq2 *= p
    if n_15_sq2 <= 10**18:
        candidates.append(n_15_sq2)

    # Candidate 13: Product of first 15 primes, with 3^2
    n_15_sq3 = 2 * 9 * 1
    for p in primes[2:15]: # 5..47
        n_15_sq3 *= p
    if n_15_sq3 <= 10**18:
        candidates.append(n_15_sq3)

    # Candidate 14: Product of first 14 primes, with 2^2 * 5^2
    n_14_sq25 = 4 * 25 * 3 * 1
    for p in primes[2:14]: # 5..43, but we already have 5^2, so 7..43
        if p == 5:
            continue
        n_14_sq25 *= p
    if n_14_sq25 <= 10**18:
        candidates.append(n_14_sq25)

    # Candidate 15: Product of first 14 primes, with 2^2 * 3^2 * 5^2
    n_14_sq235 = 4 * 9 * 25 * 1
    for p in primes[3:14]: # 7..43
        n_14_sq235 *= p
    if n_14_sq235 <= 10**18:
        candidates.append(n_14_sq235)

    # Candidate 16: Product of first 13 primes, with 2^3 * 3^2
    n_13_cub2_sq3 = 8 * 9 * 1
    for p in primes[2:13]: # 5..41
        n_13_cub2_sq3 *= p
    if n_13_cub2_sq3 <= 10**18:
        candidates.append(n_13_cub2_sq3)

    # Candidate 17: Product of first 13 primes, with 2^2 * 3^3
    n_13_sq2_cub3 = 4 * 27 * 1
    for p in primes[2:13]: # 5..41
        n_13_sq2_cub3 *= p
    if n_13_sq2_cub3 <= 10**18:
        candidates.append(n_13_sq2_cub3)

    # Candidate 18: Product of first 12 primes, with 2^2 * 3^2 * 5^2 * 7^2 * 11^2
    n_12_sq2357_11 = 4 * 9 * 25 * 49 * 121 * 1
    for p in primes[5:12]: # 13..37
        n_12_sq2357_11 *= p
    if n_12_sq2357_11 <= 10**18:
        candidates.append(n_12_sq2357_11)

    # Candidate 19: Product of first 11 primes, with 2^2 * 3^2 * 5^2 * 7^2 * 11^2 * 13^2
    n_11_sq2357_11_13 = 4 * 9 * 25 * 49 * 121 * 169 * 1
    for p in primes[6:11]: # 17..31
        n_11_sq2357_11_13 *= p
    if n_11_sq2357_11_13 <= 10**18:
        candidates.append(n_11_sq2357_11_13)

    # Candidate 20: Product of first 10 primes, with 2^2 * 3^2 * 5^2 * 7^2 * 11^2 * 13^2 * 17^2
    n_10_sq2357_11_13_17 = 4 * 9 * 25 * 49 * 121 * 169 * 289 * 1
    for p in primes[7:10]: # 19..29
        n_10_sq2357_11_13_17 *= p
    if n_10_sq2357_11_13_17 <= 10**18:
        candidates.append(n_10_sq2357_11_13_17)

    best_n = candidates[0]
    best_score = -1
    
    for n in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    return best_n
