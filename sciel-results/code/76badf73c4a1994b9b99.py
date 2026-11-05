import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    def get_divisors_fast(prime_powers):
        """Get sorted divisors from prime factorization as list of (prime, exponent) tuples."""
        divs = [1]
        for p, e in prime_powers:
            if e == 0:
                continue
            current_len = len(divs)
            power = 1
            for _ in range(e):
                power *= p
                for i in range(current_len):
                    divs.append(divs[i] * power)
        divs.sort()
        return divs

    def calc_score_from_divisors(divs):
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score

    def evaluate_factorization(powers):
        """powers: list of (prime, exponent). Returns (score, n) or (-1, -1) if invalid."""
        n = 1
        for p, e in powers:
            if e > 0:
                n *= (p ** e)
        if n > 10**18 or n < 2:
            return -1, -1
        divs = get_divisors_fast(powers)
        s = calc_score_from_divisors(divs)
        return s, n

    best_n = 2
    best_score = 0

    # 1. Find the max primorial that fits
    # Product of first k primes
    # P_15 = 614889782588491410
    # P_16 = 32589158477190044730 > 10^18
    
    # Base candidate: First 15 primes
    base_k = 15
    base_pows = [(primes[i], 1) for i in range(base_k)]
    base_s, base_n = evaluate_factorization(base_pows)
    if base_s > best_score:
        best_score = base_s
        best_n = base_n

    # 2. Try swapping the largest prime(s) with larger primes if they fit
    # Current product is P_15. We can replace p_15 (47) with p_j (j>14) if P_15 / 47 * p_j <= 10^18
    # P_15 / 47 = 13082761331669966. 10^18 / 13082761331669966 = 76.4
    # Primes in range [47, 76]: 53, 61, 71, 73
    for i in range(base_k):
        if time.time() > deadline: break
        p_orig = primes[i]
        # Only swap large primes to keep the small ones which are more effective
        if i < base_k - 3: 
            continue
        
        for j in range(i + 1, len(primes)):
            p_new = primes[j]
            if p_new > 100: break
            
            # Calculate new n
            n_val = 1
            for k_idx in range(base_k):
                if k_idx == i:
                    n_val *= p_new
                else:
                    n_val *= primes[k_idx]
            
            if n_val > 10**18:
                break # Primes are increasing, so larger j will be larger n
            
            pows = [(primes[k_idx], 1) for k_idx in range(base_k)]
            pows[i] = (p_new, 1)
            s, n = evaluate_factorization(pows)
            if s > best_score:
                best_score = s
                best_n = n

    # 3. Try dropping the largest prime and increasing the exponent of the smallest prime
    # e.g., 2^2 * P_14
    # P_14 = 13082761331669966
    # 2^2 * P_14 = 4 * 13082761331669966 = 52331045326679864 < 10^18
    # 3^2 * P_14 = 9 * ... > 10^18? 9 * 1.3e16 = 1.17e17 < 10^18.
    # 5^2 * P_14 = 25 * 1.3e16 = 3.2e17 < 10^18.
    # 7^2 * P_14 = 49 * 1.3e16 = 6.4e17 < 10^18.
    # 11^2 * P_14 = 121 * 1.3e16 = 1.5e18 > 10^18.
    
    # So we can drop p_15 (47) and square p_1..p_4 (2,3,5,7)
    for drop_idx in range(base_k - 2, base_k): # Drop last 2 primes potentially
        if time.time() > deadline: break
        remaining_pows = [(primes[i], 1) for i in range(drop_idx)]
        prod_rem = 1
        for p, e in remaining_pows:
            prod_rem *= p**e
            
        # Try increasing exponent of first few primes
        for inc_idx in range(min(5, drop_idx)):
            p_inc = primes[inc_idx]
            # Max exponent e such that prod_rem * p_inc^e <= 10^18
            max_e = 1
            while prod_rem * (p_inc ** (max_e + 1)) <= 10**18:
                max_e += 1
            
            for e in range(2, max_e + 1):
                pows = remaining_pows[:]
                pows[inc_idx] = (p_inc, e)
                s, n = evaluate_factorization(pows)
                if s > best_score:
                    best_score = s
                    best_n = n

    # 4. Try dropping 2 largest primes and increasing exponents of 2 and 3
    # Drop p_14, p_15. Remaining P_13.
    # P_13 = 304250263527210
    # Limit 10^18.
    # 2^a * 3^b * P_13 <= 10^18
    # P_13 * 2^a * 3^b <= 10^18
    # 2^a * 3^b <= 3286
    for a in range(1, 13): # 2^12 = 4096
        if time.time() > deadline: break
        for b in range(1, 9): # 3^8 = 6561
            if 2**a * 3**b > 3286:
                break
            pows = [(2, a), (3, b)]
            for i in range(2, 13):
                pows.append((primes[i], 1))
            s, n = evaluate_factorization(pows)
            if s > best_score:
                best_score = s
                best_n = n

    # 5. Local search on the best found n
    if time.time() < deadline - 10:
        # Reconstruct factorization of best_n
        temp_n = best_n
        start_pows = []
        for p in primes:
            e = 0
            while temp_n % p == 0:
                temp_n //= p
                e += 1
            if e > 0:
                start_pows.append((p, e))
        
        if start_pows:
            # Simple local search: try +/- 1 on exponents
            best_pows = start_pows[:]
            best_s, _ = evaluate_factorization(best_pows)
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for idx in range(len(best_pows)):
                    p, e = best_pows[idx]
                    # Try increase
                    if e < 20:
                        new_pows = best_pows[:]
                        new_pows[idx] = (p, e + 1)
                        s, n = evaluate_factorization(new_pows)
                        if s > best_s and n > 0:
                            best_pows = new_pows
                            best_s = s
                            best_n = n
                            improved = True
                            break
                    # Try decrease
                    if e > 1:
                        new_pows = best_pows[:]
                        new_pows[idx] = (p, e - 1)
                        s, n = evaluate_factorization(new_pows)
                        if s > best_s and n > 0:
                            best_pows = new_pows
                            best_s = s
                            best_n = n
                            improved = True
                            break
                    # Try swap with another prime
                    # This is expensive, skip for now unless we have lots of time
                # Try adding a new prime if not present
                if len(best_pows) < len(primes):
                    for i in range(len(primes)):
                        if any(p == primes[i] for p, _ in best_pows):
                            continue
                        new_pows = best_pows + [(primes[i], 1)]
                        s, n = evaluate_factorization(new_pows)
                        if s > best_s and n > 0:
                            best_pows = new_pows
                            best_s = s
                            best_n = n
                            improved = True
                            break
                # Try removing a prime
                if len(best_pows) > 5:
                    for idx in range(len(best_pows) - 1, -1, -1):
                        new_pows = best_pows[:idx] + best_pows[idx+1:]
                        s, n = evaluate_factorization(new_pows)
                        if s > best_s and n > 0:
                            best_pows = new_pows
                            best_s = s
                            best_n = n
                            improved = True
                            break

    if best_n < 2:
        best_n = 2
        best_score = 1

    return best_n
