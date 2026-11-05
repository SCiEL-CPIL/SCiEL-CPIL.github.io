import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute primes up to a reasonable limit
    # 10^18 is approx 2^60. 
    # Product of first 15 primes: 2*3*5*...*47 = 614889782588491410 < 10^18
    # Product of first 16 primes: * 53 = 3.25 * 10^19 > 10^18
    # So max square-free k is 15.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    def get_divisors_and_score(prime_powers):
        """
        Efficiently get score for a given factorization.
        prime_powers: list of (prime, exponent)
        Returns: (score, n, divs_count)
        """
        n = 1
        # Check bounds early
        for p, e in prime_powers:
            if e > 0:
                # Check if n * p^e exceeds 10^18
                # Instead of calculating full power, we can just multiply
                # but we need divisors anyway.
                pass
        
        # Generate divisors
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
        
        if not divs:
            return 0, 1, 0
            
        n = 1
        for p, e in prime_powers:
            if e > 0:
                n *= (p ** e)
                
        if n > 10**18 or n < 2:
            return -1, -1, 0
            
        divs.sort()
        
        if len(divs) < 2:
            return 0, n, len(divs)
            
        score = 0
        # Local variables for speed
        gcd = math.gcd
        d = divs
        for i in range(len(d) - 1):
            if gcd(d[i], d[i + 1]) == 1:
                score += 1
        return score, n, len(divs)

    def evaluate_factorization(powers):
        """Wrapper to return just score and n, or -1 if invalid."""
        s, n, _ = get_divisors_and_score(powers)
        return s, n

    def local_search(pows, max_rounds=200):
        """
        Perform local search.
        """
        current_pows = list(pows)
        current_score, current_n = evaluate_factorization(current_pows)
        if current_score < 0:
            return -1, -1
            
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            if time.time() > deadline:
                break
            improved = False
            rounds += 1
            
            # Try modifying exponents of existing primes
            for idx in range(len(current_pows)):
                if time.time() > deadline:
                    break
                p, e = current_pows[idx]
                
                # Try increasing exponent
                if e + 1 <= 60:
                    new_pows = list(current_pows)
                    new_pows[idx] = (p, e + 1)
                    # Quick check: n * p <= 10^18?
                    if current_n * p <= 10**18:
                        new_score, _ = evaluate_factorization(new_pows)
                        if new_score > current_score:
                            current_pows = new_pows
                            current_n *= p
                            current_score = new_score
                            improved = True
                            
                # Try decreasing exponent
                if e - 1 >= 1:
                    new_pows = list(current_pows)
                    new_pows[idx] = (p, e - 1)
                    new_n = current_n // p
                    if new_n >= 2:
                        new_score, _ = evaluate_factorization(new_pows)
                        if new_score > current_score:
                            current_pows = new_pows
                            current_n = new_n
                            current_score = new_score
                            improved = True
                            
                # Try removing prime entirely
                if e >= 1:
                    new_pows = [pp for pp in current_pows if pp[0] != p]
                    if new_pows:
                        new_n = current_n // (p ** e)
                        if new_n >= 2:
                            new_score, _ = evaluate_factorization(new_pows)
                            if new_score > current_score:
                                current_pows = new_pows
                                current_n = new_n
                                current_score = new_score
                                improved = True

            # Try adding a new prime factor
            if time.time() > deadline:
                break
            existing_primes = set(pp[0] for pp in current_pows)
            for p in primes:
                if p in existing_primes:
                    continue
                if current_n * p <= 10**18:
                    new_pows = current_pows + [(p, 1)]
                    new_score, _ = evaluate_factorization(new_pows)
                    if new_score > current_score:
                        current_pows = new_pows
                        current_n *= p
                        current_score = new_score
                        improved = True
                        # Don't break, maybe another prime fits too? 
                        # But adding primes increases n significantly. 
                        # Usually only one more prime fits.
                        break
                        # Actually, if we add a prime, n increases. 
                        # We should probably restart the loop or just continue.
                        # Breaking is safer to ensure we don't add too many at once 
                        # and miss a better combination of exponents.
        
        if current_n >= 2 and current_n <= 10**18:
            return current_score, current_n
        return -1, -1

    best_n = 2
    best_score = 0
    best_pows = [(2, 1)]
    
    # Precompute prefix products of primes
    prefix_prod = [1]
    for p in primes:
        prefix_prod.append(prefix_prod[-1] * p)
    
    # Find max k such that product of first k primes <= 10^18
    max_k = 0
    for i in range(1, len(primes) + 1):
        if prefix_prod[i] <= 10**18:
            max_k = i
        else:
            break
            
    # 1. Evaluate square-free products of first k primes
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        pows = [(primes[i], 1) for i in range(k)]
        s, n = evaluate_factorization(pows)
        if s > best_score and n > 0:
            best_score = s
            best_n = n
            best_pows = pows
            
    # 2. Structured search with exponent variations
    # Focus on the region where k is close to max_k
    for k in range(max(1, max_k - 5), max_k + 1):
        if time.time() > deadline:
            break
        # Base: all exponents 1
        base_pows = [(primes[i], 1) for i in range(k)]
        
        # Try increasing exponent of prime i by 1, 2, 3, 4
        for i in range(min(5, k)): # Only small primes have room for exponents
            for extra in range(1, 5):
                if time.time() > deadline:
                    break
                pows = list(base_pows)
                pows[i] = (primes[i], 1 + extra)
                
                # Check if it fits
                n_val = 1
                for p, e in pows:
                    n_val *= (p ** e)
                
                if n_val <= 10**18:
                    s, n = evaluate_factorization(pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                        best_pows = pows
                else:
                    # Try dropping largest primes until it fits
                    # This is a heuristic to find nearby valid configurations
                    # We drop from the end (largest primes)
                    current_pows = list(pows)
                    while n_val > 10**18 and len(current_pows) > i + 1:
                        # Remove the largest prime factor
                        last_p, last_e = current_pows[-1]
                        n_val //= (last_p ** last_e)
                        current_pows.pop()
                        
                        if n_val <= 10**18:
                            s, n = evaluate_factorization(current_pows)
                            if s > best_score and n > 0:
                                best_score = s
                                best_n = n
                                best_pows = current_pows
                            break
                            
    # 3. "Almost square-free": drop one large prime, increase exponent of 2
    for k in range(max(2, max_k - 3), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 20):
            if time.time() > deadline:
                break
            pows = [(primes[0], a)]
            for i in range(1, k - 1):
                pows.append((primes[i], 1))
            s, n = evaluate_factorization(pows)
            if s > best_score and n > 0:
                best_score = s
                best_n = n
                best_pows = pows
                
    # 4. Drop two large primes, increase exponents of 2 and 3
    for k in range(max(3, max_k - 3), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 15):
            if time.time() > deadline:
                break
            for b in range(1, 10):
                if time.time() > deadline:
                    break
                pows = [(primes[0], a), (primes[1], b)]
                for i in range(2, k - 2):
                    pows.append((primes[i], 1))
                s, n = evaluate_factorization(pows)
                if s > best_score and n > 0:
                    best_score = s
                    best_n = n
                    best_pows = pows

    # 5. Systematic search around known good region
    if max_k >= 12:
        for k in range(max(10, max_k - 2), max_k + 1):
            if time.time() > deadline:
                break
            for a in range(1, 10):
                if time.time() > deadline:
                    break
                for b in range(1, 6):
                    if time.time() > deadline:
                        break
                    for c in range(1, 4):
                        if time.time() > deadline:
                            break
                        pows = [(primes[0], a), (primes[1], b), (primes[2], c)]
                        for i in range(3, k - 1):
                            pows.append((primes[i], 1))
                        s, n = evaluate_factorization(pows)
                        if s > best_score and n > 0:
                            best_score = s
                            best_n = n
                            best_pows = pows

    # 6. Small primes with higher exponents (fallback)
    if time.time() < deadline - 30:
        for a in range(1, 61):
            if time.time() > deadline:
                break
            for b in range(1, a + 1):
                if time.time() > deadline:
                    break
                for c in range(1, b + 1):
                    if time.time() > deadline:
                        break
                    for d in range(1, c + 1):
                        if time.time() > deadline:
                            break
                        pows = [(2, a), (3, b), (5, c), (7, d)]
                        s, n = evaluate_factorization(pows)
                        if s > best_score and n > 0:
                            best_score = s
                            best_n = n
                            best_pows = pows

    # 7. LOCAL SEARCH: Try to improve the best candidate found so far
    if time.time() < deadline - 10:
        ls_score, ls_n = local_search(best_pows, max_rounds=100)
        if ls_score > best_score and ls_n > 0:
            best_score = ls_score
            best_n = ls_n
            # Update best_pows? local_search returns n, not pows. 
            # We need to recover pows or just use n. 
            # Since we return n, we don't strictly need pows for the final answer.
            # But for further search, we might. 
            # For now, just update best_n.
            
    # 8. Try more diverse starting points for local search
    if time.time() < deadline - 5:
        for k in [max_k, max_k - 1, max_k - 2]:
            if time.time() > deadline:
                break
            base_pows = [(primes[i], 1) for i in range(k)]
            ls_score, ls_n = local_search(base_pows, max_rounds=50)
            if ls_score > best_score and ls_n > 0:
                best_score = ls_score
                best_n = ls_n

    # Final check: ensure best_n is valid
    if best_n < 2:
        best_n = 2
        best_score = 1

    return best_n
