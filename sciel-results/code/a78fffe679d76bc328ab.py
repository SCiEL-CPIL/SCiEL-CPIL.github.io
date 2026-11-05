import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Primes up to 47 (15 primes)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    num_primes = len(primes)
    
    def get_divisors(prime_powers):
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
        
    def calc_score(divs):
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score
        
    def evaluate(pows):
        """Returns (score, n) or (-1, -1) if invalid."""
        n = 1
        for p, e in pows:
            if e > 0:
                n *= (p ** e)
        if n > 10**18 or n < 2:
            return -1, -1
        divs = get_divisors(pows)
        s = calc_score(divs)
        return s, n
        
    # Initialize with primorial of first 15 primes
    best_pows = [(p, 1) for p in primes]
    best_score, best_n = evaluate(best_pows)
    
    if best_n < 2:
        return 2

    # Local search
    # We will try to increase exponents of smaller primes and decrease exponents of larger primes
    # to maintain n <= 10^18 while potentially increasing the score.
    
    improved = True
    rounds = 0
    max_rounds = 5000
    
    while improved and rounds < max_rounds:
        if time.time() > deadline:
            break
        improved = False
        rounds += 1
        
        current_pows = list(best_pows)
        current_n = best_n
        current_score = best_score
        
        # Try increasing exponent of each prime
        for i in range(num_primes):
            if time.time() > deadline:
                break
            p, e = current_pows[i]
            new_e = e + 1
            new_n = current_n * p
            
            # If new_n is too big, we must remove factors from larger primes
            if new_n > 10**18:
                # Try to reduce exponents of primes j > i
                temp_pows = list(current_pows)
                temp_pows[i] = (p, new_e)
                temp_n = new_n
                
                # Reduce largest primes first
                for j in range(num_primes - 1, i, -1):
                    if temp_n <= 10**18:
                        break
                    pj, ej = temp_pows[j]
                    if ej > 1:
                        # Reduce by 1
                        temp_pows[j] = (pj, ej - 1)
                        temp_n //= pj
                    elif ej == 1:
                        # Remove prime
                        temp_pows[j] = (pj, 0)
                        temp_n //= pj
                    else:
                        continue
                        
                if temp_n <= 10**18 and temp_n >= 2:
                    # Check if this state is worth evaluating
                    # Heuristic: Only evaluate if we actually changed something
                    if temp_pows != current_pows:
                        s, n = evaluate(temp_pows)
                        if s > current_score:
                            current_score = s
                            current_n = n
                            current_pows = temp_pows
                            improved = True
            else:
                # new_n is valid
                temp_pows = list(current_pows)
                temp_pows[i] = (p, new_e)
                s, n = evaluate(temp_pows)
                if s > current_score:
                    current_score = s
                    current_n = n
                    current_pows = temp_pows
                    improved = True
        
        best_pows = current_pows
        best_score = current_score
        best_n = current_n
        
    # Final check
    if best_n < 2:
        return 2
    return best_n
