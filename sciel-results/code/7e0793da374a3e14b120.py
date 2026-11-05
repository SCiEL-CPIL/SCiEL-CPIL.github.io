import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
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
    
    def local_search(pows, max_rounds=50):
        """Perform local search by trying to adjust exponents with cached evaluation."""
        current_pows = list(pows)
        current_n = 1
        for p, e in current_pows:
            if e > 0:
                current_n *= (p ** e)
        if current_n > 10**18 or current_n < 2:
            return -1, -1
        
        # Cache for divisor scores
        score_cache = {}
        
        def get_cached_score(pows_tuple):
            key = tuple(sorted(pows_tuple))
            if key not in score_cache:
                s, n = evaluate_factorization(list(key))
                if n > 0:
                    score_cache[key] = s
                else:
                    score_cache[key] = -1
            return score_cache[key]
        
        current_score = get_cached_score(tuple(current_pows))
        
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            if time.time() > deadline:
                break
            improved = False
            rounds += 1
            
            # Generate all possible moves
            moves = []
            
            for idx in range(len(current_pows)):
                p, e = current_pows[idx]
                # Try increasing exponent
                if e + 1 <= 60:
                    new_pows = list(current_pows)
                    new_pows[idx] = (p, e + 1)
                    new_n = current_n * p
                    if new_n <= 10**18:
                        moves.append((new_pows, new_n))
                # Try decreasing exponent
                if e - 1 >= 1:
                    new_pows = list(current_pows)
                    new_pows[idx] = (p, e - 1)
                    new_n = current_n // p
                    if new_n >= 2:
                        moves.append((new_pows, new_n))
                # Try removing this prime entirely
                if e >= 1:
                    new_pows = [pp for pp in current_pows if pp[0] != p]
                    if new_pows:
                        new_n = current_n // (p ** e)
                        if new_n >= 2:
                            moves.append((new_pows, new_n))
            
            # Try adding a new prime factor
            for p in primes:
                if p in [pp[0] for pp in current_pows]:
                    continue
                if current_n * p <= 10**18:
                    new_pows = current_pows + [(p, 1)]
                    moves.append((new_pows, current_n * p))
            
            # Evaluate all moves and pick the best improvement
            best_move_score = current_score
            best_move_pows = None
            best_move_n = current_n
            
            for new_pows, new_n in moves:
                if time.time() > deadline:
                    break
                new_score = get_cached_score(tuple(new_pows))
                if new_score > best_move_score:
                    best_move_score = new_score
                    best_move_pows = new_pows
                    best_move_n = new_n
            
            if best_move_pows is not None:
                current_pows = best_move_pows
                current_n = best_move_n
                current_score = best_move_score
                improved = True
        
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
    for k in range(max(1, max_k - 6), max_k + 1):
        if time.time() > deadline:
            break
        for i in range(min(4, k)):
            for extra in range(1, 4):
                if time.time() > deadline:
                    break
                pows = [(primes[j], 1) for j in range(k)]
                pows[i] = (primes[i], 1 + extra)
                
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
                    for drop_start in range(k - 1, i, -1):
                        if drop_start >= k:
                            continue
                        new_pows = pows[:drop_start]
                        if i >= drop_start:
                            continue
                        n_val = 1
                        for p, e in new_pows:
                            n_val *= (p ** e)
                        if n_val <= 10**18 and n_val >= 2:
                            s, n = evaluate_factorization(new_pows)
                            if s > best_score and n > 0:
                                best_score = s
                                best_n = n
                                best_pows = new_pows
                        break
    
    # 3. Try "almost square-free": drop one large prime, increase exponent of 2
    for k in range(max(2, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 15):
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
    
    # 4. Try dropping two large primes, increase exponents of 2 and 3
    for k in range(max(3, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 10):
            for b in range(1, 5):
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
    if max_k >= 14:
        for k in range(max(12, max_k - 3), max_k + 1):
            if time.time() > deadline:
                break
            for a in range(1, 8):
                for b in range(1, 4):
                    for c in range(1, 3):
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
    
    # 6. Try small primes with higher exponents
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
    
    # 8. Try more diverse starting points for local search
    if time.time() < deadline - 5:
        # Start from primorial with some modifications
        for k in [max_k, max_k - 1, max_k - 2]:
            if time.time() > deadline:
                break
            base_pows = [(primes[i], 1) for i in range(k)]
            ls_score, ls_n = local_search(base_pows, max_rounds=30)
            if ls_score > best_score and ls_n > 0:
                best_score = ls_score
                best_n = ls_n
    
    # 9. NEW: Try configurations with non-consecutive primes (skip some large primes)
    if time.time() < deadline - 5:
        for k in range(max_k - 3, max_k + 1):
            if time.time() > deadline:
                break
            # Try different subsets of primes
            for skip1 in range(k - 1, max(0, k - 5), -1):
                if time.time() > deadline:
                    break
                for skip2 in range(skip1 - 1, max(0, k - 8), -1):
                    if time.time() > deadline:
                        break
                    pows = []
                    for i in range(k):
                        if i != skip1 and i != skip2:
                            pows.append((primes[i], 1))
                    s, n = evaluate_factorization(pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                        best_pows = pows
    
    # 10. NEW: Try asymmetric exponent distributions
    if time.time() < deadline - 5:
        for k in range(max(8, max_k - 2), max_k + 1):
            if time.time() > deadline:
                break
            # Give extra exponent to early primes, less to later
            for a in range(2, 6):
                for b in range(1, 4):
                    if time.time() > deadline:
                        break
                    pows = [(primes[0], a), (primes[1], b)]
                    for i in range(2, k):
                        pows.append((primes[i], 1))
                    s, n = evaluate_factorization(pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                        best_pows = pows
    
    # Final check: ensure best_n is valid
    if best_n < 2:
        best_n = 2
        best_score = 1
    
    return best_n
