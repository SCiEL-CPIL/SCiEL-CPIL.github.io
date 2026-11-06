import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin
    
    # Primes up to 100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    max_val = 10**18
    
    # Cache for scores to avoid recalculating
    score_cache = {}
    
    def get_score(n):
        if n in score_cache:
            return score_cache[n]
        
        # Factorize n
        temp = n
        factors = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate divisors
        divs = [1]
        for p, e in factors:
            new_divs = []
            mult = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * mult)
                mult *= p
            divs = new_divs
            
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        
        score_cache[n] = score
        return score

    # Initialize with the primorial (product of smallest primes)
    n = 1
    exponents = [0] * len(primes)
    for i, p in enumerate(primes):
        if n * p <= max_val:
            n *= p
            exponents[i] = 1
        else:
            break
            
    best_n = n
    best_score = get_score(n)
    
    current_exps = exponents[:]
    
    def eval_exps(exps):
        val = 1
        for i, e in enumerate(exps):
            if e > 0:
                # Check overflow
                if val > max_val // (primes[i] ** e):
                    return -1
                val *= primes[i] ** e
        if val > max_val:
            return -1
        return get_score(val)

    def update_best(exps):
        nonlocal best_n, best_score
        val = 1
        for i, e in enumerate(exps):
            if e > 0:
                val *= primes[i] ** e
        score = get_score(val)
        if score > best_score:
            best_score = score
            best_n = val

    # Systematic local search around the primorial
    # We will try changing one exponent at a time
    
    # First, a pass to try setting exponents to 0, 1, 2 for each prime
    # This explores the neighborhood of the square-free solution
    
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() - start_time < TIME_LIMIT - 5:
        improved = False
        
        # Randomly select a prime index to modify
        # Bias towards smaller primes as they have more impact on value
        # But we must ensure n <= max_val
        idx = random.randint(0, len(primes) - 1)
        
        old_e = current_exps[idx]
        
        # Candidate exponents: 0, 1, 2, and old_e +/- 1
        candidates = [0, 1, 2, old_e - 1, old_e + 1]
        candidates = [e for e in candidates if e >= 0]
        random.shuffle(candidates)
        
        for new_e in candidates:
            if new_e == old_e:
                continue
                
            current_exps[idx] = new_e
            score = eval_exps(current_exps)
            
            if score > best_score:
                best_score = score
                # Reconstruct n
                best_n = 1
                for i, e in enumerate(current_exps):
                    if e > 0:
                        best_n *= primes[i] ** e
                improved = True
                no_improve_count = 0
                break
            else:
                current_exps[idx] = old_e
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a more drastic move: swap a prime
                # Or just reset no_improve count to continue random search
                no_improve_count = 0
                
    # Final polish: Systematic check of single exponent changes from the best found state
    # Extract exponents from best_n
    final_exps = [0] * len(primes)
    temp = best_n
    for i, p in enumerate(primes):
        if temp % p == 0:
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            final_exps[i] = count
        else:
            if p * p > temp:
                break
        
    current_exps = final_exps[:]
    
    for _ in range(3): # Repeat polish a few times
        for idx in range(len(primes)):
            old_e = current_exps[idx]
            for new_e in [0, 1, 2, 3, old_e - 1, old_e + 1]:
                if new_e < 0 or new_e == old_e:
                    continue
                current_exps[idx] = new_e
                score = eval_exps(current_exps)
                if score > best_score:
                    best_score = score
                    best_n = 1
                    for i, e in enumerate(current_exps):
                        if e > 0:
                            best_n *= primes[i] ** e
                else:
                    current_exps[idx] = old_e
                    
    return best_n

