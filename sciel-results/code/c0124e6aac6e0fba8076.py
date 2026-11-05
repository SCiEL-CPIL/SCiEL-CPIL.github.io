import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Primes to consider
    # Product of first 15 primes is ~6e17.
    # We can use up to 15 distinct primes.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    K = len(primes)
    
    # Precompute logs for fast limit check
    log_limit = math.log(10**18)
    log_primes = [math.log(p) for p in primes]
    
    def calc_score_fast(exponents):
        # Check product limit using logs for speed
        log_n = 0
        for i in range(K):
            e = exponents[i]
            if e:
                log_n += e * log_primes[i]
        if log_n > log_limit:
            return -1
        
        # Generate divisors
        divs = [1]
        for i in range(K):
            p = primes[i]
            e = exponents[i]
            if e == 0:
                continue
            new_divs = []
            # Optimization: generate powers once
            powers = [1] * (e + 1)
            for k in range(1, e + 1):
                powers[k] = powers[k-1] * p
            
            for d in divs:
                for pow_p in powers:
                    new_divs.append(d * pow_p)
            divs = new_divs
            
        divs.sort()
        
        # Count coprime adjacent pairs
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Initial state: Start with squarefree product of first 15 primes
    # This is a valid n <= 10^18 (approx 6.1e17)
    current_exp = [1] * K
    current_score = calc_score_fast(current_exp)
    best_exp = current_exp.copy()
    best_score = current_score
    
    # If initial is invalid (shouldn't be, but just in case), fallback
    if current_score < 0:
         current_exp = [1] * 14
         current_score = calc_score_fast(current_exp)
         best_exp = current_exp.copy()
         best_score = current_score

    # Greedy Hill Climbing with Random Restarts / Perturbations
    # We will try to improve by changing one exponent at a time.
    # To escape local optima, we will occasionally accept neutral or slightly worse moves
    # but primarily focus on strict improvements.
    
    # We can also try a "swap" move: increase one exponent and decrease another to keep n roughly same size.
    
    no_improve_count = 0
    max_no_improve = 50 # If we don't improve for 50 full passes, restart/perturb
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Try all possible increments
        for i in range(K):
            if time.time() > deadline:
                break
            new_exp = current_exp.copy()
            new_exp[i] += 1
            new_score = calc_score_fast(new_exp)
            
            if new_score > current_score:
                current_exp = new_exp
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_exp = current_exp.copy()
                    no_improve_count = 0
                # If we improved, we might want to restart the loop to check new neighbors
                # But for speed, we continue in this pass.
                # Actually, changing exp[i] affects neighbors. 
                # Let's just break and restart the outer loop to be safe and thorough.
                break
        
        if improved:
            continue
            
        # Pass 2: Try all possible decrements (if we are stuck, maybe we are too heavy on one prime?)
        for i in range(K):
            if time.time() > deadline:
                break
            if current_exp[i] <= 0:
                continue
            new_exp = current_exp.copy()
            new_exp[i] -= 1
            new_score = calc_score_fast(new_exp)
            
            if new_score > current_score:
                current_exp = new_exp
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_exp = current_exp.copy()
                    no_improve_count = 0
                break
        
        if improved:
            continue
            
        # Pass 3: Try "swaps" - increase one, decrease another
        # This explores the space of n with similar magnitude but different factorization
        # We only do this if we are stuck, as it's O(K^2)
        if no_improve_count > 10: # Do this less frequently
            for i in range(K):
                if time.time() > deadline:
                    break
                if current_exp[i] == 0:
                    continue
                for j in range(K):
                    if i == j:
                        continue
                    new_exp = current_exp.copy()
                    new_exp[i] -= 1
                    new_exp[j] += 1
                    new_score = calc_score_fast(new_exp)
                    
                    if new_score > current_score:
                        current_exp = new_exp
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_exp = current_exp.copy()
                            no_improve_count = 0
                        break
                if improved:
                    break
        
        if improved:
            continue
            
        # If no improvement found in this full cycle
        no_improve_count += 1
        
        # If stuck for a long time, perform a random perturbation to escape local optimum
        if no_improve_count > max_no_improve:
            # Perturb: change 1-3 random exponents significantly
            num_changes = random.randint(1, 3)
            for _ in range(num_changes):
                idx = random.randint(0, K - 1)
                # Bias towards small primes for perturbations?
                # Or just random.
                delta = random.choice([-2, -1, 1, 2])
                new_val = current_exp[idx] + delta
                if new_val >= 0:
                    current_exp[idx] = new_val
            
            # Check if perturbed state is valid
            perturbed_score = calc_score_fast(current_exp)
            if perturbed_score < 0:
                # Revert to best if invalid
                current_exp = best_exp.copy()
                current_score = best_score
            else:
                current_score = perturbed_score
                if current_score > best_score:
                    best_score = current_score
                    best_exp = current_exp.copy()
            
            no_improve_count = 0
            max_no_improve = random.randint(20, 100) # Randomize stagnation threshold

    # Final check: ensure best is valid and within limit
    # The calc_score returns -1 for invalid, so best_score should be >= 0
    # Construct n from best_exp
    n = 1
    for p, e in zip(primes, best_exp):
        n *= (p ** e)
        
    if n > 10**18 or n < 2:
        # Fallback to a known good squarefree number
        return 614889782588491410
        
    return n
