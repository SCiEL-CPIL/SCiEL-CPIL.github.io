"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin for 150s limit
    
    # 1. Generate enough primes
    # Product of first 15 primes is ~6e17. 
    # We allow up to ~16-17 primes with small exponents.
    # Let's generate first 30 primes to be safe for the search space.
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(150)
    
    # We restrict search to first 18 primes. 
    # Product of first 18 primes: 2*3*5*...*59 
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53*59
    # Log10: sum log10(p) for p in first 18 primes.
    # Approx 17.8. So product is ~6e17. 
    # If we use 19th prime (61), product is ~3.7e19 > 1e18.
    # So max distinct primes is 18 if all exponents are 1.
    # If we use higher exponents for small primes, we use fewer distinct primes.
    
    max_primes = 18
    
    # Initialize with squarefree product of first 15 primes
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
    init_exps = [1] * 15 + [0] * (max_primes - 15)
    
    # Calculate initial score
    def calc_score(exps):
        n = 1
        for i, e in enumerate(exps):
            if e > 0:
                n *= primes[i] ** e
        if n > 10**18 or n < 2:
            return -1
        
        # Generate divisors
        divs = [1]
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        
        divs.sort()
        
        # Calculate score
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_exps = init_exps[:]
    best_score = calc_score(best_exps)
    current_exps = best_exps[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    cooling_rate = 0.99999
    
    # Precompute prime powers for speed? 
    # The bottleneck is sorting divisors and gcd.
    # Number of divisors for n~10^18 can be up to ~100k-1M.
    # Sorting 100k ints is fast. GCD is fast.
    # Main cost is generating divisors.
    
    # To speed up, we can limit the number of iterations based on time.
    # Each iteration involves 1-2 evaluations.
    # If evaluation takes ~0.01s, we can do ~10k iterations in 100s.
    
    # Let's add a fast check for "too many divisors" to skip sorting if possible?
    # No, we need the score.
    
    # Optimization: 
    # If n is squarefree, divisors are subsets.
    # We can use bitmasks for small k? 
    # But k can be up to 18. 2^18 = 262k. 
    # Generating 262k divisors is okay.
    
    # Let's just run the loop.
    
    improved = True
    no_improve_count = 0
    
    while time.time() < deadline:
        if T < 0.01:
            # Switch to local search if cooled down
            if not improved:
                break
            
        # Pick a random prime index to change
        i = random.randint(0, max_primes - 1)
        
        # Pick a move: +1, -1, or swap with another prime?
        # Let's stick to +1/-1 for simplicity, but maybe allow jumping to 0 from high exp?
        # Or jump to a random exponent?
        # Let's try:
        move_type = random.random()
        if move_type < 0.4:
            # Increase
            new_exps = current_exps[:]
            new_exps[i] += 1
        elif move_type < 0.8:
            # Decrease
            if current_exps[i] > 0:
                new_exps = current_exps[:]
                new_exps[i] -= 1
            else:
                continue
        else:
            # Random jump
            new_exps = current_exps[:]
            new_exps[i] = random.randint(0, 5)
            
        new_score = calc_score(new_exps)
        
        if new_score < 0:
            continue
            
        delta = new_score - current_score
        
        if delta > 0 or random.random() < math.exp(delta / T):
            current_exps = new_exps
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_exps = current_exps[:]
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
                
                # If we've gone a long time without improvement, maybe reset temperature?
                # Or just continue.
                
        T *= cooling_rate
        
        # Optional: If T is very low, do a local search pass to ensure we are at a local max
        if T < 0.1 and time.time() < deadline - 5:
             # Simple local search
             local_improved = True
             while local_improved and time.time() < deadline - 5:
                 local_improved = False
                 for j in range(max_primes):
                     # Try +1
                     temp_exps = current_exps[:]
                     temp_exps[j] += 1
                     s = calc_score(temp_exps)
                     if s > 0 and s > current_score:
                         current_exps = temp_exps
                         current_score = s
                         local_improved = True
                         if current_score > best_score:
                             best_score = current_score
                             best_exps = current_exps[:]
                         break
                     
                     # Try -1
                     if current_exps[j] > 0:
                         temp_exps = current_exps[:]
                         temp_exps[j] -= 1
                         s = calc_score(temp_exps)
                         if s > 0 and s > current_score:
                             current_exps = temp_exps
                             current_score = s
                             local_improved = True
                             if current_score > best_score:
                                 best_score = current_score
                                 best_exps = current_exps[:]
                             break
             
             if not local_improved:
                 # Restart with higher T if stuck in local max?
                 # Or just break if we've been stuck for a while?
                 # Let's restart T occasionally to escape.
                 if no_improve_count > 500:
                     T = 1.0
                     no_improve_count = 0
                     # Maybe perturb current solution?
                     # Keep it simple for now.
                     pass
                     
    # Final n
    final_n = 1
    for i in range(max_primes):
        if best_exps[i] > 0:
            final_n *= primes[i] ** best_exps[i]
            
    if final_n > 10**18 or final_n < 2:
        # Fallback to a known valid number
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
        
    return final_n
# EVOLVE-BLOCK-END
