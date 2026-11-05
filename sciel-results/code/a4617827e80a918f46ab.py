import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Primes to consider
    # Product of first 15 primes is ~6e17, so 15 is a safe upper bound for distinct primes.
    # We can use up to 16 or 17 if exponents are small, but 15 is a good balance.
    # Let's use 16 primes to be safe, as 2*3*...*53 is approx 3e18?
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53
    # Let's check product of first 16 primes.
    # 2*3=6, *5=30, *7=210, *11=2310, *13=30030, *17=510510, *19=9699690
    # *23=223092870, *29=6469693230, *31=200560490130, *37=7420738134810
    # *41=304250263527210, *43=13082761331670030, *47=614889782588491410 (15 primes)
    # *53 = 32589158477190044730 > 10^18.
    # So max distinct primes is 15 if all exponents are 1.
    # If we use 16 primes, at least one exponent must be 0? No, we want to maximize score.
    # Using 15 primes with some exponents > 1 allows more divisors.
    # So we fix the set of primes to the first 15.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    K = len(primes)
    
    def calc_score(exponents):
        # Check product limit
        log_n = 0
        for p, e in zip(primes, exponents):
            log_n += e * math.log(p)
        if log_n > math.log(10**18):
            return -1  # Invalid
        
        # Generate divisors
        divs = [1]
        for p, e in zip(primes, exponents):
            if e == 0:
                continue
            new_divs = []
            # Optimization: only extend existing divs
            # Actually, the standard way is:
            # new_divs = [d * p^k for d in divs for k in range(e+1)]
            # This creates a list of size len(divs) * (e+1)
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

    # Initial state: All 1s
    current_exp = [1] * K
    current_score = calc_score(current_exp)
    best_exp = current_exp.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 1000.0  # Initial temperature
    cooling = 0.999
    min_T = 0.1
    
    # Pre-calculate logs for limit check speed? No, calc_score does it.
    # To speed up SA, we can make the move generation smarter.
    # Moves:
    # 1. Increment a random exponent (if product allows)
    # 2. Decrement a random exponent (if > 0)
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate move
        # Try a few moves per temperature step to reduce overhead
        moves_per_step = 5
        for _ in range(moves_per_step):
            idx = random.randint(0, K - 1)
            # Decide increment or decrement
            if random.random() < 0.7:
                # Increment
                new_exp = current_exp.copy()
                new_exp[idx] += 1
            else:
                # Decrement
                if current_exp[idx] <= 0:
                    continue
                new_exp = current_exp.copy()
                new_exp[idx] -= 1
            
            new_score = calc_score(new_exp)
            
            if new_score < 0:
                # Invalid, reject
                continue
                
            if new_score > current_score:
                # Accept
                current_exp = new_exp
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_exp = current_exp.copy()
            else:
                # Accept with probability exp((new-old)/T)
                delta = new_score - current_score
                if delta < 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_exp = new_exp
                        current_score = new_score
        
        T *= cooling
        
    # Final check: ensure best is valid and within limit
    # The calc_score returns -1 for invalid, so best_score should be >= 0
    # Construct n from best_exp
    n = 1
    for p, e in zip(primes, best_exp):
        n *= (p ** e)
        
    if n > 10**18:
        # Fallback to a known good squarefree number
        return 614889782588491410
        
    return n
