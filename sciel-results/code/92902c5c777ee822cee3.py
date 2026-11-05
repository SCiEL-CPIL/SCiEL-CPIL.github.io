import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Primes up to ~70
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    K = len(primes)
    LIMIT = 10**18
    
    def calc_score_from_exps(exps):
        """
        Calculate tau_perp(n) given the exponents of the first K primes.
        exps is a list of length K.
        Returns (score, n). Score is -1 if n > LIMIT.
        """
        # Check validity and calculate n
        n = 1
        for i in range(K):
            if exps[i] > 0:
                # Quick check to avoid huge intermediate numbers if possible, 
                # but Python ints are arbitrary precision, so we just check at the end or periodically.
                n *= primes[i] ** exps[i]
                if n > LIMIT:
                    return -1, n
        if n < 2:
            return 0, n
            
        # Generate divisors
        divs = [1]
        for i in range(K):
            if exps[i] == 0:
                continue
            p = primes[i]
            e = exps[i]
            new_divs = []
            power = 1
            for _ in range(e):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        
        divs.sort()
        
        score = 0
        # Optimization: gcd(a,b) for adjacent divisors
        # math.gcd is C-implemented and fast
        for j in range(len(divs) - 1):
            if math.gcd(divs[j], divs[j+1]) == 1:
                score += 1
        return score, n

    # Initialize with a strong baseline: Primorial of first 15 primes
    # Exponents: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    initial_exps = [0]*K
    for i in range(15):
        initial_exps[i] = 1
        
    best_score, best_n = calc_score_from_exps(initial_exps)
    current_exps = initial_exps[:]
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0  # Initial temperature
    cooling_rate = 0.9995
    
    # To avoid infinite loops on the same state, we track no_improve
    no_improve_count = 0
    max_no_improve = 1500
    
    while time.time() < deadline:
        # Choose a random prime index to modify
        i = random.randint(0, K - 1)
        
        # Choose a random step
        # Small steps are more likely
        if random.random() < 0.8:
            step = random.choice([-1, 1])
        else:
            step = random.choice([-2, 2, -3, 3])
            
        new_exps = current_exps[:]
        new_exps[i] += step
        
        # Basic validity check: exponents must be non-negative
        if new_exps[i] < 0:
            continue
            
        new_score, new_n = calc_score_from_exps(new_exps)
        
        if new_score < 0:
            # Invalid (n > LIMIT)
            # Reject
            continue
            
        if new_score > current_score:
            # Improvement
            current_score = new_score
            current_exps = new_exps
            if new_score > best_score:
                best_score = new_score
                best_n = new_n
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Worse or equal
            # Accept with probability exp(-delta / T)
            delta = current_score - new_score
            if delta == 0:
                # Equal score, accept to explore
                if random.random() < 0.5:
                    current_exps = new_exps
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_exps = new_exps
            no_improve_count += 1
            
        # Cool down
        T *= cooling_rate
        if T < 0.1:
            T = 0.1
            
        # If stuck for a long time, restart with a random configuration
        if no_improve_count > max_no_improve:
            # Generate a random valid configuration
            # Bias towards small exponents
            random_exps = [0]*K
            for i in range(K):
                # Max exponent for prime i
                max_e = int(math.log(LIMIT, primes[i]))
                # Use a distribution that favors small exponents
                # Geometric-like
                e = 0
                while e < max_e and random.random() < 0.6:
                    e += 1
                random_exps[i] = e
            
            r_score, r_n = calc_score_from_exps(random_exps)
            if r_score > 0:
                current_exps = random_exps
                current_score = r_score
                if r_score > best_score:
                    best_score = r_score
                    best_n = r_n
            no_improve_count = 0
            T = 5.0 # Reheat

    return best_n
