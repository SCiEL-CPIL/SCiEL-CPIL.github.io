import math
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Precompute small primes
    def _sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, p in enumerate(is_prime) if p]

    primes = _sieve(200)
    odd_primes = primes[1:]  # 3, 5, 7, ...

    def _score(n):
        """
        Compute tau_perp(n) by generating sorted divisors and counting adjacent coprime pairs.
        This is the ground truth scorer.
        """
        if n < 2:
            return 0
        
        # Factorize n
        temp = n
        factors = {}
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors[p] = exp
            p += 1 if p == 2 else 2
        if temp > 1:
            factors[temp] = 1
            
        # Generate divisors
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            p_powers = [1]
            for _ in range(e):
                p_powers.append(p_powers[-1] * p)
            for d in divs:
                for pp in p_powers:
                    new_divs.append(d * pp)
            divs = new_divs
            
        divs.sort()
        
        score = 0
        for i in range(1, len(divs)):
            if math.gcd(divs[i-1], divs[i]) == 1:
                score += 1
        return score

    best_n = 2
    best_score = 0

    # Phase 1: Exhaustive search over "canonical" structures: n = 2^a * prod(first k odd primes)
    # This establishes a strong baseline and identifies the region of high scores.
    
    prefix_prod = [1]
    for p in odd_primes:
        prefix_prod.append(prefix_prod[-1] * p)
        
    max_k = 0
    for i, prod in enumerate(prefix_prod):
        if prod <= 10**18:
            max_k = i
        else:
            break

    # We will search k from 0 to max_k, and for each k, scan a.
    # To save time, we can estimate the score or just evaluate.
    # Number of candidates is small (~15 * 60 = 900).
    # Each evaluation takes ~0.1s for large tau(n). Total ~90s.
    # We need to be careful. Let's optimize the scorer or limit the scan.
    
    # Optimized scorer for squarefree part + power of 2?
    # Actually, let's just run it but with a tight deadline check inside.
    
    # We'll iterate k in a way that likely finds high scores early?
    # Scores generally increase with k up to a point, then drop.
    # Let's just iterate all.
    
    for k in range(0, max_k + 1):
        if time.time() > deadline - 10:
            break
            
        m = prefix_prod[k]
        if m > 10**18:
            break
            
        max_a = int(math.log2(10**18 / m))
        
        # For k=0, n=2^a. Score is 1.
        if k == 0:
            if max_a >= 1:
                n_cand = 2
                s = 1
                if s > best_score:
                    best_score = s
                    best_n = n_cand
            continue
            
        # For k >= 1
        # Scan a.
        # We can narrow the range of a?
        # Let's scan all valid a.
        for a in range(1, max_a + 1):
            if time.time() > deadline - 10:
                break
                
            n_cand = (2 ** a) * m
            if n_cand > 10**18:
                break
                
            s = _score(n_cand)
            
            if s > best_score:
                best_score = s
                best_n = n_cand

    # Phase 2: Local Search / Random Perturbation
    # The optimal n might not be the "canonical" one (first k primes).
    # Swapping a prime for a slightly larger one might change the divisor ordering and gcds.
    # Or changing the exponent of 2 might be suboptimal if we change the odd part.
    
    # We will perform random restarts or local moves around the best found structure.
    # Represent n by its prime factorization: {p: e}
    
    # Helper to get factorization of best_n
    def _factorize(n):
        temp = n
        factors = {}
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors[p] = exp
            p += 1 if p == 2 else 2
        if temp > 1:
            factors[temp] = 1
        return factors

    def _rebuild(factors):
        n = 1
        for p, e in factors.items():
            n *= (p ** e)
        return n

    # Start local search from best_n
    current_factors = _factorize(best_n)
    current_score = best_score
    current_n = best_n
    
    # Define a set of candidate primes to swap in/out.
    # We can use primes up to 500.
    candidate_primes = [p for p in primes if p > 2]
    
    # Local Search Loop
    # While time remains, try random modifications.
    # Modification types:
    # 1. Increase/Decrease exponent of a prime.
    # 2. Swap a prime factor for another.
    
    # To make it efficient, we limit the number of steps.
    
    steps = 0
    max_steps = 5000
    
    while time.time() < deadline - 5 and steps < max_steps:
        steps += 1
        if time.time() > deadline - 5:
            break
            
        # Pick a random move
        move_type = random.random()
        
        # Copy current factors
        new_factors = dict(current_factors)
        
        if move_type < 0.5:
            # Change exponent
            if not new_factors:
                continue
            p = random.choice(list(new_factors.keys()))
            e = new_factors[p]
            if e == 1:
                # Remove
                del new_factors[p]
            else:
                new_factors[p] = e - 1
            # Or increase
            if random.random() < 0.5:
                new_factors[p] = new_factors.get(p, 0) + 1
                
        else:
            # Swap prime
            if not new_factors:
                continue
            # Pick a prime to remove
            p_out = random.choice(list(new_factors.keys()))
            e_out = new_factors[p_out]
            if e_out > 1:
                # Just reduce exponent? Or swap one instance?
                # Let's just reduce exponent by 1 and add a new prime with exponent 1
                new_factors[p_out] = e_out - 1
                p_in = random.choice(candidate_primes)
                if p_in == p_out:
                    continue
                new_factors[p_in] = new_factors.get(p_in, 0) + 1
            else:
                # Remove p_out
                del new_factors[p_out]
                p_in = random.choice(candidate_primes)
                new_factors[p_in] = new_factors.get(p_in, 0) + 1

        # Check validity
        if not new_factors:
            continue
            
        n_new = _rebuild(new_factors)
        
        if n_new < 2 or n_new > 10**18:
            continue
            
        # Check if it's the same
        if n_new == current_n:
            continue
            
        s_new = _score(n_new)
        
        if s_new > current_score:
            current_score = s_new
            current_n = n_new
            current_factors = new_factors
            if current_score > best_score:
                best_score = current_score
                best_n = current_n

    return best_n

# EVOLVE-BLOCK-END
