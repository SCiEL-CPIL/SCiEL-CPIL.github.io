"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 120  # safety margin before 150s kill
    
    # Sieve for primes up to 511
    is_prime = [True] * 512
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(512**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 512, i):
                is_prime[j] = False
    
    primes = [p for p in range(2, 512) if is_prime[p]]
    
    # Start with all primes (they're pairwise coprime)
    chosen = list(primes)
    
    # Precompute score contribution for each candidate
    def score_of(a):
        return 1.0 / (512 - a)
    
    # Consider composites from 511 down to 1
    # A composite a can be added if gcd(a, b) == 1 for all b in chosen
    # But adding a composite blocks its prime factors' contributions
    # Since primes are already included, adding a composite that shares a prime factor
    # with any chosen prime is impossible (gcd > 1)
    # So composites can only be added if they're coprime to ALL chosen primes
    # That means the composite must have no prime factor in [2, 511] that's in our prime set
    # But ALL primes < 512 are in our set, so NO composite can be added!
    
    # Wait - that's the key insight. Since we include ALL primes < 512,
    # every composite < 512 has at least one prime factor < 512, which is in our set.
    # So gcd(composite, that_prime) > 1, meaning we CANNOT add any composite.
    
    # Therefore the optimal solution is simply all primes < 512!
    # Let me verify: the score is sum of 1/(512-p) for all primes p < 512.
    
    # But wait - could we do better by EXCLUDING some primes to include composites?
    # For example, if we exclude prime p, we might include a composite c that's coprime
    # to all other chosen numbers. But c < 512 and c is composite, so c has prime factors.
    # If we exclude all prime factors of c, we could include c.
    # But c < 512, and the prime factors of c are smaller than c, so 1/(512-c) < 1/(512-p)
    # for each prime factor p of c (since c > p means 512-c < 512-p).
    # Actually, c > p means 512-c < 512-p, so 1/(512-c) > 1/(512-p).
    # Hmm, so a composite could give higher value than its prime factors!
    
    # Example: c = 510 = 2*3*5*17. 1/(512-510) = 1/2 = 0.5
    # Prime factors: 2,3,5,17. 1/(512-2)+1/(512-3)+1/(512-5)+1/(512-17) = 1/510+1/509+1/507+1/495 ≈ 0.00196+0.00196+0.00197+0.00202 ≈ 0.00791
    # So 0.5 >> 0.00791! We should include 510 and exclude 2,3,5,17.
    
    # This changes everything. Let me do a proper search.
    
    # Strategy: Greedy from high a to low a, but with backtracking or local search.
    # For now, let's do a greedy that considers both primes and composites.
    
    # Reset
    chosen = []
    
    # Candidates sorted by a descending (highest value first)
    candidates = list(range(511, 0, -1))
    
    for a in candidates:
        if time.time() > deadline - 5:
            break
        # Check if a is coprime to all chosen
        compatible = True
        for b in chosen:
            if gcd(a, b) != 1:
                compatible = False
                break
        if compatible:
            chosen.append(a)
    
    return chosen
# EVOLVE-BLOCK-END
