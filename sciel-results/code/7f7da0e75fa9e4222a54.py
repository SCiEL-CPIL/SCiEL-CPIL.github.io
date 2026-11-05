"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # Generate primes up to 1000
    def gen_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = gen_primes(1000)
    MAX_L = 5000  # Look ahead to ensure we can extend the prefix
    
    # Weight function: prioritize smaller numbers to maintain contiguous prefix
    def weight(n):
        if n <= MAX_L:
            return MAX_L - n + 1
        return 0

    uncovered = set(range(1, MAX_L + 1))
    used_primes = set()
    classes = []
    budget = Fraction(1, 1)
    
    # Precompute costs
    prime_costs = {p: Fraction(1, p) for p in primes}
    
    while True:
        best_gain = 0
        best_prime = None
        best_residue = None
        
        for p in primes:
            if p in used_primes:
                continue
            cost = prime_costs[p]
            if budget < cost:
                continue
            
            # Count weighted coverage for each residue
            residue_weights = {}
            for n in uncovered:
                r = n % p
                w = weight(n)
                if w > 0:
                    residue_weights[r] = residue_weights.get(r, 0) + w
            
            if not residue_weights:
                continue
                
            # Find the residue with the maximum weight
            best_r = max(residue_weights, key=residue_weights.get)
            gain = residue_weights[best_r]
            
            if gain > best_gain:
                best_gain = gain
                best_prime = p
                best_residue = best_r
        
        if best_prime is None or best_gain == 0:
            break
        
        p = best_prime
        r = best_residue
        used_primes.add(p)
        classes.append([p, r])
        budget -= prime_costs[p]
        
        # Update uncovered
        uncovered = {n for n in uncovered if n % p != r}
        
        if not uncovered:
            break
            
    # Verify budget
    total = sum(Fraction(1, p) for p, r in classes)
    assert total <= 1, f"Budget exceeded: {total}"
    
    return classes
# EVOLVE-BLOCK-END
