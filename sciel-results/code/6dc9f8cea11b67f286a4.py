"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    primes = _primes(N)
    num_primes = len(primes)
    
    # Initialize set S as a boolean array for fast membership/removal
    # And a list of active elements
    active = [True] * (N + 1)
    S = list(range(N + 1))
    
    # Initialize counts for each prime
    # counts[i] is an array of size primes[i]
    # counts[i][r] = number of x in S such that x % primes[i] == r
    counts = []
    for p in primes:
        c = [0] * p
        # Efficiently compute initial counts
        # x ranges from 0 to N
        # Number of x in [0, N] with x % p == r
        # This is roughly (N+1)//p
        for r in range(p):
            # First element is r (if r <= N), then r+p, ...
            # Count = floor((N - r) / p) + 1 if r <= N else 0
            if r <= N:
                c[r] = (N - r) // p + 1
            else:
                c[r] = 0
        counts.append(c)
        
    satisfied = [False] * num_primes
    
    # We need to find the "cost" to fix a prime p.
    # Cost is min(counts[i]).
    # We iterate until no prime p <= |S| is violating (min > 0).
    
    while True:
        M = sum(active) # Current size of S. O(N) each time? 
        # To optimize, maintain current_size
        # Let's maintain current_size outside
        
        # Find violating primes
        min_cost = float('inf')
        best_prime_idx = -1
        best_residue = -1
        
        # Only check primes p <= M
        # We can break early if primes[i] > M
        for i, p in enumerate(primes):
            if p > M:
                break
            if satisfied[i]:
                continue
            
            min_c = min(counts[i])
            if min_c == 0:
                # This prime is already satisfied
                satisfied[i] = True
            elif min_c < min_cost:
                min_cost = min_c
                best_prime_idx = i
                # Find the residue with min count
                # We need to find r such that counts[i][r] == min_c
                # Since we picked min_c, we can just find the first one
                for r in range(p):
                    if counts[i][r] == min_c:
                        best_residue = r
                        break
        
        if best_prime_idx == -1:
            # No violating primes found
            break
            
        # Remove all x in S such that x % primes[best_prime_idx] == best_residue
        p = primes[best_prime_idx]
        r = best_residue
        
        # Iterate through S to find elements to remove
        # We can iterate through the range 0..N
        for x in range(N + 1):
            if active[x] and x % p == r:
                active[x] = False
                M -= 1
                # Update counts for all primes
                for j in range(num_primes):
                    counts[j][x % primes[j]] -= 1
                    # Optimization: if counts[j] becomes all 0? No.
                    # Check if this prime j is now satisfied?
                    # We can't easily check min(counts[j]) here without O(p) cost.
                    # But we only need to know if min becomes 0.
                    # We can just let the main loop detect it.
        
        satisfied[best_prime_idx] = True
        
    # Return the list of active elements
    return [x for x in range(N + 1) if active[x]]
# EVOLVE-BLOCK-END
