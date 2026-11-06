import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    
    # State
    used_primes = set()
    classes = []
    budget = Fraction(1)
    
    # Coverage array, size 100000 (capped by task)
    LIMIT = 100000
    covered = bytearray(LIMIT + 1) # 0 is uncovered, 1 is covered
    
    L = 0
    
    # Helper to check if a move is valid and calculate new L
    def simulate_move(p, r, n):
        # n is the first number covered by this move (L+1)
        # We check from n upwards
        # A number k is covered if covered[k] == 1 OR (k % p == r)
        # We stop at the first k where covered[k] == 0 AND k % p != r
        
        # Optimization: We know [1, L] are covered.
        # So we start checking at n.
        
        k = n
        # We can jump? No, we need to check every number.
        # But we can skip covered blocks?
        # For now, simple loop.
        
        while k < LIMIT:
            if covered[k] == 0:
                if k % p != r:
                    return k - 1
            k += 1
        return LIMIT - 1

    while time.time() < deadline:
        n = L + 1
        if n > LIMIT:
            break
            
        best_L = L
        best_p = None
        best_r = None
        
        # Try all unused primes
        # Sort primes? Maybe try smaller primes first as they are denser?
        # Or just iterate.
        for p in primes:
            if p in used_primes:
                continue
            
            cost = Fraction(1, p)
            if budget < cost:
                continue
            
            r = n % p
            
            # Quick check: does this move even cover n? Yes, by definition.
            # Does it extend L?
            new_L = simulate_move(p, r, n)
            
            if new_L > best_L:
                best_L = new_L
                best_p = p
                best_r = r
                # If we reached LIMIT, we can break early?
                # But maybe a different p allows further extension later?
                # No, if we hit LIMIT, we are done.
                if best_L == LIMIT - 1:
                    break
            elif new_L == best_L:
                # Tie: pick larger p to save budget
                if best_p is not None and p > best_p:
                    best_p = p
                    best_r = r
        
        if best_p is None or best_L == L:
            # Stuck
            break
        
        # Commit move
        used_primes.add(best_p)
        classes.append([best_p, best_r])
        budget -= Fraction(1, best_p)
        
        # Update covered array
        # Mark best_r, best_r+best_p, ... up to LIMIT
        # Start from n? No, the class covers ALL numbers congruent to r mod p.
        # We need to update the whole array to be accurate for future simulations.
        # But we only care about [1, LIMIT].
        # First number >= 1 congruent to r mod p is r if r>=1 else p.
        # Actually, r is in [0, p-1].
        # If r == 0, first is p. If r > 0, first is r.
        # But we know n is covered. n = L+1.
        # We can just update from 1 to LIMIT.
        # This is O(LIMIT/p).
        
        start = best_r
        if start == 0:
            start = best_p
        # Wait, if best_r is 0, it covers p, 2p...
        # If best_r > 0, it covers best_r, best_r+p...
        
        for j in range(start, LIMIT + 1, best_p):
            covered[j] = 1
            
        L = best_L
        
    return classes

# EVOLVE-BLOCK-START
# The code above is the implementation.
# I will wrap it in the required format.
# EVOLVE-BLOCK-END
