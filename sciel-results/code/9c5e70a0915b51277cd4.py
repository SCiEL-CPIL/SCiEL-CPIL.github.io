import time
import random
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def construct():
    deadline = time.time() + 110  # Safety margin
    all_primes = _primes_upto(1000)
    
    # Select primes 13 to 1000
    # This matches the budget constraint sum(1/p) <= 1
    # Sum of reciprocals of primes < 13 is approx 1.26
    # Sum of reciprocals of primes <= 1000 is approx 2.2
    # So primes >= 13 sum to approx 0.94 <= 1
    selected_primes = [p for p in all_primes if p >= 13]
    
    # Verify budget
    cost = sum(Fraction(1, p) for p in selected_primes)
    if cost > Fraction(1, 1):
        # Fallback: remove largest primes until fits
        # But 0.94 is safe.
        pass
        
    N = len(selected_primes)
    # Max L to track. Primes up to 1000. 
    # With 160 primes, we can definitely cover 1000.
    # Let's track up to 5000 to be safe for extensions.
    MAX_L = 5000
    
    # Initialize residues using diagonal heuristic
    # i-th prime covers i+1
    residues = [0] * N
    for i, p in enumerate(selected_primes):
        residues[i] = (i + 1) % p
        
    # Initialize coverage counts
    # count[n] = number of primes covering n
    counts = [0] * (MAX_L + 2)
    
    def update_counts(p_idx, old_r, new_r):
        p = selected_primes[p_idx]
        # Decrement old
        n = old_r
        if n == 0:
            n = p
        while n <= MAX_L:
            counts[n] -= 1
            n += p
        # Increment new
        n = new_r
        if n == 0:
            n = p
        while n <= MAX_L:
            counts[n] += 1
            n += p

    # Apply initial residues
    for i in range(N):
        p = selected_primes[i]
        r = residues[i]
        n = r
        if n == 0:
            n = p
        while n <= MAX_L:
            counts[n] += 1
            n += p
            
    # Calculate initial L
    L = 0
    while L < MAX_L and counts[L+1] > 0:
        L += 1
        
    best_L = L
    best_residues = residues[:]
    
    # Local Search
    # We try to extend L by 1 at a time.
    # For each prime, check if we can change its residue to cover L+1
    # without breaking any n <= L.
    
    while time.time() < deadline:
        improved = False
        # Shuffle primes to avoid bias?
        # Or just iterate.
        for i in range(N):
            if time.time() > deadline:
                break
            p = selected_primes[i]
            r_old = residues[i]
            r_new = (L + 1) % p
            
            if r_old == r_new:
                continue
                
            # Check if changing r_old to r_new breaks any n <= L
            # A number n <= L is broken if:
            # 1. n is covered by p (n % p == r_old)
            # 2. count[n] == 1 (p is the only one covering it)
            
            safe = True
            n = r_old
            if n == 0:
                n = p
            
            # Only need to check n <= L
            while n <= L:
                if counts[n] == 1:
                    safe = False
                    break
                n += p
                
            if safe:
                # Apply change
                update_counts(i, r_old, r_new)
                residues[i] = r_new
                
                # Extend L
                while L < MAX_L and counts[L+1] > 0:
                    L += 1
                    
                if L > best_L:
                    best_L = L
                    best_residues = residues[:]
                    
                improved = True
                break # Restart loop with new L
        
        if not improved:
            # Stuck. Try a random perturbation to escape local minimum.
            # Pick a random prime and change its residue randomly.
            # Then try to recover.
            # To avoid infinite loops of bad moves, we revert if L doesn't recover quickly?
            # Or just accept the drop and try to extend again?
            # Since we want max L, a drop is bad unless it leads to a higher peak.
            # Given the strong diagonal start, we might not need this.
            # But let's add a simple random restart of the local search phase.
            # Actually, if we are stuck, it means for every prime p, 
            # changing it to cover L+1 breaks something <= L.
            # This is a common "trap".
            
            # Try swapping residues of two primes?
            # Or just break if time is low.
            if time.time() > deadline - 20:
                break
            
            # Random perturbation
            i1 = random.randint(0, N-1)
            p1 = selected_primes[i1]
            old_r1 = residues[i1]
            new_r1 = random.randint(0, p1-1)
            
            # Just try it and see if we can recover
            update_counts(i1, old_r1, new_r1)
            residues[i1] = new_r1
            
            # Recalculate L (might have dropped)
            # Find first gap
            temp_L = 0
            while temp_L < MAX_L and counts[temp_L+1] > 0:
                temp_L += 1
            L = temp_L
            
            # If L dropped significantly, revert?
            # Let's just continue the loop. The greedy extension might fix it.
            # If not, we might oscillate.
            # To be safe, if L < best_L - 100, revert?
            # For now, keep it simple. The diagonal start is very robust.
            pass

    # Final verification
    final_cost = sum(Fraction(1, p) for p in selected_primes)
    if final_cost > Fraction(1, 1):
        # Should not happen
        return [[p, (i + 1) % p] for i, p in enumerate(selected_primes)]
        
    return [[selected_primes[i], best_residues[i]] for i in range(N)]
