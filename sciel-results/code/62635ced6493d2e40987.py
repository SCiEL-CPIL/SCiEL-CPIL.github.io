"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time

def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0: sieve[0] = False
    if limit >= 1: sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Start with all integers in [0, N]
    S = set(range(N + 1))
    
    # Precompute primes up to N
    primes = _primes(N)
    
    # Iterate through primes in increasing order.
    # For each prime p <= len(S), if S covers all residues mod p,
    # remove the least populated residue class.
    # Once a prime p is satisfied (a residue is missing), it remains satisfied
    # as we only remove elements in subsequent steps.
    
    for p in primes:
        if time.time() > deadline:
            break
            
        current_len = len(S)
        if p > current_len:
            # If p > len(S), it is impossible to cover all p residues
            # with only len(S) elements. Thus, the condition is automatically satisfied.
            # Since primes are increasing, all subsequent primes will also be > len(S).
            break
            
        # Count residues mod p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        
        # Check if any residue is missing
        if min(counts) == 0:
            # Admissibility condition satisfied for this p
            continue
            
        # Find the residue class with the minimum count
        min_count = min(counts)
        target_residue = counts.index(min_count)
        
        # Remove all elements in the target residue class
        # This is O(|S|)
        S = {x for x in S if x % p != target_residue}
        
    return sorted(S)
# EVOLVE-BLOCK-END
