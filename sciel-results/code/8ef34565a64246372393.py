import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    primes = _primes(N)
    
    # Start with all numbers in [0, N]
    S = set(range(N + 1))
    
    # For each prime p in increasing order:
    # - If p > len(S), stop
    # - Check if S covers all p residues mod p
    # - If yes, remove the residue class with the fewest elements
    for p in primes:
        if time.time() > deadline:
            break
        if p > len(S):
            break
        
        # Check which residues are present mod p
        residues = set()
        for v in S:
            residues.add(v % p)
        
        if len(residues) < p:
            # Already missing at least one residue, no action needed
            continue
        
        # All p residues are present. Must remove one class.
        # Count elements in each residue class
        class_counts = {}
        for r in residues:
            class_counts[r] = sum(1 for v in S if v % p == r)
        
        # Remove the class with the minimum count
        min_count = min(class_counts.values())
        # Pick the residue class with minimum count (break ties arbitrarily)
        target_residue = min(class_counts, key=lambda r: class_counts[r])
        
        # Remove all elements in that residue class
        S = {v for v in S if v % p != target_residue}
    
    return list(S)
