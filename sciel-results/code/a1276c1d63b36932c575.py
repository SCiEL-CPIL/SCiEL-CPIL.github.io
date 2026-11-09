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
    # Use a deadline that leaves margin for the evaluator's overhead
    deadline = start_time + 290.0
    
    # Initialize with the full set
    points = set(range(N + 1))
    primes = _primes(N)
    
    while True:
        # Check time
        if time.time() > deadline:
            break
            
        k = len(points)
        if k == 0:
            break
            
        # Find the smallest prime p <= k that is "bad" (covers all residues)
        bad_prime = None
        for p in primes:
            if p > k:
                break
            # Check coverage
            # Optimization: if k < p, it's impossible to cover all p residues, 
            # but we already checked p <= k.
            # To check if all residues are present, we can use a set of mods.
            # For small p, this is fast.
            residues = {x % p for x in points}
            if len(residues) == p:
                bad_prime = p
                break
        
        if bad_prime is None:
            break
            
        p = bad_prime
        
        # Find the residue class with the minimum count
        counts = [0] * p
        for x in points:
            counts[x % p] += 1
            
        min_residue = min(range(p), key=lambda r: counts[r])
        
        # Remove all elements in this residue class
        # Using a set comprehension is efficient for moderate sizes
        points = {x for x in points if x % p != min_residue}
        
    return list(points)
