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
    deadline = start_time + 290  # 10s margin
    
    primes = _primes(N + 1)
    
    # Start with all integers in [0, N]
    points = list(range(N + 1))
    
    for p in primes:
        if time.time() > deadline:
            break
            
        if p > len(points):
            break
            
        # Count residues
        counts = [0] * p
        for x in points:
            counts[x % p] += 1
            
        # Check if admissible for this p
        if 0 in counts:
            continue
            
        # Find min count residue
        min_count = min(counts)
        # If multiple have min count, pick the first one? Or random?
        # Picking the first is deterministic.
        # To be slightly smarter, we could pick the one that is "least useful" for future primes?
        # But min count is the standard greedy choice.
        
        target_residue = counts.index(min_count)
        
        # Remove elements with this residue
        points = [x for x in points if x % p != target_residue]
        
    return points
