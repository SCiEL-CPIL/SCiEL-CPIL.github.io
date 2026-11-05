"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000
import time

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_list):
    """Check if points form an admissible tuple: for every prime p <= len(points),
    some residue class mod p is missed."""
    n = len(points)
    for p in primes_list:
        if p > n:
            break
        residues = set(v % p for v in points)
        if len(residues) == p:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin before 300s limit
    
    # Start with full set [0, N]
    points = list(range(N + 1))
    
    # Get primes up to N+1 (since len(points) can be up to N+1)
    primes = _primes(N + 1)
    
    # Greedily remove least-populated residue classes for violating primes
    # Process in order of increasing prime (smaller primes have fewer classes)
    while True:
        if time.time() > deadline:
            break
        
        changed = False
        for p in primes:
            if p > len(points):
                break
            
            # Count residue classes
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            # Check if all classes are covered
            if min(counts) > 0:
                # All p classes are present; remove the least-populated class
                min_count = min(counts)
                # Find which class(es) have this minimum count
                min_class = counts.index(min_count)
                
                # Remove all points in this residue class
                points = [v for v in points if v % p != min_class]
                changed = True
                break
        
        if not changed:
            break
    
    # Final admissibility check
    if not _is_admissible(points, primes):
        # Fallback: use the seed approach if something went wrong
        pts2 = [1] + [p for p in _primes(N) if p * p > N]
        changed2 = True
        while changed2:
            if time.time() > deadline:
                break
            changed2 = False
            for p in _primes(len(pts2)):
                classes = {}
                for v in pts2:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    pts2 = [v for v in pts2 if v not in smallest]
                    changed2 = True
                    break
        return pts2
    
    return points
# EVOLVE-BLOCK-END
