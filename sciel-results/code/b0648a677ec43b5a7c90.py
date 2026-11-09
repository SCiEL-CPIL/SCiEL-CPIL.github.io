"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 5000
DEADLINE = 300.0

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes):
    """Check if points form an admissible tuple for the given primes."""
    for p in primes:
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True


def construct():
    start_time = time.time()
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    # Strategy: Use a greedy repair approach with multiple starting configurations
    # and try to maximize the tuple size.
    
    best_points = []
    
    # Configuration 1: Start with 1 and primes > sqrt(N)
    points = [1] + [p for p in all_primes if p * p > N]
    
    # Repair loop
    changed = True
    while changed:
        if time.time() - start_time > DEADLINE - 10:
            break
        changed = False
        for p in _primes(len(points)):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                # Find the least populated class
                smallest = min(classes.values(), key=len)
                smallest_set = set(smallest)
                points = [v for v in points if v not in smallest_set]
                changed = True
                break
    
    if _is_admissible(points, _primes(len(points))):
        best_points = points
    
    # Try to improve by adding points from [0, N]
    # We can try to add points that don't violate admissibility
    current_size = len(best_points)
    current_set = set(best_points)
    
    # Candidate points: 0 and primes <= sqrt(N) that might help
    candidates = [0] + [p for p in all_primes if p * p <= N]
    
    # Try adding candidates one by one, checking if admissibility is maintained
    for cand in candidates:
        if time.time() - start_time > DEADLINE - 5:
            break
        if cand in current_set:
            continue
        new_points = best_points + [cand]
        # Check if still admissible
        if _is_admissible(new_points, _primes(len(new_points))):
            best_points = new_points
            current_set.add(cand)
    
    # Try to add more points from the full range [0, N] using a greedy approach
    # This is expensive, so we limit it
    if time.time() - start_time < DEADLINE - 30:
        # Try adding random points
        np.random.seed(42)
        attempts = 0
        while time.time() - start_time < DEADLINE - 10 and attempts < 1000:
            cand = np.random.randint(0, N + 1)
            if cand in current_set:
                attempts += 1
                continue
            new_points = best_points + [cand]
            if _is_admissible(new_points, _primes(len(new_points))):
                best_points = new_points
                current_set.add(cand)
            attempts += 1
    
    # Final validation
    if not _is_admissible(best_points, _primes(len(best_points))):
        # Fallback to a known valid small set
        best_points = [0, 1, 2, 3, 5, 7, 11, 13]
        # Ensure it's valid
        while not _is_admissible(best_points, _primes(len(best_points))):
            if len(best_points) > 1:
                best_points.pop()
    
    return best_points
# EVOLVE-BLOCK-END
