"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to a safe bound (we need primes up to ~N/2 at most for tuple length)
# Max tuple size is N+1 = 5001, so we need primes up to 5001
_PRIMES_UP_TO_5001 = _primes(5001)
# Also primes up to N for initial set construction
_PRIMES_UP_TO_N = _primes(N)

def _is_admissible(points, primes_needed):
    """Check if points form an admissible tuple for given primes."""
    for p in primes_needed:
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                return False
    return True

def _count_covered_classes(points, p):
    """Count how many residue classes mod p are covered by points."""
    residues = set()
    for v in points:
        residues.add(v % p)
    return len(residues)

def _greedy_repair(points, primes_to_check):
    """
    Greedy repair: for each prime p in primes_to_check, if all p classes are covered,
    remove the least-populated class. Repeat until admissible.
    Returns the repaired list of points.
    """
    points = list(points)
    changed = True
    while changed:
        changed = False
        for p in primes_to_check:
            if p > len(points):
                continue
            classes = defaultdict(list)
            for v in points:
                classes[v % p].append(v)
            if len(classes) == p:
                # Find the smallest class to remove
                smallest = min(classes.values(), key=len)
                smallest_set = set(smallest)
                points = [v for v in points if v not in smallest_set]
                changed = True
                break
    return points

def _random_greedy(points, primes_to_check, rng, max_iters=200):
    """
    Randomized greedy repair with random prime ordering and random tie-breaking.
    """
    points = list(points)
    for _ in range(max_iters):
        # Shuffle primes to check
        shuffled_primes = list(primes_to_check)
        rng.shuffle(shuffled_primes)
        changed = False
        for p in shuffled_primes:
            if p > len(points):
                continue
            classes = defaultdict(list)
            for v in points:
                classes[v % p].append(v)
            if len(classes) == p:
                # Find smallest classes; if tie, pick randomly
                min_size = min(len(c) for c in classes.values())
                smallest_classes = [c for c in classes.values() if len(c) == min_size]
                chosen = rng.choice(smallest_classes)
                chosen_set = set(chosen)
                points = [v for v in points if v not in chosen_set]
                changed = True
                break
        if not changed:
            break
    return points

def _initial_points(rng, strategy=0):
    """
    Generate initial set of points.
    Strategy 0: {1} + primes p where p*p > N (the seed approach)
    Strategy 1: Random subset of [0, N]
    Strategy 2: {1} + random selection of large primes
    """
    if strategy == 0:
        return [1] + [p for p in _PRIMES_UP_TO_N if p * p > N]
    elif strategy == 1:
        # Random sample
        k = rng.randint(50, 200)
        return rng.sample(range(N + 1), k)
    elif strategy == 2:
        large_primes = [p for p in _PRIMES_UP_TO_N if p > 100]
        k = rng.randint(30, 150)
        return [1] + rng.sample(large_primes, min(k, len(large_primes)))
    else:
        return [1] + [p for p in _PRIMES_UP_TO_N if p * p > N]

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s kill
    
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    best_points = None
    best_size = 0
    
    # First, run the deterministic seed approach as baseline
    primes_for_check = _PRIMES_UP_TO_5001
    initial = _initial_points(rng, strategy=0)
    repaired = _greedy_repair(initial, primes_for_check)
    if len(repaired) > best_size:
        best_points = repaired
        best_size = len(repaired)
    
    # Now do randomized restarts with different strategies
    strategies = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    restart_count = 0
    
    while time.time() < deadline:
        strategy = strategies[restart_count % len(strategies)]
        initial = _initial_points(rng, strategy=strategy)
        
        # Try randomized greedy with random prime ordering
        repaired = _random_greedy(initial, primes_for_check, rng, max_iters=300)
        
        if len(repaired) > best_size:
            best_points = repaired
            best_size = len(repaired)
        
        restart_count += 1
        
        # If we've been stuck for a while, try more aggressive randomization
        if restart_count % 10 == 0:
            # Try a completely random set
            k = rng.randint(100, 400)
            initial = rng.sample(range(N + 1), k)
            repaired = _random_greedy(initial, primes_for_check, rng, max_iters=500)
            if len(repaired) > best_size:
                best_points = repaired
                best_size = len(repaired)
    
    # Final check: ensure the best result is valid
    if best_points is None:
        # Fallback to seed
        initial = _initial_points(rng, strategy=0)
        best_points = _greedy_repair(initial, primes_for_check)
    
    return best_points
# EVOLVE-BLOCK-END
