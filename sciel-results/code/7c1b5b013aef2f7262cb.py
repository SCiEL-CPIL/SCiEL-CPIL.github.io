"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 300.0

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _greedy_reduce(points, primes, rng, deadline):
    """
    Greedily reduce points until admissible.
    """
    points = list(points)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        n = len(points)
        # Find all problematic primes and their residue counts
        bad = []
        for p in primes:
            if p > n:
                break
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            if all(c > 0 for c in counts):
                min_c = min(counts)
                bad.append((p, counts, min_c))
        
        if not bad:
            break
        
        # Choose which prime to fix: prefer the one with smallest min count
        if rng is not None and len(bad) > 1:
            # Randomly pick among the worst few
            bad.sort(key=lambda x: x[2])
            k = min(3, len(bad))
            p, counts, _ = rng.choice(bad[:k])
        else:
            p, counts, _ = min(bad, key=lambda x: x[2])
        
        # Find residue classes with minimum count
        min_c = min(counts)
        min_classes = [r for r, c in enumerate(counts) if c == min_c]
        chosen = rng.choice(min_classes) if rng is not None else min_classes[0]
        
        # Remove all points in that residue class
        points = [v for v in points if v % p != chosen]
        changed = True
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - 10.0  # 10s safety margin
    
    primes = _primes(N + 1)
    
    best = []
    best_size = 0
    
    # Multiple initial sets to try
    initial_sets = []
    
    # Set 1: Original - primes > sqrt(N) + 1
    sqrt_N = int(N ** 0.5)
    initial_sets.append([1] + [p for p in primes if p * p > N])
    
    # Set 2: All integers
    initial_sets.append(list(range(N + 1)))
    
    # Set 3: Numbers not divisible by very small primes
    small_primes = [2, 3, 5, 7]
    initial_sets.append([v for v in range(N + 1) if all(v % p != 0 for p in small_primes)])
    
    # Set 4: Random subset
    rng_init = random.Random(42)
    initial_sets.append(rng_init.sample(range(N + 1), 1500))
    
    for init_idx, init_set in enumerate(initial_sets):
        if time.time() > deadline:
            break
        
        # Deterministic pass
        result = _greedy_reduce(init_set, primes, None, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
        
        # Randomized passes
        num_trials = 30 if len(init_set) > 1000 else 100
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            rng = random.Random(init_idx * 1000 + trial)
            result = _greedy_reduce(init_set, primes, rng, deadline)
            if len(result) > best_size:
                best = result
                best_size = len(result)
    
    # Fallback
    if len(best) < 10:
        points = [1] + [p for p in primes if p * p > N]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            n = len(points)
            for p in primes:
                if p > n:
                    break
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points = [v for v in points if v not in smallest]
                    changed = True
                    break
        best = points
    
    return best
# EVOLVE-BLOCK-END

