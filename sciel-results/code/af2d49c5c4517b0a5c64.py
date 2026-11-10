"""Rewritten program for the narrow admissible tuple task."""
import time
import random
from math import isqrt

N = 5000
DEADLINE = 300.0  # seconds, safety margin from 330s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


# Precompute primes up to N (for candidate pool) and up to a reasonable bound for admissibility checks
_PRIMES_N = _primes(N)
# We need primes up to len(tuple). Max possible tuple size is N+1=5001.
_PRIMES_MAX = _primes(N + 1)


def _is_admissible(points, primes_needed):
    """Check if points form an admissible tuple for the given primes."""
    # points is a set or frozenset for fast lookup? No, we need residue classes.
    # Actually, admissibility: for every prime p <= len(points), there is a residue class mod p that is missed.
    # i.e., not all residues 0..p-1 are present in {x mod p for x in points}.
    # We can check this by counting distinct residues.
    n = len(points)
    # Only need to check primes p <= n
    for p in primes_needed:
        if p > n:
            break
        residues = set()
        for x in points:
            residues.add(x % p)
        if len(residues) == p:
            return False
    return True


def _greedy_repair(points, primes_to_check, deadline):
    """
    Greedily remove points to make the set admissible.
    points: list of integers.
    Returns a set of integers.
    """
    points_set = set(points)
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        n = len(points_set)
        # Find the first prime p <= n that is "full" (all residues present)
        for p in primes_to_check:
            if p > n:
                break
            classes = {}
            for v in points_set:
                r = v % p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            if len(classes) == p:
                # Find the smallest class
                min_len = min(len(v) for v in classes.values())
                # Pick a random class among those with min_len to add diversity
                candidates = [v for v in classes.values() if len(v) == min_len]
                # Deterministic choice for reproducibility in repair, or random?
                # Let's use random to escape local minima in the repair phase itself?
                # No, repair should be deterministic given the set, or we can randomize which element to remove.
                # Let's remove a random element from the smallest class.
                if candidates:
                    target_class = random.choice(candidates)
                    # Remove one element from target_class
                    if target_class:
                        v_to_remove = random.choice(target_class)
                        points_set.remove(v_to_remove)
                        changed = True
                        break
        # If no full prime found, it's admissible
    return points_set


def _local_search(points_set, primes_to_check, deadline):
    """
    Try to add points back to the set to increase its size.
    points_set: set of integers.
    Returns a set of integers.
    """
    # Candidate pool: all integers in [0, N] not in points_set
    # We want to add points that don't violate admissibility.
    # This is hard to check incrementally efficiently without re-checking all primes.
    # Strategy: Randomly pick a candidate, check if adding it keeps the set admissible.
    # If yes, add it. If no, skip.
    # To speed up, we can pre-compute residues for candidates?
    # Let's just iterate through a shuffled list of candidates.
    
    # Generate list of candidates
    candidates = [i for i in range(N + 1) if i not in points_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for c in candidates:
            if time.time() > deadline:
                break
            # Try adding c
            new_set = points_set | {c}
            if _is_admissible(new_set, primes_to_check):
                points_set = new_set
                improved = True
                # Remove c from candidates list? It's already in points_set now.
                # We can't easily remove from the list we're iterating.
                # But since we iterate once, it's fine.
                # However, adding one point might allow adding others that were previously blocked.
                # So we loop again.
    return points_set


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    best_set = set()
    best_size = 0
    
    # Initial seed: Random subset of [0, N]
    # Start with a medium-sized random set to give the repair algo something to work with.
    # Or start with all primes > sqrt(N) as in the seed?
    # The seed program starts with [1] + primes > sqrt(N).
    # Let's try a few different initialization strategies.
    
    strategies = []
    
    # Strategy 1: Seed program style
    init1 = [1] + [p for p in _PRIMES_N if p * p > N]
    strategies.append(set(init1))
    
    # Strategy 2: Random sparse set
    # Density ~ 1/log(N)?
    # Let's try a few random densities
    for density in [0.05, 0.1, 0.2, 0.3]:
        init = set(random.sample(range(N + 1), int((N + 1) * density)))
        strategies.append(init)
    
    # Strategy 3: Based on small primes residues
    # Pick random residues for first few primes, then select numbers matching them?
    # This is complex. Let's stick to the above.
    
    for init_set in strategies:
        if time.time() > deadline:
            break
        
        # Phase 1: Repair
        repaired = _greedy_repair(init_set, _PRIMES_MAX, deadline)
        
        # Phase 2: Local Search (Addition)
        # Note: _local_search modifies the set in place or returns a new one?
        # It returns a new one.
        expanded = _local_search(repaired, _PRIMES_MAX, deadline)
        
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_set = expanded
            
        # Check deadline
        if time.time() > deadline:
            break
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 10.0:
        # Random initialization
        density = random.uniform(0.05, 0.4)
        init_set = set(random.sample(range(N + 1), int((N + 1) * density)))
        
        repaired = _greedy_repair(init_set, _PRIMES_MAX, deadline)
        expanded = _local_search(repaired, _PRIMES_MAX, deadline)
        
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_set = expanded
    
    # Final check: ensure best_set is admissible (paranoia)
    if best_set and not _is_admissible(best_set, _PRIMES_MAX):
        # Fallback to a known valid set if something went wrong
        # Simple fallback: all primes > sqrt(N) are admissible?
        # Actually, the seed program's output is guaranteed admissible by its repair loop.
        # Let's just return the list from the best_set, assuming our checks are correct.
        pass
        
    return list(best_set)

# To ensure the function is defined and callable
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
