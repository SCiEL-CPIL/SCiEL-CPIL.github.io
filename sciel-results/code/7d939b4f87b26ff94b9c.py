"""Rewritten program: maximise admissible tuple size in [0, N].

Strategy:
- Start from the full range [0, N].
- For each prime p <= len(tuple), if all p residue classes are present,
  remove the least-populated class (with a small random tie-break to escape
  local minima).
- Use a time budget to allow multiple restarts with different random seeds,
  keeping the best (largest) admissible tuple found.
"""
import time
import random

N = 5000
DEADLINE = 280.0  # seconds, leaving margin before 330s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_needed):
    """Check if points form an admissible tuple for the given primes."""
    for p in primes_needed:
        if len(points) < p:
            continue  # can't cover all p classes if fewer than p points
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) == p:
                break
        if len(residues) == p:
            return False
    return True


def _construct_once(seed):
    """Single attempt: start from [0..N], iteratively remove least-populated
    residue class for each prime that is fully covered."""
    rng = random.Random(seed)
    points = list(range(N + 1))
    
    # Primes up to initial length
    primes_initial = _primes(len(points))
    
    # We'll iterate: for each prime, if all classes present, remove least-populated
    # Repeat until stable or time runs out
    max_iter = 200
    for _ in range(max_iter):
        if time.time() > DEADLINE:
            break
        changed = False
        # Primes needed are those <= len(points)
        current_len = len(points)
        primes_needed = _primes(current_len)
        
        for p in primes_needed:
            if len(points) < p:
                continue
            # Count residue classes
            class_counts = {}
            for v in points:
                r = v % p
                class_counts[r] = class_counts.get(r, 0) + 1
            
            if len(class_counts) == p:
                # All classes present; remove the least-populated one
                # Tie-break randomly to escape local minima
                min_count = min(class_counts.values())
                candidates = [r for r, c in class_counts.items() if c == min_count]
                chosen = rng.choice(candidates)
                points = [v for v in points if v % p != chosen]
                changed = True
                break
        
        if not changed:
            break
    
    return points


def construct():
    """Main entry point: try multiple random restarts, keep best admissible tuple."""
    start_time = time.time()
    
    # First, get a baseline from the seed-like approach
    best = _construct_once(0)
    
    # Validate baseline
    if not _is_admissible(best, _primes(len(best))):
        # Fallback: use the simple seed approach
        points = [1] + [p for p in _primes(N) if p * p > N]
        changed = True
        while changed:
            changed = False
            for p in _primes(len(points)):
                classes = {}
                for v in points:
                    classes.setdefault(v % p, []).append(v)
                if len(classes) == p:
                    smallest = min(classes.values(), key=len)
                    points = [v for v in points if v not in smallest]
                    changed = True
                    break
        best = points
    
    best_len = len(best)
    
    # Try multiple restarts with different seeds
    seed = 1
    while time.time() < DEADLINE:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 5:  # safety margin
            break
        
        candidate = _construct_once(seed)
        if len(candidate) > best_len:
            if _is_admissible(candidate, _primes(len(candidate))):
                best = candidate
                best_len = len(candidate)
        
        seed += 1
    
    # Final validation
    if not _is_admissible(best, _primes(len(best))):
        # Emergency fallback
        points = [1] + [p for p in _primes(N) if p * p > N]
        changed = True
        while changed:
            changed = False
            for p in _primes(len(points)):
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
