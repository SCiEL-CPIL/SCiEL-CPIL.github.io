"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(points, primes_to_check):
    """Check if points is admissible with respect to primes_to_check."""
    n = len(points)
    for p in primes_to_check:
        if p > n:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
            if len(residues) < p:
                # Still need to check all, but if we have all p residues, we fail
                pass
        if len(residues) == p:
            return False
    return True


def _check_all_primes(points):
    """Check admissibility for all primes <= len(points)."""
    n = len(points)
    primes_needed = _primes(n)
    return _is_admissible(points, primes_needed)


def _greedy_repair(points):
    """Repair points by removing elements to make it admissible."""
    points = list(points)
    changed = True
    while changed:
        changed = False
        n = len(points)
        for p in _primes(n):
            classes = {}
            for v in points:
                classes.setdefault(v % p, []).append(v)
            if len(classes) == p:
                # Find the smallest class to remove
                smallest = min(classes.values(), key=len)
                # Remove all elements in the smallest class
                to_remove = set(smallest)
                points = [v for v in points if v not in to_remove]
                changed = True
                break
    return points


def _local_search(points, deadline, max_swaps=500000):
    """Local search to improve the size of an admissible set."""
    points = list(points)
    point_set = set(points)
    best_size = len(points)
    
    # Precompute primes up to N+10 for efficiency
    max_possible = N + 20
    all_primes = _primes(max_possible)
    
    # Candidate points: all integers in [0, N] not in points
    candidates = [i for i in range(N + 1) if i not in point_set]
    random.shuffle(candidates)
    
    iterations = 0
    while time.time() < deadline and iterations < max_swaps:
        iterations += 1
        
        if not candidates:
            break
        
        # Try to add a random candidate
        idx = random.randint(0, len(candidates) - 1)
        x = candidates[idx]
        
        # Try direct add
        new_points = list(points) + [x]
        new_size = len(new_points)
        
        # Check admissibility
        is_adm = True
        for p in all_primes:
            if p > new_size:
                break
            residues = set(v % p for v in new_points)
            if len(residues) == p:
                is_adm = False
                break
        
        if is_adm:
            points = new_points
            point_set.add(x)
            candidates.pop(idx)
            best_size = len(points)
        else:
            # Try swap: remove a random y from points, add x
            if points:
                y_idx = random.randint(0, len(points) - 1)
                y = points[y_idx]
                
                new_points = [v for i, v in enumerate(points) if i != y_idx] + [x]
                new_points.sort()  # Keep sorted for consistency
                
                # Check admissibility
                is_adm = True
                for p in all_primes:
                    if p > len(new_points):
                        break
                    residues = set(v % p for v in new_points)
                    if len(residues) == p:
                        is_adm = False
                        break
                
                if is_adm:
                    points = new_points
                    point_set.discard(y)
                    point_set.add(x)
                    candidates.pop(idx)
                    candidates.append(y)
                    best_size = len(points)
                # else: swap failed, do nothing
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 280  # 20s safety margin
    
    # Phase 1: Greedy construction
    # Start with a good initial set
    # Use numbers that are 1 mod small primes to avoid covering all residue classes
    
    # Strategy: pick numbers that are congruent to 1 mod p for small primes
    # This way, for each prime p, residue 1 is always present, but we need to miss some other residue
    # Actually, to be admissible, we need to MISS some residue class for each prime p <= len(tuple)
    
    # Better: start with a set and repair
    
    # Initial guess: all numbers in [0, N] that are not divisible by 2
    # This misses residue 0 mod 2, so it's admissible for p=2
    # But for p=3, we might cover all residues
    
    # Let's start with a smaller set and grow it
    
    # Start with just {1}
    points = [1]
    
    # Try to add numbers greedily
    # For efficiency, maintain residue counts
    current_n = len(points)
    
    # We'll try to build up by adding numbers from 0 to N
    # but this is slow. Instead, let's use a smarter approach.
    
    # Phase 1: Repair a large initial set
    # Start with all odd numbers in [0, N]
    # This is admissible for p=2 (misses 0)
    # Check for other primes and repair
    
    initial_points = [i for i in range(N + 1) if i % 2 == 1]
    points = _greedy_repair(initial_points)
    
    if time.time() > deadline:
        return sorted(points)
    
    # Phase 2: Local search to grow the set
    points = _local_search(points, deadline, max_swaps=300000)
    
    if time.time() > deadline:
        return sorted(points)
    
    # Phase 3: Final check and cleanup
    # Ensure admissibility
    final_check = _check_all_primes(points)
    if not final_check:
        points = _greedy_repair(points)
    
    return sorted(points)


# EVOLVE-BLOCK-END
