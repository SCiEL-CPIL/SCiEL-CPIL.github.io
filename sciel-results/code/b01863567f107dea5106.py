"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000

# EVOLVE-BLOCK-START

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

# Precompute primes up to N for the admissibility check (primes <= len(tuple) <= N)
_PRIMES_UP_TO_N = _primes(N)

def _is_admissible(points, primes=None):
    """Check if the set of points is admissible: for every prime p <= len(points),
    the points do not cover all residue classes mod p."""
    if primes is None:
        primes = _PRIMES_UP_TO_N
    n = len(points)
    for p in primes:
        if p > n:
            break
        residues = set()
        for v in points:
            residues.add(v % p)
        if len(residues) == p:
            return False
    return True

def _score_tuple(points):
    """Return the size if admissible, else 0."""
    if _is_admissible(points):
        return len(points)
    return 0

def construct():
    """Construct the largest admissible tuple within [0, 5000].
    
    Strategy: Start with all integers in [0, N] and iteratively remove elements
    to make the set admissible. We use a greedy approach that prioritizes removing
    elements that are 'most constrained' - i.e., elements that complete many
    residue classes for small primes.
    
    We also try multiple strategies and keep the best result.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    best_points = None
    best_score = 0
    
    # Strategy 1: Start with all points, remove greedily
    # We want to remove points that are in "full" residue classes for small primes.
    # A point is "bad" if for some small prime p, removing it would break a full class.
    
    def greedy_remove(points, primes):
        """Greedy removal: repeatedly find a prime p where all residues are covered,
        and remove the point that is in the most 'critical' position."""
        points = list(points)
        if not _is_admissible(points, primes):
            pass  # Need to fix
        
        # Build a frequency map for each prime
        # For each prime p <= len(points), check if all residues are present.
        # If so, we must remove at least one point from each residue class... 
        # Actually, we just need to miss at least one residue class mod p.
        # So we need to remove all points from at least one residue class mod p.
        
        # Better approach: For each prime p where all residues are covered,
        # we have a constraint. We want to remove points to satisfy all constraints
        # while removing as few points as possible.
        
        # This is a hitting set / set cover problem. Let's use a greedy heuristic.
        
        # For each prime p <= len(points), if all p residue classes are non-empty,
        # we need to "break" at least one class. Breaking class r mod p means removing
        # all points x where x % p == r.
        
        # Greedy: repeatedly find the prime p with the smallest minimum class size,
        # and remove all points in that smallest class.
        
        changed = True
        while changed:
            changed = False
            n = len(points)
            if n == 0:
                break
            
            # Find the "most constrained" prime
            best_p = None
            best_class = None
            best_min_size = float('inf')
            
            for p in primes:
                if p > n:
                    break
                classes = {}
                for v in points:
                    r = v % p
                    if r not in classes:
                        classes[r] = []
                    classes[r].append(v)
                
                if len(classes) == p:
                    # All residues covered - need to break one
                    min_size = min(len(c) for c in classes.values())
                    if min_size < best_min_size:
                        best_min_size = min_size
                        best_p = p
                        best_class = min(classes.values(), key=len)
            
            if best_p is not None:
                # Remove all points in best_class
                to_remove = set(best_class)
                points = [v for v in points if v not in to_remove]
                changed = True
        
        return points
    
    # Try Strategy 1: Full greedy from all points
    all_points = list(range(N + 1))
    points1 = greedy_remove(all_points, _PRIMES_UP_TO_N)
    score1 = _score_tuple(points1)
    if score1 > best_score:
        best_score = score1
        best_points = points1
    
    if time.time() > deadline:
        return best_points if best_points is not None else [1]
    
    # Strategy 2: Start with a random subset and greedily add
    import random
    random.seed(42)
    
    def greedy_add(points, primes):
        """Try to add more points to an admissible set."""
        points = set(points)
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        
        for v in candidates:
            if v in points:
                continue
            test_points = list(points) + [v]
            if _is_admissible(test_points, primes):
                points.add(v)
        
        return list(points)
    
    # Start with small random sets and grow
    for trial in range(10):
        if time.time() > deadline:
            break
        size = random.randint(100, 500)
        initial = random.sample(range(N + 1), min(size, N + 1))
        points2 = greedy_add(initial, _PRIMES_UP_TO_N)
        score2 = _score_tuple(points2)
        if score2 > best_score:
            best_score = score2
            best_points = points2
    
    if time.time() > deadline:
        return best_points if best_points is not None else [1]
    
    # Strategy 3: Local search / hill climbing
    # Try to improve the best solution found so far by swapping elements
    
    def local_search(points, primes, max_iterations=10000):
        """Hill climbing: try to add points, remove points, or swap to improve size."""
        points = list(points)
        current_score = len(points)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a random point
            for _ in range(10):
                v = random.randint(0, N)
                if v not in points:
                    test = points + [v]
                    if _is_admissible(test, primes):
                        points = test
                        current_score += 1
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try swapping: remove one, add another
            if len(points) > 1:
                for _ in range(10):
                    idx = random.randint(0, len(points) - 1)
                    removed = points[idx]
                    v = random.randint(0, N)
                    if v == removed:
                        continue
                    if v in points:
                        continue
                    test = [p for i, p in enumerate(points) if i != idx] + [v]
                    if _is_admissible(test, primes):
                        # Same size, but might enable further additions
                        # Only accept if it leads to improvement later
                        # For now, accept with small probability to explore
                        if random.random() < 0.1:
                            points = test
                            improved = True
                            break
        
        return points
    
    if best_points is not None:
        improved_points = local_search(best_points, _PRIMES_UP_TO_N, max_iterations=5000)
        score3 = _score_tuple(improved_points)
        if score3 > best_score:
            best_score = score3
            best_points = improved_points
    
    return best_points if best_points is not None else [1]

# EVOLVE-BLOCK-END
