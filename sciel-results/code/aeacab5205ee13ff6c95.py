"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000
DEADLINE = 325  # seconds, safety margin before 330s kill


def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes):
    """Check if points form an admissible tuple: for every prime p <= len(points),
    some residue class mod p is missed."""
    n = len(points)
    # Only need to check primes <= n
    for p in primes:
        if p > n:
            break
        # Check if all residue classes mod p are covered
        seen = set()
        for v in points:
            seen.add(v % p)
            if len(seen) == p:
                # All classes covered -> not admissible for this p
                return False
    return True


def construct():
    start_time = time.time()
    
    # Get all primes up to N (we'll need primes up to len(tuple) <= N)
    all_primes = _primes(N)
    
    # Strategy: Start with the full set [0, N] and greedily remove elements
    # to make it admissible. We want to maximize the remaining size.
    # 
    # Key insight: For each prime p <= len(tuple), we need at least one
    # residue class mod p to be "missed" (no elements in that class).
    # 
    # Greedy approach: Start with all elements [0, N]. For each prime p 
    # (in some order), if all residue classes mod p are covered, remove
    # the least-populated residue class. Repeat until admissible.
    #
    # To maximize size, we should:
    # 1. Process primes in an order that minimizes total removals
    # 2. When removing, remove the class with fewest elements
    
    # Start with all points
    points = list(range(N + 1))
    
    # We'll iteratively fix violations
    # For efficiency, track which primes are currently violated
    # A prime p is violated if all p residue classes are present
    
    def _get_violated_primes(points, primes):
        """Return list of primes p <= len(points) where all residue classes are covered."""
        n = len(points)
        violated = []
        for p in primes:
            if p > n:
                break
            seen = set()
            for v in points:
                seen.add(v % p)
                if len(seen) == p:
                    violated.append(p)
                    break
            else:
                continue
        return violated
    
    # Greedy removal loop
    # We'll repeatedly find a violated prime and remove the least-populated class
    # To be smart about it, we can try multiple strategies and keep the best
    
    # Strategy 1: Process primes in increasing order
    # Strategy 2: Process primes in decreasing order  
    # Strategy 3: Process the "most constrained" prime first (smallest class)
    
    best_result = points[:]
    best_size = len(best_result)
    
    # Try multiple greedy strategies
    strategies = [
        "increasing",  # smallest primes first
        "decreasing",  # largest primes first
        "smallest_class",  # prime with smallest least-populated class
    ]
    
    for strategy in strategies:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        pts = list(range(N + 1))
        
        while True:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            n = len(pts)
            # Find violated primes
            violated = []
            for p in all_primes:
                if p > n:
                    break
                seen = set()
                for v in pts:
                    seen.add(v % p)
                    if len(seen) == p:
                        violated.append(p)
                        break
                else:
                    continue
            
            if not violated:
                # Admissible!
                if len(pts) > best_size:
                    best_result = pts[:]
                    best_size = len(pts)
                break
            
            # Choose which prime to fix
            if strategy == "increasing":
                p = min(violated)
            elif strategy == "decreasing":
                p = max(violated)
            else:  # smallest_class
                # Find the prime whose least-populated class is smallest
                best_p = None
                best_min_count = float('inf')
                for p in violated:
                    counts = [0] * p
                    for v in pts:
                        counts[v % p] += 1
                    min_count = min(counts)
                    if min_count < best_min_count:
                        best_min_count = min_count
                        best_p = p
                p = best_p
            
            # Find the least-populated residue class for p
            counts = [0] * p
            for v in pts:
                counts[v % p] += 1
            
            min_count = min(counts)
            # Find all classes with min_count (in case of ties, pick the one that helps most)
            # For simplicity, pick the first one with min_count
            target_class = counts.index(min_count)
            
            # Remove all elements in that residue class
            pts = [v for v in pts if v % p != target_class]
        
        if len(pts) > best_size:
            best_result = pts[:]
            best_size = len(pts)
    
    # Additional optimization: try to add back elements that were removed
    # if they don't violate admissibility
    if time.time() - start_time < DEADLINE - 10:
        # Try to greedily add back elements from [0, N] that aren't in best_result
        current = set(best_result)
        for v in range(N + 1):
            if time.time() - start_time > DEADLINE - 5:
                break
            if v not in current:
                # Try adding v
                test_set = current | {v}
                test_list = list(test_set)
                if _is_admissible(test_list, all_primes):
                    current = test_set
                    best_result = list(current)
                    best_size = len(best_result)
    
    return best_result
