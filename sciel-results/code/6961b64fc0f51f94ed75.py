import numpy as np
import time
import random
from collections import defaultdict

N = 5000
RANGE = N + 1

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def _is_admissible(points, primes):
    """Check if a set of points is admissible."""
    n = len(points)
    if n == 0:
        return True
    point_set = set(points)
    for p in primes:
        if p > n:
            break
        # Check if all residue classes mod p are covered
        residues = set(x % p for x in point_set)
        if len(residues) == p:
            return False
    return True

def _greedy_construct(seed, deadline):
    """
    Construct an admissible tuple using greedy residue removal.
    Returns the list of points.
    """
    random.seed(seed)
    active = np.ones(RANGE, dtype=bool)
    x_indices = np.arange(RANGE)
    current_size = RANGE
    
    # Precompute residues for speed? No, just use numpy.
    
    for p in PRIMES:
        if time.time() > deadline:
            break
        if p > current_size:
            break
        
        # Count active elements in each residue class mod p
        # Only consider active elements
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        if len(residues) == 0:
            break
        
        # Count occurrences of each residue
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            # Already admissible for this prime
            continue
        
        # Find all classes with min count
        candidate_classes = np.where(counts == min_count)[0]
        
        # Randomly pick one
        r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
        
        # Deactivate elements in this class
        # We need to deactivate only the ACTIVE ones in this class
        # mask = (x_indices % p) == r AND active
        mask = (x_indices % p == r) & active
        current_size -= np.sum(mask)
        active = active & ~mask
        
        if current_size == 0:
            break
    
    return np.where(active)[0].tolist()

def _polish(points, deadline):
    """
    Try to add back removed elements to maximize size.
    """
    # This is a heuristic polish.
    # We know the current set is admissible.
    # We try to add elements from [0, N] that are not in points.
    
    point_set = set(points)
    current_size = len(points)
    
    # Get all possible candidates to add
    candidates = [x for x in range(RANGE) if x not in point_set]
    random.shuffle(candidates)
    
    # To make this fast, we can check admissibility incrementally?
    # Or just try a few.
    # Given N=5000, and points likely ~2000-3000, candidates ~2000.
    # Checking admissibility for each candidate is O(P * N) which is too slow for 2000 candidates.
    
    # Optimization:
    # A set S is admissible if for all p <= |S|, not all residues mod p are in S.
    # If we add x to S, S' = S U {x}.
    # S' is non-admissible if there exists p <= |S'| such that S' covers all residues mod p.
    # Since S was admissible, for all p <= |S|, S did NOT cover all residues.
    # So S' can only fail if:
    # 1. p <= |S| and x fills the last missing residue for p.
    # 2. p = |S|+1 (if prime) and S' covers all residues mod p. (Unlikely if |S|+1 > |S|, so S had at most |S| elements, cannot cover all |S|+1 residues unless |S|+1 <= |S| which is false. Wait. If p = |S|+1, S has |S| elements. It cannot cover all p residues because there are p residues and only |S| elements. So adding 1 element makes |S|+1 elements. It *can* cover all p residues if the elements were distinct residues and x fills the last one. But since S had |S| elements, it occupied |S| distinct residues (or fewer). If it occupied |S| distinct residues, there is 1 missing. If x is that missing one, it covers all. So yes, possible.)
    
    # So we only need to check primes p <= |S| where S has exactly 1 missing residue, and p = |S|+1 if prime.
    
    # Let's identify "critical" primes for the current set S.
    # A prime p is critical if |S| >= p and the number of distinct residues of S mod p is p-1.
    # If p is critical, let r_p be the missing residue.
    # Then x can be added only if x % p != r_p for all critical p.
    # And if p = |S|+1 is prime, we need to check if S U {x} covers all residues mod p.
    
    # Let's compute critical primes for the initial set.
    # This requires iterating through primes and counting residues.
    
    def get_critical_primes(S):
        critical = {}
        n = len(S)
        for p in PRIMES:
            if p > n:
                break
            residues = set(x % p for x in S)
            if len(residues) == p - 1:
                # Find the missing residue
                all_residues = set(range(p))
                missing = all_residues - residues
                # Should be exactly one
                if len(missing) == 1:
                    critical[p] = missing.pop()
        return critical

    critical = get_critical_primes(points)
    
    added_count = 0
    # We will iterate through candidates.
    # If we add an element, the critical set changes.
    # This is complex. 
    # Given time constraints, let's just try to add elements and verify full admissibility 
    # but limit the number of attempts or use a faster check.
    
    # Actually, for N=5000, a full check is ~250 primes * 3000 elements = 750k ops.
    # 2000 candidates * 750k = 1.5 billion ops. Too slow in Python.
    
    # Let's skip the complex dynamic polish and just do a simple local search:
    # Swap elements? No, we want to ADD.
    
    # Alternative: Just return the greedy result. The multi-start is the main gain.
    # The polish might be too slow to implement correctly and fast enough.
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = 0
    best_points = []
    
    # Multi-start greedy
    # We will run the greedy construction multiple times with different seeds.
    # Each run takes O(N * pi(N)) which is fast.
    
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        seed = attempts
        points = _greedy_construct(seed, deadline)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points
            
        # Optional: If we have time, try a polish on the best found so far?
        # Or just keep restarting.
        
    return best_points
