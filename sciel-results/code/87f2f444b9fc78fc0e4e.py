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

def _greedy_residue_removal(x_indices, active, PRIMES, lookahead=1):
    """
    Greedy residue removal with lookahead tie-breaking.
    """
    current_size = int(np.sum(active))
    
    # Precompute residues for all primes? No, do on the fly.
    # But we need to check the "next" prime's state.
    
    for i, p in enumerate(PRIMES):
        if p > current_size:
            break
        
        # Count active elements in each residue class mod p
        residues = x_indices[active] % p
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
        
        if len(candidate_classes) == 1:
            r = candidate_classes[0]
        else:
            # Lookahead: pick the candidate that minimizes the min_count of the next prime
            best_r = candidate_classes[0]
            best_score = float('inf')
            
            # Find the next prime to check
            next_p = None
            for q in PRIMES[i+1:]:
                if q <= current_size - min_count: # Approx check
                    next_p = q
                    break
            
            if next_p is not None:
                for r in candidate_classes:
                    # Simulate removal
                    # We need to count residues mod next_p for active & (x % p != r)
                    # To avoid creating a new boolean array every time, we can do:
                    # new_active_indices = x_indices[active & (x_indices % p != r)]
                    # This is O(N).
                    
                    mask = (x_indices % p) != r
                    new_active = active & mask
                    new_residues = x_indices[new_active] % next_p
                    new_counts = np.bincount(new_residues, minlength=next_p)
                    new_min = np.min(new_counts)
                    
                    # We want to minimize the "difficulty" for the next step.
                    # A smaller min_count means we have to remove fewer elements later?
                    # Actually, if min_count is 0, it's already satisfied.
                    # If min_count is large, it's "harder" to satisfy?
                    # Let's say we prefer candidates that leave the next prime with min_count == 0 if possible.
                    # Or generally, prefer lower min_count?
                    # If min_count is 0, we skip removal. That's good.
                    # So score = new_min. Lower is better.
                    
                    if new_min < best_score:
                        best_score = new_min
                        best_r = r
            else:
                # No lookahead, pick random or smallest
                best_r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
            
            r = best_r
            
        # Deactivate elements in this class
        mask = (x_indices % p) == r
        current_size -= min_count
        active = active & (x_indices % p != r)
        
        if current_size == 0:
            break
    
    return active

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = 0
    best_points = []
    
    x_indices = np.arange(RANGE)
    
    # Beam Search Parameters
    BEAM_WIDTH = 20
    MAX_BRANCHES = 5
    
    # Initial beam: one state with all elements active
    # State: (active_mask, current_size)
    beam = [(np.ones(RANGE, dtype=bool), RANGE)]
    
    for p in PRIMES:
        if time.time() > deadline - 10:
            break
            
        next_beam = []
        
        for active, size in beam:
            # If this state cannot beat the current best, prune it
            if size <= best_size:
                continue
            
            # If the current prime is larger than the set size,
            # no more constraints apply (since primes are increasing).
            if p > size:
                # This is a valid complete admissible tuple
                if size > best_size:
                    best_size = size
                    best_points = np.where(active)[0].tolist()
                # Do not add to next_beam, it's terminal
                continue
            
            # Check if constraint for p is already satisfied
            # Get active indices
            active_indices = np.where(active)[0]
            
            # Count residues
            res_counts = np.bincount(active_indices % p, minlength=p)
            
            if np.min(res_counts) == 0:
                # Constraint satisfied, keep state
                next_beam.append((active, size))
            else:
                # Constraint violated, must remove a residue class
                min_count = np.min(res_counts)
                candidate_classes = np.where(res_counts == min_count)[0]
                
                # Limit branching factor
                if len(candidate_classes) > MAX_BRANCHES:
                    # Randomly select MAX_BRANCHES
                    idx = np.random.choice(len(candidate_classes), MAX_BRANCHES, replace=False)
                    candidate_classes = candidate_classes[idx]
                
                for r in candidate_classes:
                    # Create mask for keeping elements NOT in residue r
                    mask = (x_indices % p) != r
                    new_active = active & mask
                    new_size = size - min_count
                    next_beam.append((new_active, new_size))
        
        if not next_beam:
            break
            
        # Prune beam to BEAM_WIDTH
        # We want the largest sizes
        scored_beam = []
        for active, size in next_beam:
            scored_beam.append((size, active))
            
        # Sort by size descending
        scored_beam.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top BEAM_WIDTH
        beam = [(a, s) for s, a in scored_beam[:BEAM_WIDTH]]
        
    return best_points
