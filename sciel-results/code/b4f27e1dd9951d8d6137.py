import time
import random
import math
import numpy as np
from itertools import product

N = 5001 # Range [0, 5000]
DEADLINE = 290.0

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# We need primes up to 5003 to guarantee validity for any subset of [0, 5000]
PRIMES = _primes(5003)
P_COUNT = len(PRIMES)

def _compute_score_and_cover(residues):
    """
    Computes cover_count and score for a given residue assignment.
    """
    cover_count = np.zeros(N, dtype=np.int32)
    for i, p in enumerate(PRIMES):
        r = residues[i]
        cover_count[r::p] += 1
    score = N - np.sum(cover_count > 0)
    return cover_count, score

def _greedy_init():
    """
    Greedy initialization: pick residue for each prime that removes fewest available numbers.
    """
    available = np.ones(N, dtype=bool)
    residues = [0] * P_COUNT
    for i, p in enumerate(PRIMES):
        min_count = N
        best_r = 0
        for r in range(p):
            count = available[r::p].sum()
            if count < min_count:
                min_count = count
                best_r = r
        residues[i] = best_r
        available[best_r::p] = False
    return residues

def _hill_climb(residues, cover_count, current_score, time_limit):
    """
    Greedy Hill Climbing: find best single move, apply, repeat until local optimum.
    Returns updated residues, cover_count, current_score.
    """
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_delta = 0
        best_move = None
        
        for i, p in enumerate(PRIMES):
            old_r = residues[i]
            # Calculate gain from removing old_r
            # Elements with count 1 in old_r::p will become 0 (good)
            gain_old = np.sum(cover_count[old_r::p] == 1)
            
            # Optimization: If gain_old is 0, no move for this prime can improve score.
            if gain_old == 0:
                continue
            
            # Precompute loss for all residues at once using a 2D approach
            # For each residue r, loss = number of elements with cover_count==0 in positions r::p
            # We can compute this by reshaping or by looping (p is small)
            min_loss = N
            best_new_r = -1
            
            # Use a vectorized approach: for each r in range(p), count zeros in cover_count[r::p]
            # Since p can be up to ~5000, we need efficiency
            # For small p (< 100), direct loop is fine
            # For larger p, use numpy broadcasting
            if p <= 200:
                for r in range(p):
                    if r == old_r:
                        continue
                    loss = np.sum(cover_count[r::p] == 0)
                    if loss < min_loss:
                        min_loss = loss
                        best_new_r = r
            else:
                # For large primes, sample a subset of residues to keep it fast
                # Focus on residues that might have fewer covered elements
                step = max(1, p // 50)
                candidates = list(range(0, p, step))
                if old_r not in candidates:
                    candidates.append(old_r)
                for r in candidates:
                    if r == old_r:
                        continue
                    loss = np.sum(cover_count[r::p] == 0)
                    if loss < min_loss:
                        min_loss = loss
                        best_new_r = r
            
            if best_new_r != -1:
                delta = gain_old - min_loss
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, best_new_r)
                    
        if best_move is not None:
            i, new_r = best_move
            p = PRIMES[i]
            old_r = residues[i]
            
            # Apply move
            residues[i] = new_r
            
            # Update cover_count
            cover_count[old_r::p] -= 1
            cover_count[new_r::p] += 1
            
            current_score += best_delta
            improved = True
            
    return residues, cover_count, current_score

def _branch_first_primes():
    """
    Branch over all combinations of residues for the first 5 primes (2,3,5,7,11).
    This explores 2*3*5*7*11 = 2310 distinct initial configurations.
    For each, do a quick greedy fill + hill climb, keep the best.
    """
    start_time = time.time()
    first_5_primes = PRIMES[:5]
    first_5_ranges = [range(p) for p in first_5_primes]
    
    best_score = -1
    best_residues = None
    best_cover = None
    
    # We'll iterate through all 2310 combinations
    for combo in product(*first_5_ranges):
        if time.time() > start_time + 60:  # 60 second budget for branching
            break
        
        # Initialize residues with the combo for first 5 primes
        residues = list(combo) + [0] * (P_COUNT - 5)
        
        # Complete with greedy for remaining primes
        available = np.ones(N, dtype=bool)
        for i, p in enumerate(PRIMES[:5]):
            r = residues[i]
            available[r::p] = False
        
        for i in range(5, P_COUNT):
            p = PRIMES[i]
            min_count = N
            best_r = 0
            for r in range(p):
                count = available[r::p].sum()
                if count < min_count:
                    min_count = count
                    best_r = r
            residues[i] = best_r
            available[best_r::p] = False
        
        # Compute score
        cover_count, current_score = _compute_score_and_cover(residues)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            best_cover = cover_count.copy()
    
    return best_residues, best_cover, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Branch over first 5 primes to find a good starting point
    best_residues, best_cover, best_score = _branch_first_primes()
    
    if best_residues is None:
        # Fallback to greedy
        residues = _greedy_init()
        best_cover, best_score = _compute_score_and_cover(residues)
        best_residues = residues.copy()
    
    # Phase 2: Hill climb from the best branched start
    residues = best_residues.copy()
    cover_count = best_cover.copy()
    current_score = best_score
    
    residues, cover_count, current_score = _hill_climb(residues, cover_count, current_score, DEADLINE - 30)
    
    if current_score > best_score:
        best_score = current_score
        best_residues = residues.copy()
    
    # Phase 3: Restarts with Perturbation
    rng = np.random.default_rng()
    phase3_deadline = DEADLINE - 15
    
    while time.time() < phase3_deadline:
        # Perturb the current local optimum to escape it
        perturbed_residues = residues.copy()
        # Change 3-8 random residues (smaller perturbations for finer exploration)
        num_perturb = rng.integers(3, 9)
        indices = rng.choice(P_COUNT, size=num_perturb, replace=False)
        for i in indices:
            p = PRIMES[i]
            perturbed_residues[i] = rng.integers(0, p)
            
        # Recalculate state for perturbed
        perturbed_cover, perturbed_score = _compute_score_and_cover(perturbed_residues)
        
        # Hill Climb from perturbed with remaining time
        remaining = phase3_deadline - time.time()
        if remaining < 5:
            break
        p_res, p_cover, p_score = _hill_climb(perturbed_residues, perturbed_cover, perturbed_score, phase3_deadline - 5)
        
        if p_score > best_score:
            best_score = p_score
            best_residues = p_res.copy()
            
        # Update current state to the result of this climb
        residues = p_res
        cover_count = p_cover
        current_score = p_score
    
    # Phase 4: Final polish - try very small perturbations (1-2 changes)
    polish_deadline = DEADLINE - 5
    while time.time() < polish_deadline:
        perturbed_residues = best_residues.copy()
        num_perturb = rng.integers(1, 3)
        indices = rng.choice(P_COUNT, size=num_perturb, replace=False)
        for i in indices:
            p = PRIMES[i]
            perturbed_residues[i] = rng.integers(0, p)
            
        perturbed_cover, perturbed_score = _compute_score_and_cover(perturbed_residues)
        
        remaining = polish_deadline - time.time()
        if remaining < 3:
            break
        p_res, p_cover, p_score = _hill_climb(perturbed_residues, perturbed_cover, perturbed_score, polish_deadline - 2)
        
        if p_score > best_score:
            best_score = p_score
            best_residues = p_res.copy()
    
    # Reconstruct the set from best_residues
    mask = np.ones(N, dtype=bool)
    for i, p in enumerate(PRIMES):
        r = best_residues[i]
        mask[r::p] = False
        
    result = [x for x in range(N) if mask[x]]
    
    return result
