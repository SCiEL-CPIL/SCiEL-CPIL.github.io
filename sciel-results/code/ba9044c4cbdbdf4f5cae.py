import time
import random
import math
import numpy as np

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
    Optimized Greedy Hill Climbing: vectorized move selection.
    Returns updated residues, cover_count, current_score.
    """
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        best_delta = 0
        best_move = None
        
        # Precompute gain for each prime (how many elements with count==1 in their current residue)
        gains = np.zeros(P_COUNT, dtype=np.int32)
        for i, p in enumerate(PRIMES):
            old_r = residues[i]
            gains[i] = np.sum(cover_count[old_r::p] == 1)
        
        # Only consider primes where gain > 0
        candidate_indices = np.where(gains > 0)[0]
        
        for i in candidate_indices:
            p = PRIMES[i]
            old_r = residues[i]
            gain = gains[i]
            
            # For each possible new residue r, compute loss = count of elements with cover_count==0 in r::p
            # We want to minimize loss
            # Vectorize: for each r in 0..p-1, loss[r] = sum(cover_count[r::p] == 0)
            
            # Efficient approach: precompute zero_mask
            zero_mask = (cover_count == 0)
            
            # For this prime, compute losses for all residues
            # losses[r] = sum(zero_mask[r::p])
            # This is a strided sum
            
            # Use reshape trick if possible, otherwise loop over residues
            # Since p varies, we'll compute losses array
            losses = np.zeros(p, dtype=np.int32)
            for r in range(p):
                losses[r] = np.sum(zero_mask[r::p])
            
            # Exclude old_r
            if p > 1:
                min_loss = np.min(np.delete(losses, old_r))
                # Find best new_r
                best_new_r = -1
                for r in range(p):
                    if r != old_r and losses[r] == min_loss:
                        best_new_r = r
                        break
            else:
                continue
            
            if best_new_r != -1:
                delta = gain - min_loss
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

def construct():
    start_time = time.time()
    
    # Initial residues: Greedy
    residues = _greedy_init()
    cover_count, current_score = _compute_score_and_cover(residues)
    
    best_score = current_score
    best_residues = residues.copy()
    
    # Hill Climb
    residues, cover_count, current_score = _hill_climb(residues, cover_count, current_score, DEADLINE - 5)
    
    if current_score > best_score:
        best_score = current_score
        best_residues = residues.copy()
    
    # Restarts with Perturbation
    rng = np.random.default_rng()
    
    while time.time() < DEADLINE - 10:
        # Perturb current local optimum
        perturbed_residues = residues.copy()
        # Change 5-10 random residues
        num_perturb = rng.integers(3, 10)
        indices = rng.choice(P_COUNT, size=num_perturb, replace=False)
        for i in indices:
            p = PRIMES[i]
            perturbed_residues[i] = rng.integers(0, p)
            
        # Recalculate state for perturbed
        perturbed_cover, perturbed_score = _compute_score_and_cover(perturbed_residues)
        
        # Hill Climb from perturbed
        p_res, p_cover, p_score = _hill_climb(perturbed_residues, perturbed_cover, perturbed_score, DEADLINE - 5)
        
        if p_score > best_score:
            best_score = p_score
            best_residues = p_res.copy()
            
        # Update current state to the result of this climb
        residues = p_res
        cover_count = p_cover
        current_score = p_score

    # Reconstruct the set from best_residues
    mask = np.ones(N, dtype=bool)
    for i, p in enumerate(PRIMES):
        r = best_residues[i]
        mask[r::p] = False
        
    result = [x for x in range(N) if mask[x]]
    
    return result
