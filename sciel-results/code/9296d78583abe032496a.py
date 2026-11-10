import time
import math
import numpy as np

N = 5001  # Range [0, 5000]
DEADLINE = 325.0  # Safety margin before 330s kill

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Primes up to 5003 to guarantee validity for any subset of [0, 5000]
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
            zero_mask = (cover_count == 0)
            
            # For this prime, compute losses for all residues
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
            
            if best_new_r != -1:
                delta = gain - min_loss
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, best_new_r)
                    
        if best_move is None:
            # Try moves with delta == 0 to escape plateaus (random tie-breaking)
            for i in candidate_indices:
                p = PRIMES[i]
                old_r = residues[i]
                gain = gains[i]
                zero_mask = (cover_count == 0)
                losses = np.zeros(p, dtype=np.int32)
                for r in range(p):
                    losses[r] = np.sum(zero_mask[r::p])
                min_loss = np.min(np.delete(losses, old_r))
                delta = gain - min_loss
                if delta == 0:
                    # Pick a random residue with min loss
                    valid_rs = [r for r in range(p) if r != old_r and losses[r] == min_loss]
                    if valid_rs:
                        best_new_r = valid_rs[np.random.randint(len(valid_rs))]
                        best_delta = 0
                        best_move = (i, best_new_r)
                        break
        
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

def _branch_first_primes(num_primes=5, time_limit=320.0):
    """
    Branch on the first num_primes primes, exploring all combinations.
    For each combination, run greedy init for remaining primes and hill climb.
    """
    start_time = time.time()
    best_score = -1
    best_residues = None
    
    # Get the first num_primes primes
    first_primes = PRIMES[:num_primes]
    first_primes_count = num_primes
    
    # Generate all combinations of residues for the first primes
    # Use itertools.product for efficiency
    import itertools
    
    # Calculate total number of combinations
    total_combos = 1
    for p in first_primes:
        total_combos *= p
    
    # If too many combos, sample them
    max_combos_to_try = 50000  # Limit to keep within time
    if total_combos <= max_combos_to_try:
        combo_list = list(itertools.product(*[range(p) for p in first_primes]))
    else:
        # Random sampling
        rng = np.random.default_rng()
        combo_list = []
        for _ in range(max_combos_to_try):
            combo = tuple(int(rng.integers(0, p)) for p in first_primes)
            combo_list.append(combo)
    
    for combo in combo_list:
        if time.time() > time_limit:
            break
        
        # Set residues for first primes
        residues = list(combo) + [0] * (P_COUNT - first_primes_count)
        
        # Run greedy for remaining primes
        available = np.ones(N, dtype=bool)
        
        # First apply the fixed residues
        for i in range(first_primes_count):
            p = PRIMES[i]
            r = residues[i]
            available[r::p] = False
        
        # Then greedy for remaining
        for i in range(first_primes_count, P_COUNT):
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
        
        # Quick hill climb with short time limit
        time_remaining = time_limit - time.time()
        if time_remaining > 1.0:
            # Limit hill climb to a small fraction of remaining time
            hc_time = min(0.5, time_remaining * 0.1)
            residues, cover_count, current_score = _hill_climb(
                residues, cover_count, current_score, time.time() + hc_time
            )
            if current_score > best_score:
                best_score = current_score
                best_residues = residues.copy()
    
    return best_residues, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Branch on first 5 primes
    # This explores 2*3*5*7*11 = 2310 distinct initial configurations
    best_residues, best_score = _branch_first_primes(5, DEADLINE - 50)
    
    if best_residues is None:
        # Fallback to standard greedy
        residues = _greedy_init()
        cover_count, current_score = _compute_score_and_cover(residues)
        best_residues = residues.copy()
        best_score = current_score
    else:
        # Recompute for the best found
        cover_count, current_score = _compute_score_and_cover(best_residues)
    
    # Phase 2: Full hill climb from the best configuration
    residues = best_residues.copy()
    residues, cover_count, current_score = _hill_climb(
        residues, cover_count, current_score, DEADLINE - 20
    )
    
    if current_score > best_score:
        best_score = current_score
        best_residues = residues.copy()
    
    # Phase 3: Restarts with perturbation if time remains
    rng = np.random.default_rng()
    
    while time.time() < DEADLINE - 10:
        # Perturb current local optimum
        perturbed_residues = best_residues.copy()
        num_perturb = int(rng.integers(3, 10))
        indices = rng.choice(P_COUNT, size=num_perturb, replace=False)
        for i in indices:
            p = PRIMES[i]
            perturbed_residues[i] = int(rng.integers(0, p))
        
        # Recalculate state for perturbed
        perturbed_cover, perturbed_score = _compute_score_and_cover(perturbed_residues)
        
        # Hill Climb from perturbed
        time_remaining = DEADLINE - 10 - time.time()
        if time_remaining < 1.0:
            break
        hc_time = min(5.0, time_remaining * 0.5)
        p_res, p_cover, p_score = _hill_climb(
            perturbed_residues, perturbed_cover, perturbed_score, time.time() + hc_time
        )
        
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
