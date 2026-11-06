"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _max_sidon_subset_size(S):
    """Compute the largest subset of S with all pair sums (i<=j) unique."""
    S = sorted(S)
    n = len(S)
    
    # Check all subsets of size k from largest down
    # For n=12, we check sizes 12, 11, 10, ...
    # But we can prune: if we find a collision in a subset, skip it
    
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if _is_sidon(subset):
                return k
    return 0

def _is_sidon(S):
    """Check if all pair sums a_i + a_j (i <= j) are distinct."""
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _evaluate(S):
    """Return the max Sidon subset size of S (lower is better)."""
    return _max_sidon_subset_size(S)

def _local_search(initial, time_budget=55):
    """Local search to minimize max Sidon subset size."""
    start_time = time.time()
    best = sorted(initial)
    best_score = _evaluate(best)
    
    if best_score <= 4:
        return best  # Can't do better than 4 (pigeonhole: need 15 sums, range < 15)
    
    rng = random.Random(42)
    
    current = best[:]
    current_score = best_score
    
    # Try to reduce score by swapping elements
    while time.time() - start_time < time_budget:
        # Try random swaps
        improved = False
        for _ in range(200):
            i = rng.randint(0, 11)
            j = rng.randint(0, 11)
            if i == j:
                continue
            new_val = rng.randint(0, 128)
            if new_val in current and current.index(new_val) not in (i, j):
                continue
            
            trial = current[:]
            trial[i] = new_val
            if len(set(trial)) != 12:
                continue
            
            trial_score = _evaluate(trial)
            if trial_score < current_score:
                current = trial
                current_score = trial_score
                improved = True
                if current_score <= 4:
                    break
            elif trial_score == current_score:
                # Accept with some probability (simulated annealing-like)
                if rng.random() < 0.3:
                    current = trial
                    improved = True
        
        if current_score < best_score:
            best = current[:]
            best_score = current_score
            if best_score <= 4:
                break
        
        if not improved and rng.random() < 0.5:
            # Restart from best with perturbation
            current = best[:]
            # Perturb 2-3 elements
            for _ in range(rng.randint(2, 3)):
                idx = rng.randint(0, 11)
                current[idx] = rng.randint(0, 128)
            # Ensure distinct
            for attempt in range(100):
                if len(set(current)) == 12:
                    break
                idx = rng.randint(0, 11)
                current[idx] = rng.randint(0, 128)
            if len(set(current)) != 12:
                current = best[:]
    
    return best

def construct():
    start_time = time.time()
    time_budget = 55  # Safety margin before 90s kill
    
    # Start with seed
    initial = list(range(12))
    seed_score = _evaluate(initial)
    
    best = initial[:]
    best_score = seed_score
    
    # Try several random initializations
    rng = random.Random(12345)
    
    while time.time() - start_time < time_budget:
        # Random initial set
        candidate = rng.sample(range(129), 12)
        candidate_score = _evaluate(candidate)
        
        if candidate_score < best_score:
            best = candidate[:]
            best_score = candidate_score
            if best_score <= 4:
                break
        
        # Local search from this candidate
        if time.time() - start_time < time_budget - 5:
            result = _local_search(candidate, time_budget=10)
            result_score = _evaluate(result)
            if result_score < best_score:
                best = result[:]
                best_score = result_score
                if best_score <= 4:
                    break
    
    # Final local search from best
    if time.time() - start_time < time_budget - 3:
        final = _local_search(best, time_budget=time_budget - 3)
        final_score = _evaluate(final)
        if final_score < best_score:
            best = final[:]
            best_score = final_score
    
    return sorted(best)

# EVOLVE-BLOCK-END
