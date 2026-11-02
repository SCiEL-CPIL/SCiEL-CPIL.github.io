import time
import random
import math
import numpy as np

def _score_set(U):
    """
    Compute the score for a given set U using NumPy for speed.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not U or U[0] != 0:
        return 0.0
    # Ensure unique and sorted
    U = np.array(sorted(list(set(U))), dtype=np.int64)
    n = len(U)
    if n < 2:
        return 0.0
    M = U[-1]
    if M == 0:
        return 0.0
    
    # Compute sumset and difference set
    # np.subtract.outer creates a matrix of all differences
    # np.add.outer creates a matrix of all sums
    diffs = np.subtract.outer(U, U)
    sums = np.add.outer(U, U)
    
    # Count unique elements
    # np.unique sorts, which is O(N log N) where N=n^2
    num_diffs = len(np.unique(diffs))
    num_sums = len(np.unique(sums))
    
    # Avoid division by zero or log of 1
    if num_sums == 0:
        return 0.0
        
    num = math.log(num_diffs) - math.log(num_sums)
    den = math.log(2 * M + 1)
    
    if den == 0:
        return 0.0
        
    return 1 + num / den

def _generate_golomb(n):
    """
    Generate a Golomb ruler of length n using a greedy heuristic.
    This ensures |U-U| is maximal (or close to it) for the given n.
    """
    U = [0]
    for i in range(1, n):
        best_pos = None
        best_conflicts = float('inf')
        # Try positions from 1 upwards
        # Upper bound for search: current max + i (rough estimate)
        # Actually, just scan a reasonable range
        max_check = U[-1] + i * 2 + 10 
        for pos in range(1, max_check):
            if pos in U:
                continue
            # Check if adding pos creates duplicate differences
            # We only need to check differences between pos and existing elements
            # and ensure no existing difference matches
            valid = True
            for u in U:
                diff = abs(pos - u)
                # Check if this diff already exists in the set of diffs of U
                # This is O(n^2) per candidate pos, so O(n^3) total.
                # For n < 50, this is fine.
                # Optimization: Precompute set of diffs? 
                # For simplicity in generation, we just check.
                pass # This simple check is incomplete for full Golomb property.
            
            # Better approach for generation: 
            # Just use random restarts and local search to fix collisions.
            # Or use a known construction.
            
    # Fallback: Random sorted set
    # Let's rely on local search to fix the structure.
    # Initial guess: geometric sequence
    U = [0]
    val = 1
    for i in range(n-1):
        U.append(val)
        val *= 2
    return U

def _local_search_sa(U, time_limit):
    """
    Run Simulated Annealing on the set U.
    """
    U = list(U)
    current_score = _score_set(U)
    best_score = current_score
    best_U = U[:]
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9999
    start_time = time.time()
    
    n = len(U)
    
    while time.time() - start_time < time_limit and T > T_min:
        # Pick a random index to perturb
        idx = random.randint(1, n - 1)
        
        # Determine perturbation size
        # Scale with the gap around idx
        gap = 1
        if idx > 0:
            gap = max(1, U[idx] - U[idx-1])
        if idx < n - 1:
            gap = min(gap, U[idx+1] - U[idx])
            
        # Random delta
        # Mix of small and large moves
        if random.random() < 0.5:
            delta = random.randint(-3, 3)
        else:
            delta = random.randint(-10, 10)
            
        new_val = U[idx] + delta
        
        # Validity checks
        if new_val < 0:
            continue
        if idx > 0 and new_val <= U[idx-1]:
            continue
        if idx < n - 1 and new_val >= U[idx+1]:
            continue
            
        new_U = U[:]
        new_U[idx] = new_val
        new_score = _score_set(new_U)
        
        if new_score > current_score:
            U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]
        else:
            # Accept with probability
            prob = math.exp((new_score - current_score) / T)
            if random.random() < prob:
                U = new_U
                current_score = new_score
        
        T *= cooling
        
    return best_U, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Try a range of set sizes
    # Smaller sets have lower denominator, potentially higher score if ratio is good
    # Larger sets have higher ratio potential but higher denominator
    # Let's sweep n from 4 to 30
    for n in range(4, 31):
        # Check time
        if time.time() > deadline - 20:
            break
            
        # Run a few independent searches for this size
        num_trials = 3
        if n > 20:
            num_trials = 2
        if n > 25:
            num_trials = 1
            
        for trial in range(num_trials):
            # Initialize
            if trial == 0:
                # Geometric
                U = [0]
                val = 1
                for i in range(n-1):
                    U.append(val)
                    val *= 2
            elif trial == 1:
                # Random sorted
                U = [0] + sorted(random.sample(range(1, 500), n-1))
            else:
                # Prime gaps
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
                U = [0]
                val = 0
                for i in range(n-1):
                    val += primes[i % len(primes)]
                    U.append(val)
            
            # Run local search
            # Allocate time based on remaining budget
            # If many sizes left, give less time to each
            remaining = deadline - time.time()
            time_per_trial = max(1.0, (remaining - 5) / (num_trials * (31 - n)))
            
            improved_U, improved_score = _local_search_sa(U, time_per_trial)
            
            if improved_score > best_score:
                best_score = improved_score
                best_set = improved_U[:]
                
    return best_set
