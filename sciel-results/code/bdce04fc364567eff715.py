"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START_TIME = time.time()
DEADLINE = 115.0

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return DEADLINE - _elapsed()

def _score_of(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U or len(U) < 2:
        return 0.0
    U = sorted(set(U))
    maxU = max(U)
    denom_log = math.log(2 * maxU + 1)
    if denom_log <= 0:
        return 0.0
    
    # Compute U+U and U-U using sets for simplicity (small enough for our search range)
    sum_set = set()
    diff_set = set()
    n = len(U)
    for i in range(n):
        ui = U[i]
        for j in range(n):
            uj = U[j]
            sum_set.add(ui + uj)
            diff_set.add(ui - uj)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1.0:
        # If ratio <= 1, the log is <= 0, so score <= 1
        # But we want to maximize, and ratio > 1 is preferred
        pass
    
    val = 1.0 + math.log(ratio) / denom_log
    return val

def _score_of_fast(U):
    """Faster scoring using numpy FFT or direct computation for small sets."""
    if not U or len(U) < 2:
        return 0.0
    U = np.sort(np.unique(np.array(U, dtype=np.int64)))
    maxU = int(U[-1])
    denom_log = math.log(2 * maxU + 1)
    if denom_log <= 0:
        return 0.0
    
    # For small sets, direct set construction is fast enough
    # But let's try a vectorized approach for differences and sums
    # Sum set: U+U
    # Diff set: U-U
    
    # Use broadcasting for small n
    if len(U) <= 100:
        u_col = U[:, None]
        u_row = U[None, :]
        sums = u_col + u_row
        diffs = u_col - u_row
        sum_count = np.unique(sums).size
        diff_count = np.unique(diffs).size
    else:
        # Fallback to python sets for larger
        sum_set = set()
        diff_set = set()
        for i in range(len(U)):
            ui = U[i]
            for j in range(len(U)):
                uj = U[j]
                sum_set.add(ui + uj)
                diff_set.add(ui - uj)
        sum_count = len(sum_set)
        diff_count = len(diff_set)
        
    ratio = diff_count / sum_count
    val = 1.0 + math.log(ratio) / denom_log
    return val

def _greedy_construct(n_target, max_val, rng):
    """Greedy construction using marginal gain heuristic."""
    U = [0]
    current_diffs = {0}
    current_sums = {0}
    
    # Pre-compute candidate elements? Or just try randoms?
    # The prompt suggests "descending order of their marginal difference yield"
    # But candidates depend on current U.
    
    # Let's try a local search / hill climbing approach instead, 
    # as greedy with random additions is tricky to order.
    # Actually, let's stick to the prompt's suggestion if possible, 
    # but "marginal difference yield" needs a candidate set.
    
    # Alternative: Start with a small set, then add elements that improve the score.
    # Or start with random set and optimize.
    
    # Let's implement a simple local search:
    # 1. Initialize U with 0 and n_target-1 random small integers.
    # 2. Iteratively try to add a new element (up to max_val) or replace an existing one?
    # Adding elements increases span (denominator) but potentially increases ratio.
    # Replacing is complex. Let's try growing from small to n_target.
    
    # Actually, the prompt says "randomized greedy algorithm that selects elements in descending order of their 'marginal difference yield'"
    # This implies we have a pool of candidates.
    
    # Let's try this:
    # Maintain a set of "promising" candidates.
    # Start with U = [0].
    # At each step, consider a pool of candidates (e.g., 0 to max_val).
    # For each candidate c not in U, calculate the new score if c is added.
    # Pick the best one.
    # Stop when size is n_target or no improvement.
    
    # To speed this up, we can't check all 0..max_val every time if max_val is large.
    # But max_val for n~10-20 is small (e.g., 100).
    
    U_set = {0}
    U_list = [0]
    
    for _ in range(n_target - 1):
        if _elapsed() > DEADLINE - 5:
            break
            
        best_gain = -np.inf
        best_c = None
        
        # Candidate range: 0 to max_val
        # We can prune: candidates that are too large might be bad, 
        # but we don't know yet. Let's try a subset of candidates to save time.
        # Or just iterate 0..max_val if max_val is small.
        
        # Let's estimate max_val for n elements. 
        # If we use powers of 2, max is 2^(n-1).
        # For n=10, max=512. 512 candidates is fine.
        
        for c in range(0, max_val + 1):
            if c in U_set:
                continue
            
            # Calculate new score
            new_U = U_list + [c]
            s = _score_of_fast(new_U)
            
            # We want to maximize score. 
            # But _score_of_fast is O(n^2). 
            # Doing this for all c in range(max_val) is O(max_val * n^2).
            # For n=10, max_val=100, this is 100 * 100 = 10k ops per step. 
            # 10 steps -> 100k ops. Very fast.
            
            if s > best_gain:
                best_gain = s
                best_c = c
        
        if best_c is None:
            break
            
        if best_gain < _score_of_fast(U_list) - 1e-6:
            # If adding any element decreases score, stop?
            # Or maybe we should allow it if we haven't reached target size?
            # The objective is to maximize score, not necessarily size.
            # But the task says "Construct a finite set ... to maximize ...".
            # It doesn't fix the size.
            # So we should stop if adding elements hurts the score.
            # However, sometimes a "bad" addition enables a better subsequent addition?
            # Greedy doesn't handle that. 
            # Let's stop if no improvement.
            break
            
        U_set.add(best_c)
        U_list.append(best_c)
        U_list.sort()
        
    return U_list

def _local_search(U, max_val, iterations, rng):
    """Improve a set by trying to add, remove, or swap elements."""
    U = sorted(set(U))
    best_score = _score_of_fast(U)
    
    for _ in range(iterations):
        if _elapsed() > DEADLINE - 5:
            break
            
        improved = False
        
        # 1. Try adding a random element
        # 2. Try removing a random element
        # 3. Try swapping an element with a random other
        
        # Add
        c = rng.randint(0, max_val + 1)
        if c not in U:
            new_U = U + [c]
            new_U.sort()
            s = _score_of_fast(new_U)
            if s > best_score:
                U = new_U
                best_score = s
                improved = True
        
        # Remove
        if len(U) > 1:
            idx = rng.randint(0, len(U))
            if U[idx] != 0:  # Must keep 0
                new_U = [x for i, x in enumerate(U) if i != idx]
                s = _score_of_fast(new_U)
                if s > best_score:
                    U = new_U
                    best_score = s
                    improved = True
        
        # Swap
        if len(U) > 1:
            idx = rng.randint(0, len(U))
            if U[idx] != 0:
                c = rng.randint(0, max_val + 1)
                if c not in U:
                    new_U = U[:]
                    new_U[idx] = c
                    new_U.sort()
                    s = _score_of_fast(new_U)
                    if s > best_score:
                        U = new_U
                        best_score = s
                        improved = True
                    
        if not improved:
            # Try a few more random moves to escape local optima?
            # Or just break if we've tried a few times without improvement?
            # For simplicity, break.
            # In a real implementation, we'd use simulated annealing.
            # Let's try 10 non-improving moves then break.
            # This is a simple hill climb, so we break.
            break
            
    return U

def construct():
    rng = np.random.default_rng(42)
    best_U = [0]
    best_score = 0.0
    
    # Try greedy constructions with different parameters
    # n_target: 2 to 15
    # max_val: 50 to 200
    
    # The prompt mentions sizes 2003 and 54265, but those are likely for specific constructions.
    # Given the 120s limit, we can't search huge sets.
    # The score formula penalizes large max(U).
    # Small sets with high ratio are likely better.
    
    # Let's try a few random initializations and local search.
    
    # Candidate 1: Greedy
    for n_target in range(2, 12):
        if _elapsed() > DEADLINE - 10:
            break
        for max_val in [20, 50, 100, 200]:
            if _elapsed() > DEADLINE - 10:
                break
            U = _greedy_construct(n_target, max_val, rng)
            # Local search to polish
            U = _local_search(U, max_val, 50, rng)
            s = _score_of_fast(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
                
    # Candidate 2: Random initialization + Local Search
    for trial in range(50):
        if _elapsed() > DEADLINE - 10:
            break
        
        # Random size 2-15
        n = rng.randint(2, 15)
        max_val = rng.randint(20, 200)
        
        # Random set containing 0
        U = [0]
        other = rng.choice(np.arange(1, max_val + 1), size=n-1, replace=False)
        U.extend(other)
        U.sort()
        
        # Local search
        U = _local_search(U, max_val, 100, rng)
        s = _score_of_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # Candidate 3: Powers of 2 variations
    for n in range(2, 10):
        if _elapsed() > DEADLINE - 10:
            break
        U = [1 << i for i in range(n)]
        # Local search
        max_val = U[-1] * 2
        U = _local_search(U, max_val, 50, rng)
        s = _score_of_fast(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
            
    # Ensure 0 is in the set and it's a list of ints
    if 0 not in best_U:
        best_U.append(0)
    best_U = sorted(set(best_U))
    
    return best_U

# EVOLVE-BLOCK-END
