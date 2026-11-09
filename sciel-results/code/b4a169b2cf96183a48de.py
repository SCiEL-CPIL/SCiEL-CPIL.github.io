"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    
    Strategy:
    We reformulate the problem as an optimization over residue assignments.
    For a fixed target size k, we must choose a residue r_p for each prime p <= k.
    The number of survivors is the count of x in [0, N] such that x % p != r_p for all p <= k.
    
    The current greedy approach is myopic. We use Simulated Annealing (SA) to find
    better residue assignments.
    
    1. Precompute primes up to N.
    2. Define a function `evaluate_residues(primes, residues)` that returns the count of survivors.
       We can optimize this by maintaining a boolean array `alive` and updating it efficiently,
       but for SA, we need to evaluate many configurations. 
       
       A faster evaluation method:
       Instead of iterating through all N numbers for every evaluation, we can use the inclusion-exclusion
       principle or simply numpy operations. 
       
       Let's use a numpy-based evaluation for speed.
       For a given set of primes P and residues R:
       Create a boolean array `valid` of size N+1, initially all True.
       For each (p, r) in zip(P, R):
           valid[np.arange(N+1) % p == r] = False
       Return sum(valid).
       
       This is O(P * N) per evaluation. With P ~ 600 and N=5000, that's 3M ops.
       In numpy, this is very fast (~1ms). 
       
       However, SA might need many iterations. 
       We can optimize the evaluation by noting that only the changed prime affects the `valid` array.
       But maintaining state across SA steps is tricky if we want to evaluate independent candidates.
       
       Alternative: Use a "survivor count" update.
       If we change residue for prime p from r_old to r_new:
       The numbers removed by r_old are those x where x%p==r_old and x was not removed by other primes.
       This is complex to update incrementally without tracking the full set of "first removers".
       
       Given N=5000 is small, the full numpy re-evaluation is likely fast enough for a few thousand SA steps.
       Let's estimate: 1000 steps * 1ms/step = 1s. This is well within budget.
       We can afford 10,000 steps.
       
    3. Binary search for the maximum k.
       For each k, run SA to find the max survivors.
       If max_survivors >= k, then k is feasible.
       
    4. SA Details:
       - State: list of residues for primes <= k.
       - Initial state: Greedy result (good starting point).
       - Move: Pick a random prime index i, pick a random new residue r in [0, p_i).
       - Acceptance: Metropolis criterion.
       - Temperature: Geometric cooling.
       
    5. Optimization:
       We can precompute `indices_mod_p` for each prime p to speed up the masking.
       `indices_mod_p[p] = np.arange(N+1) % p`
       
    """
    
    # Precompute primes up to N
    all_primes = _primes(N)
    
    # Precompute modulo indices for each prime
    # This is a dictionary: prime -> np.array of residues for 0..N
    mod_indices = {}
    for p in all_primes:
        mod_indices[p] = np.arange(N + 1) % p

    def get_survivors(primes_list, residues):
        """
        Compute number of survivors given a list of primes and corresponding residues.
        primes_list: list of primes
        residues: list of residues (same length)
        """
        if not primes_list:
            return N + 1
        
        valid = np.ones(N + 1, dtype=bool)
        
        for p, r in zip(primes_list, residues):
            # Mask indices where x % p == r
            mask = (mod_indices[p] == r)
            valid[mask] = False
            
        return np.sum(valid)

    def greedy_residues(primes_list):
        """
        Greedy initialization: process primes in increasing order.
        Choose residue that minimizes new removals.
        """
        residues = []
        valid = np.ones(N + 1, dtype=bool)
        
        for p in primes_list:
            # Count alive numbers in each residue class
            # We need to sum valid[x] where x%p == r
            # Using bincount is fast
            # indices = np.arange(N+1)
            # residues_arr = indices % p
            # counts = np.bincount(residues_arr, weights=valid, minlength=p)
            
            # Re-using precomputed mod_indices
            m_idx = mod_indices[p]
            counts = np.bincount(m_idx, weights=valid, minlength=p)
            
            # Find min count
            min_count = np.min(counts)
            candidates = np.where(counts == min_count)[0]
            r_min = candidates[0]
            
            residues.append(r_min)
            
            # Update valid
            mask = (m_idx == r_min)
            valid[mask] = False
            
        return residues

    def simulated_annealing(primes_list, initial_residues, num_iterations=5000, t_start=1.0, t_end=0.01):
        """
        Run SA to maximize survivors.
        """
        current_residues = initial_residues[:]
        current_score = get_survivors(primes_list, current_residues)
        
        best_residues = current_residues[:]
        best_score = current_score
        
        k = len(primes_list)
        if k == 0:
            return best_score, best_residues
        
        # Precompute prime values for quick access
        p_vals = primes_list
        
        # Temperature schedule
        # Geometric cooling
        cooling_rate = (t_end / t_start) ** (1.0 / num_iterations)
        
        for i in range(num_iterations):
            if time.time() > _DEADLINE - 10:
                break
                
            # Pick a random prime index
            idx = random.randint(0, k - 1)
            p = p_vals[idx]
            old_r = current_residues[idx]
            
            # Pick a new random residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            # Evaluate new state
            new_residues = current_residues[:]
            new_residues[idx] = new_r
            new_score = get_survivors(primes_list, new_residues)
            
            delta = new_score - current_score
            
            # Accept if better, or with probability exp(delta / T) if worse
            if delta > 0:
                current_residues = new_residues
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_residues = current_residues[:]
            else:
                # delta <= 0
                T = t_start * (cooling_rate ** i)
                if T > 0:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_residues = new_residues
                        current_score = new_score
            
        return best_score, best_residues

    # Binary search for the maximum admissible size m
    low = 0
    high = N + 1
    best_m = 0
    best_residues_final = []
    
    # To save time, we can limit the number of SA iterations for smaller m
    # and use more for larger m where the optimization is harder.
    
    while low <= high:
        if time.time() > _DEADLINE - 20:
            break
            
        mid = (low + high) // 2
        
        # Get primes <= mid
        primes_to_use = [p for p in all_primes if p <= mid]
        
        if not primes_to_use:
            # If no primes, any set of size mid is valid (trivially admissible for p<=mid? 
            # Wait, condition is "for every prime p <= len(tuple)". 
            # If len(tuple)=mid, and there are no primes <= mid, the condition is vacuously true.
            # So we can pick any mid numbers.
            # Max size is N+1.
            survivors = N + 1
        else:
            # Get greedy initial guess
            init_res = greedy_residues(primes_to_use)
            
            # Run SA
            # Scale iterations based on remaining time and problem size
            # Larger mid -> more primes -> harder optimization -> more iterations needed?
            # But also larger mid -> fewer survivors -> easier to satisfy mid <= survivors?
            # Actually, we want to maximize survivors.
            
            # Let's use a fixed budget of iterations, say 2000, which should be fast.
            # If time permits, we can do more.
            iter_count = 2000
            if mid > 100:
                iter_count = 3000
                
            survivors, final_res = simulated_annealing(primes_to_use, init_res, num_iterations=iter_count)
            
        if survivors >= mid:
            best_m = mid
            best_residues_final = final_res if primes_to_use else []
            low = mid + 1
        else:
            high = mid - 1

    # Construct the final set
    if best_m == 0:
        return []
        
    # We need to generate the actual set of integers.
    # We have the optimal residues for primes <= best_m.
    # We need to pick best_m integers from [0, N] that are not in the forbidden classes.
    
    primes_to_use = [p for p in all_primes if p <= best_m]
    
    if not primes_to_use:
        # No constraints
        return list(range(best_m))
        
    # If we didn't store the residues properly or best_m changed, re-run greedy/SA?
    # It's safer to just use the stored best_residues_final if it corresponds to best_m.
    # But what if best_m was found with a different set of primes?
    # The binary search updates best_m and best_residues_final together.
    # So best_residues_final should correspond to primes <= best_m.
    
    # However, to be absolutely safe and simple, let's just re-construct the valid set
    # using the residues we found.
    
    # If best_residues_final is empty (shouldn't happen if best_m > 0 and primes exist), 
    # fallback to greedy.
    if not best_residues_final:
        best_residues_final = greedy_residues(primes_to_use)
        
    # Generate valid indices
    valid = np.ones(N + 1, dtype=bool)
    for p, r in zip(primes_to_use, best_residues_final):
        mask = (mod_indices[p] == r)
        valid[mask] = False
        
    result_indices = np.where(valid)[0]
    
    # We need exactly best_m elements.
    if len(result_indices) < best_m:
        # This should not happen if our evaluation was correct, but just in case
        # Return what we have? Or reduce best_m?
        # The evaluator checks if the tuple is admissible and its size.
        # If we return a smaller tuple, the score is smaller.
        # But we claimed best_m was feasible.
        # Let's just return the available ones.
        return result_indices.tolist()
        
    # Pick the first best_m
    final_set = result_indices[:best_m].tolist()
    
    return final_set

# EVOLVE-BLOCK-END
