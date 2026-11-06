import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes)
    
    # The range of integers to check.
    # We check up to 2000 to allow the annealing to push the frontier.
    MAX_CHECK = 2000
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p for all x and p.
    # Shape: (MAX_CHECK, n_primes)
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Objective Function:
    # We want to maximize the length L such that every x in 1..L has count >= 2.
    # A dense objective that correlates with this is the sum of squared counts,
    # or simply the sum of counts, but squared emphasizes the "doubly covered" aspect.
    # Let's use: Score = sum(counts^2) - penalty for counts < 2 in the prefix?
    # Actually, a better objective for SA is to maximize the "coverage depth" globally.
    # Let's try: Objective = sum_{x=1}^{MAX_CHECK} (min(counts[x], 2))
    # This directly rewards every point being at least doubly covered.
    # If a point is covered 0, 1, 2, 3 times, it contributes 0, 1, 2, 2.
    # Maximizing this sum is equivalent to maximizing the total "satisfied" units of double coverage.
    # This is a smoother landscape than the step-function "first gap" score.
    
    def compute_objective(residues):
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        # Clip counts at 2 because we only care about "at least 2"
        # This makes the objective flat for counts > 2, which is good (no over-optimization).
        effective_counts = np.minimum(counts, 2)
        return np.sum(effective_counts)
    
    def compute_score(residues):
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)
            
    # Initialize
    # Start with a random configuration or a heuristic one.
    # r_p = 1 is a decent start.
    current_residues = np.ones(n_primes, dtype=np.int64)
    current_obj = compute_objective(current_residues)
    current_score = compute_score(current_residues)
    
    best_residues = current_residues.copy()
    best_score = current_score
    best_obj = current_obj
    
    # Simulated Annealing Parameters
    T = 100.0
    T_min = 0.1
    cooling_rate = 0.9995 # Tune this based on steps needed
    
    # Precompute indices for fast random selection
    # We will pick a random prime index to change
    
    step = 0
    while time.time() < deadline - 1.0:
        # Batch processing to speed up SA?
        # Each step:
        # 1. Pick random prime p_idx
        # 2. Pick random new residue r_new in [0, p-1]
        # 3. Compute delta objective
        
        # To speed up, we can do multiple steps per time check.
        # But we need to check time frequently.
        
        for _ in range(1000): # Inner loop
            if time.time() > deadline - 1.0:
                break
                
            p_idx = np.random.randint(0, n_primes)
            p_val = primes[p_idx]
            old_r = current_residues[p_idx]
            
            if p_val == 1:
                continue # p is always >= 2
            
            # Random new residue
            if np.random.random() < 0.5:
                # Small perturbation: +/- 1 mod p
                new_r = (old_r + np.random.choice([-1, 1])) % p_val
            else:
                # Large perturbation: random
                new_r = np.random.randint(0, p_val)
                
            if new_r == old_r:
                continue
                
            # Compute Delta Objective
            # Old column match: M_old = (x_mod_p[:, p_idx] == old_r)
            # New column match: M_new = (x_mod_p[:, p_idx] == new_r)
            # Old counts: C_old = counts
            # New counts: C_new = C_old - M_old + M_new
            
            # We need the current counts to compute the objective change.
            # Maintaining counts array is efficient.
            
            # Let's maintain a counts array.
            # Initially:
            if not hasattr(construct, '_counts'):
                 match = (x_mod_p == current_residues[None, :])
                 counts = match.sum(axis=1)
            else:
                 counts = construct._counts
            
            # Update counts for this move
            M_old = (x_mod_p[:, p_idx] == old_r)
            M_new = (x_mod_p[:, p_idx] == new_r)
            
            # Delta counts
            delta_counts = M_new.astype(np.int8) - M_old.astype(np.int8)
            
            # Old effective counts contribution
            old_eff = np.minimum(counts, 2).astype(np.int32)
            
            # New counts
            new_counts = counts + delta_counts
            new_eff = np.minimum(new_counts, 2).astype(np.int32)
            
            delta_obj = np.sum(new_eff) - np.sum(old_eff)
            
            # Acceptance
            if delta_obj > 0 or np.random.random() < math.exp(delta_obj / T):
                # Accept
                current_residues[p_idx] = new_r
                counts = new_counts
                construct._counts = counts
                current_obj += delta_obj
                
                # Check score periodically (expensive)
                # Score calculation is O(MAX_CHECK * n_primes) if done naively,
                # but we have counts.
                # Score is first index where counts < 2.
                # We can update score incrementally?
                # If we only changed one column, the counts changed in a structured way.
                # But finding the first gap is O(MAX_CHECK).
                # Let's just compute it.
                
                # Optimization: Only compute full score every N steps or if objective improved significantly?
                # No, we want to track the best score.
                # Computing score from counts is O(MAX_CHECK).
                # 2000 ops is very fast.
                
                invalid = counts < 2
                if not np.any(invalid):
                    new_score = MAX_CHECK
                else:
                    new_score = int(np.argmax(invalid))
                    
                if new_score > best_score:
                    best_score = new_score
                    best_residues = current_residues.copy()
                    best_obj = current_obj
                    # Optional: reset temperature if we hit a new record?
                    # T = max(T, 10.0) 
                    
        # Cooling
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If stuck at low T, maybe restart with higher T?
            # Simple restart strategy:
            # If no improvement in last X steps, restart from best with higher T.
            # For now, just keep going.
            
    # Final check
    # Ensure we return the best found
    return [int(r) for r in best_residues]

# Ensure numpy random seed is set for reproducibility if needed
np.random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(len(res))
