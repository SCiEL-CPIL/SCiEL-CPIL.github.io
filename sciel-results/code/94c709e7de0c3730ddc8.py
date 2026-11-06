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
    # The task asks for L >= 1000. 
    # We check up to 1500 to allow the annealing to push the frontier past 1000.
    # If the first gap is at 1501, the score is 1500.
    MAX_CHECK = 1500
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p for all x and p.
    # Shape: (MAX_CHECK, n_primes)
    # x_mod_p[i, j] = x_values[i] % primes[j]
    # This allows fast vectorized comparison with residues.
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    def compute_score(residues):
        """
        Computes the largest L such that every integer 1..L satisfies at least two congruences.
        residues: numpy array of shape (n_primes,)
        Returns: integer L
        """
        # Compare x_mod_p (MAX_CHECK, n_primes) with residues (n_primes,)
        # match[i, j] is True if x_values[i] % primes[j] == residues[j]
        match = (x_mod_p == residues[None, :])
        
        # Count matches for each x
        counts = match.sum(axis=1)
        
        # Find the first x (index i) where counts[i] < 2
        # We want the largest L such that counts[0:L] are all >= 2.
        # This is equivalent to finding the first index where count < 2.
        # If all are >= 2, L = MAX_CHECK.
        
        # np.argmax returns 0 if no element is True. 
        # We need to handle the case where no element is < 2.
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            # The gap is at x_values[first_invalid_idx]
            # So the valid prefix length is first_invalid_idx (since x_values starts at 1, index 0 is x=1)
            # Wait, x_values[0] is 1.
            # If x_values[0] is invalid, L=0.
            # If x_values[1] is invalid, L=1.
            # So L = first_invalid_idx.
            return int(first_invalid_idx)

    def get_residue_array(res_list):
        return np.array(res_list, dtype=np.int64)

    # Initial Solution
    # Use a heuristic initialization: r_p = 1 for all? Or random?
    # Let's try r_p = 1.
    # Or better, use the "seed" logic from the previous program if it was good, 
    # but since we are switching strategies, let's start with a strong random or heuristic.
    # A common good start for covering systems is often near 0 or 1.
    # Let's initialize with r_p = 1 for all primes.
    current_residues = np.ones(n_primes, dtype=np.int64)
    
    # Calculate initial score
    current_score = compute_score(current_residues)
    
    best_residues = current_residues.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    # Temperature starts high to allow large moves, cools down.
    T_start = 50.0
    T_end = 0.01
    cooling_rate = 0.99
    
    # We will iterate until time runs out.
    T = T_start
    steps = 0
    
    while time.time() < deadline:
        # Check time frequently to avoid overshoot
        if time.time() > deadline - 0.5:
            break
            
        # Select a random prime index to change
        idx = np.random.randint(0, n_primes)
        p_val = primes[idx]
        
        # Select a new random residue for this prime
        # To make moves more effective, we might bias towards residues that cover the current gap?
        # But standard SA uses random moves. 
        # Let's try a "smart" random move: 
        # 50% chance: completely random residue.
        # 50% chance: residue that covers the current first gap (if known).
        
        if np.random.random() < 0.5:
            new_r = np.random.randint(0, p_val)
        else:
            # Find the current gap
            if best_score < MAX_CHECK:
                gap_x = best_score + 1
                new_r = gap_x % p_val
            else:
                new_r = np.random.randint(0, p_val)
                
        old_r = current_residues[idx]
        
        if new_r == old_r:
            continue
            
        # Apply move
        current_residues[idx] = new_r
        new_score = compute_score(current_residues)
        
        # Acceptance Criterion
        if new_score > current_score:
            # Better solution
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues.copy()
        else:
            # Worse or equal solution
            # Accept with probability exp((new_score - current_score) / T)
            # Note: We are maximizing, so delta = new - current is negative.
            # Probability = exp(delta / T)
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if np.random.random() < prob:
                    current_score = new_score
                else:
                    # Reject move
                    current_residues[idx] = old_r
            # If delta == 0, accept (prob=1)
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        steps += 1
        
        # Periodic restart? 
        # If stuck for a long time, we might want to restart from best with high T?
        # SA usually handles this. Let's stick to simple cooling.
        # However, if T is very low and we are stuck, we might increase T slightly (reheating)
        # or just let it run. Given the 120s limit, simple cooling is safer.
        
        # Optimization: If we reach MAX_CHECK, we can stop early?
        # The task says "Maximize; L >= 1000". 
        # If we reach 1500, we have likely maximized for the check range.
        # But we should keep searching to see if we can push it further? 
        # The score is capped at 10^5, but our check range is 1500.
        # If we hit 1500, we should extend the check range?
        # That requires rebuilding x_mod_p, which is expensive.
        # Let's just return if we hit MAX_CHECK, assuming 1500 is sufficient for a high score.
        # Actually, the problem asks for L >= 1000. 
        # If we get 1500, that's a very good score.
        if best_score >= MAX_CHECK:
            # We can try to extend, but it's risky with time.
            # Let's just break and return.
            break

    # Convert back to list of ints
    return [int(r) for r in best_residues]

# Ensure numpy random seed is set for reproducibility if needed, 
# but SA is stochastic.
np.random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(len(res))
