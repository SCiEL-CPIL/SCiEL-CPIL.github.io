"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # New Approach: Deterministic Chirp Initialization + Hill Climbing on Peak Sidelobe Level (PSL)
    # Rationale: 
    # 1. Chirp sequences have inherently low autocorrelation sidelobes.
    # 2. Minimizing PSL (max sidelobe) is often more critical for "upper bound" metrics than Energy.
    # 3. Hill Climbing is computationally cheaper than SA, allowing more refinements.
    
    # 1. Construct a Binary Chirp Sequence
    # s[k] = sign(cos(pi * k^2 / N))
    k = np.arange(N)
    phase = np.pi * (k ** 2) / N
    f = np.sign(np.cos(phase))
    # Ensure no zeros (if cos is 0, sign is 0, which is invalid for +/-1 step function usually, or need to handle)
    # If any 0s, replace with 1
    f[f == 0] = 1
    
    def calculate_psl(seq):
        """
        Calculate the Peak Sidelobe Level of the aperiodic autocorrelation.
        PSL = max_{k != 0} |R(k)|
        """
        ac = np.correlate(seq, seq, mode='full')
        # ac[N-1] is R(0) = N
        # Sidelobes are all other elements
        # We take the max absolute value of the array excluding the center
        # Center index is N-1
        # Create a mask or just slice
        # ac has length 2N-1
        # Left side: ac[0:N-1]
        # Right side: ac[N:2N-1]
        left = ac[:N-1]
        right = ac[N:]
        max_left = np.max(np.abs(left))
        max_right = np.max(np.abs(right))
        return max(max_left, max_right)

    current_psl = calculate_psl(f)
    best_f = f.copy()
    best_psl = current_psl
    
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    # Hill Climbing Parameters
    # We flip random bits. If PSL decreases, we keep it.
    # To avoid getting stuck in local minima, we can occasionally allow small increases (Simulated Annealing lite)
    # or just stick to strict Hill Climbing with random restarts if stuck.
    # Given the time limit, strict Hill Climbing with a good start is likely strong.
    # Let's add a small "stall" counter to trigger a random perturbation if stuck.
    
    stall_count = 0
    max_stall = 500
    
    while (time.time() - start_time) < max_time:
        # Flip a random bit
        i = np.random.randint(N)
        old_val = f[i]
        f[i] = -old_val
        
        new_psl = calculate_psl(f)
        
        if new_psl < current_psl:
            current_psl = new_psl
            if current_psl < best_psl:
                best_psl = current_psl
                best_f = f.copy()
            stall_count = 0
        else:
            # Revert
            f[i] = old_val
            stall_count += 1
            
            # If stuck for too long, try a random mutation of a few bits to escape local minimum
            if stall_count > max_stall:
                # Perturb 5 random bits
                for _ in range(5):
                    j = np.random.randint(N)
                    f[j] = -f[j]
                current_psl = calculate_psl(f)
                # Update best if perturbation helped (unlikely but possible)
                if current_psl < best_psl:
                    best_psl = current_psl
                    best_f = f.copy()
                stall_count = 0
                max_stall = int(max_stall * 1.5) # Increase patience

    return best_f
# EVOLVE-BLOCK-END
