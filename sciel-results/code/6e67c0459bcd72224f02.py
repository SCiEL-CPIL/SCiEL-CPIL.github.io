import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _calc_max_autoconv(x):
    """
    Calculate the maximum absolute value of the aperiodic autocorrelation.
    x: numpy array of length N with values 1 or -1.
    """
    # Zero-pad to next power of 2 for efficient FFT
    # N=400, 2*400 - 1 = 799. Next power of 2 is 1024.
    n_fft = 1024
    
    # Compute FFT
    X = np.fft.fft(x, n=n_fft)
    # Power spectrum
    P = X * np.conj(X)
    # Inverse FFT gives circular autocorrelation
    r = np.fft.ifft(P).real
    
    # Extract lags 1 to N-1
    # r[0] is the energy N.
    # r[1:N] are A_1 to A_{N-1}
    return np.max(np.abs(r[1:N]))

def construct():
    start_time = time.time()
    deadline = 145.0 # Keep 5s margin before 150s limit
    
    # --- Initialization: Thue-Morse Sequence ---
    # The Thue-Morse sequence is known for having low autocorrelation properties.
    # We generate it using the property t(n) = popcount(n) % 2.
    # Map bits to {-1, 1}: 0 -> 1, 1 -> -1 (or vice versa, symmetry holds)
    indices = np.arange(N)
    # Calculate popcount for each index. 
    # Efficient way: use bin(x).count('1') or bit manipulation.
    # For N=400, a simple loop or vectorized trick is fine.
    # Vectorized popcount for small N is easy with bit operations.
    # Let's use a simple loop for clarity and speed is negligible for N=400.
    tm_bits = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int8)
    # Map to {-1, 1}
    x = 1 - 2 * tm_bits # 0->1, 1->-1
    
    best_x = x.copy()
    best_val = _calc_max_autoconv(best_x)
    
    current_x = best_x.copy()
    current_val = best_val
    
    # --- Optimization: Simulated Annealing ---
    # We will refine the Thue-Morse sequence using SA.
    # Objective: Minimize max |autocorr|
    
    # SA Parameters
    T = 10.0 # Initial temperature
    T_min = 0.01
    alpha = 0.999 # Cooling rate
    # We adjust alpha based on time budget roughly, but fixed is safer for structure
    
    no_improve_count = 0
    patience = 5000 # Number of steps without improvement before reducing T more aggressively
    
    while time.time() - start_time < deadline:
        # Generate a move
        # We try single bit flips primarily, but occasionally block flips
        # Single bit flip is faster to evaluate (can use incremental update, but FFT is fast enough)
        
        i = np.random.randint(N)
        
        # Apply move
        current_x[i] *= -1
        
        # Evaluate new state
        new_val = _calc_max_autoconv(current_x)
        
        # Acceptance criterion
        if new_val < current_val:
            # Improvement
            current_val = new_val
            no_improve_count = 0
            if new_val < best_val:
                best_val = new_val
                best_x = current_x.copy()
        else:
            # Deterioration or same
            # Accept with probability exp(-delta / T)
            delta = new_val - current_val
            if delta == 0:
                # Accept neutral moves to explore
                current_val = new_val
            else:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current_val = new_val
                else:
                    # Reject move
                    current_x[i] *= -1 # Revert
            
            no_improve_count += 1
            
            # Adaptive cooling / Patience
            if no_improve_count > patience:
                T *= 0.5 # Reduce temperature faster if stuck
                no_improve_count = 0
                # Optional: Random restart if T is very low? 
                # Let's just keep cooling.
        
        # Cool down
        T = max(T * alpha, T_min)
        
        # If T is too low and we are stuck for a long time, we might consider a random restart
        # but given the deadline, continuous SA is usually robust.
        # However, if T hits T_min and no improvement for a very long time, we can try a random jump.
        if T == T_min and no_improve_count > 10000:
             # Random restart to escape local minima
             # Keep best_x, but reset current_x to random or best_x?
             # Let's reset to best_x and increase T slightly to escape.
             current_x = best_x.copy()
             current_val = best_val
             T = 5.0
             no_improve_count = 0

    return best_x
# EVOLVE-BLOCK-END
