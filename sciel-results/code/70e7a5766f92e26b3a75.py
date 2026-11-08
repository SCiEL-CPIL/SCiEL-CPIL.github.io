"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3(x):
    """
    Compute the upper bound on C3 for a ±1 sequence x of length N.
    C3 is the max absolute value of the autocorrelation at non-zero lags.
    """
    N = len(x)
    size = 2 * N
    x_pad = np.zeros(size)
    x_pad[:N] = x
    X = np.fft.fft(x_pad)
    autocorr = np.fft.ifft(X * np.conj(X)).real[:N]
    # autocorr[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k]
    return np.max(np.abs(autocorr[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    best_x = None
    best_score = np.inf
    
    # Try multiple random restarts with simulated annealing
    rng = np.random.RandomState(42)
    
    for restart in range(50):
        # Check time
        if time.time() > deadline:
            break
        
        # Initialize with random ±1 sequence
        x = rng.choice([-1, 1], size=N)
        current_score = _compute_c3(x)
        
        if current_score < best_score:
            best_x = x.copy()
            best_score = current_score
        
        # Simulated annealing
        T = 2.0
        T_min = 0.001
        cooling = 0.9998
        no_improve_count = 0
        max_no_improve = 50000
        
        while time.time() < deadline:
            # Check if we should break early
            if no_improve_count > max_no_improve:
                break
            
            # Pick a random position to flip
            idx = rng.randint(N)
            x[idx] *= -1
            
            new_score = _compute_c3(x)
            delta = new_score - current_score
            
            if delta < 0:
                current_score = new_score
                no_improve_count = 0
                if current_score < best_score:
                    best_x = x.copy()
                    best_score = current_score
            else:
                # Accept with probability exp(-delta / T)
                if rng.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    x[idx] *= -1  # revert
                    no_improve_count += 1
            
            T = max(T * cooling, T_min)
            
            # Occasionally try multi-flip moves
            if rng.random() < 0.1 and T > T_min * 10:
                # Try flipping 2 random positions
                i = rng.randint(N)
                j = rng.randint(N)
                if i != j:
                    x[i] *= -1
                    x[j] *= -1
                    multi_score = _compute_c3(x)
                    if multi_score < current_score:
                        current_score = multi_score
                        no_improve_count = 0
                        if current_score < best_score:
                            best_x = x.copy()
                            best_score = current_score
                    else:
                        x[i] *= -1
                        x[j] *= -1
    
    # Final local search on the best solution found
    if best_x is not None:
        x = best_x.copy()
        current_score = _compute_c3(x)
        
        # Single flip local search until convergence
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(N):
                if time.time() > deadline:
                    break
                x[idx] *= -1
                new_score = _compute_c3(x)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_x = x.copy()
                        best_score = current_score
                else:
                    x[idx] *= -1  # revert
        
        # Try pair flips
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(i + 1, N):
                    x[i] *= -1
                    x[j] *= -1
                    new_score = _compute_c3(x)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_x = x.copy()
                            best_score = current_score
                    else:
                        x[i] *= -1
                        x[j] *= -1
                    if improved:
                        break
                if improved:
                    break
    
    if best_x is None:
        # Fallback: return a random sequence
        rng = np.random.RandomState(0)
        best_x = rng.choice([-1, 1], size=N)
    
    return best_x
# EVOLVE-BLOCK-END
