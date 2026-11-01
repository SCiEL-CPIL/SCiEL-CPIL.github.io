import numpy as np
import time
import sys
import random

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def compute_c3(h):
        """
        Compute the upper bound C3.
        Metric: Peak Sidelobe Level (PSL) normalized by Main Lobe.
        h: array of length N, values in {-1, 1}.
        """
        h_float = h.astype(float)
        # Compute linear autocorrelation
        # Using FFT
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h_float, n=n_fft)
        R_full = np.fft.irfft(F * np.conj(F), n=n_fft)
        R_pos = R_full[:N]
        
        peak = R_pos[0]
        if peak < 1e-10:
            return 1e10
        sidelobes = R_pos[1:]
        max_side = np.max(np.abs(sidelobes))
        return max_side / peak

    best_h = None
    best_val = float('inf')
    
    # Simulated Annealing
    def simulated_annealing(h_init, max_iters, start_T, end_T):
        h = h_init.copy()
        current_val = compute_c3(h)
        best_h_local = h.copy()
        best_val_local = current_val
        
        # Precompute some constants?
        # We will just recompute for simplicity, N=400 is small enough for SA steps
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Cooling schedule
            T = start_T * (end_T / start_T) ** (i / max_iters)
            
            # Pick a random index to flip
            idx = np.random.randint(N)
            h[idx] *= -1
            
            new_val = compute_c3(h)
            
            if new_val < current_val:
                current_val = new_val
                if new_val < best_val_local:
                    best_val_local = new_val
                    best_h_local = h.copy()
            else:
                # Accept with probability exp(-delta/T)
                delta = new_val - current_val
                if delta < 0:
                    # This shouldn't happen if we check new_val < current_val
                    pass
                else:
                    prob = np.exp(-delta / T) if T > 0 else 0
                    if np.random.rand() < prob:
                        current_val = new_val
                    else:
                        h[idx] *= -1 # Revert
            
        return best_h_local, best_val_local

    # Local Search (Hill Climbing)
    def local_search(h_init, max_iters):
        h = h_init.copy()
        current_val = compute_c3(h)
        best_h_local = h.copy()
        best_val_local = current_val
        
        for i in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            # Try flipping a few random bits
            for _ in range(10):
                idx = np.random.randint(N)
                h[idx] *= -1
                new_val = compute_c3(h)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if new_val < best_val_local:
                        best_val_local = new_val
                        best_h_local = h.copy()
                else:
                    h[idx] *= -1 # Revert
            
            if not improved:
                # Stagnation, try a random jump or break
                # For now, just break or do a random restart internally?
                # Let's do a small random perturbation
                for _ in range(5):
                    idx = np.random.randint(N)
                    h[idx] *= -1
                current_val = compute_c3(h)
                if current_val < best_val_local:
                    best_val_local = current_val
                    best_h_local = h.copy()
                # If still not improved after perturbation, break
                if compute_c3(h) >= best_val_local:
                    break

        return best_h_local, best_val_local

    # Initialization strategies
    def init_random():
        return np.random.choice([-1, 1], size=N)
    
    def init_chaos():
        # Logistic map
        x = 0.5
        h = np.zeros(N)
        for i in range(N):
            x = 3.9 * x * (1 - x)
            h[i] = 1 if x > 0.5 else -1
        return h

    # Run multiple starts
    n_starts = 20
    for i in range(n_starts):
        if time.time() > deadline - 10.0:
            break
            
        if i % 2 == 0:
            h_init = init_random()
        else:
            h_init = init_chaos()
        
        # Alternate between SA and Local Search
        if i % 4 == 0:
            # SA with longer run
            max_iters = 5000
            start_T = 1.0
            end_T = 0.01
            h_res, val_res = simulated_annealing(h_init, max_iters, start_T, end_T)
        elif i % 4 == 1:
            # Local Search
            max_iters = 2000
            h_res, val_res = local_search(h_init, max_iters)
        elif i % 4 == 2:
            # SA with shorter run, higher T
            max_iters = 3000
            start_T = 2.0
            end_T = 0.1
            h_res, val_res = simulated_annealing(h_init, max_iters, start_T, end_T)
        else:
            # Local Search with more iterations
            max_iters = 5000
            h_res, val_res = local_search(h_init, max_iters)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res.copy()
            
        # Update time check
        if time.time() > deadline - 15.0:
            break

    if best_h is None:
        best_h = init_random()
        
    return best_h
