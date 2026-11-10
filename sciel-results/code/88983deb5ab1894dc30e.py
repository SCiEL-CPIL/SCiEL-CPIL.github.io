import numpy as np
import time
import random
from math import exp, log

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 24
    S_search = 4000  # Sampling points for search
    S_final = 20000  # Sampling points for final polish
    
    # Precompute z powers for fast delta evaluation
    angles_search = 2 * np.pi * np.arange(S_search) / S_search
    z_powers_search = np.zeros((n, S_search), dtype=complex)
    for k in range(n):
        z_powers_search[k] = np.exp(1j * k * angles_search)
    
    # Initial sequence: Golay-Rudin-Shapiro
    def gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    best_seq = list(gr_sequence(n))
    
    # Compute initial P values
    P_vals = np.sum(np.array(best_seq)[:, None] * z_powers_search, axis=0)
    current_peak = np.max(np.abs(P_vals))
    best_peak = current_peak
    best_seq_copy = list(best_seq)
    
    # Helper to evaluate peak
    def get_peak(P_vals):
        return np.max(np.abs(P_vals))
    
    # Simulated Annealing with multiple restarts
    T_max = 2.0
    T_min = 1e-5
    alpha = 0.9995
    
    # Run multiple SA restarts to explore the space better
    num_restarts = 12
    restart_budget = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        # Reset to best found so far or a new random start
        if restart == 0:
            seq = list(best_seq_copy)
        else:
            # Start from best found so far with some random perturbations
            seq = list(best_seq_copy)
            # Increase perturbation for later restarts to explore more
            num_flips = random.randint(3, 8) + (restart // 4)
            for _ in range(num_flips):
                idx = random.randint(0, n - 1)
                seq[idx] = -seq[idx]
        
        # Recompute P_vals for this sequence
        P_vals = np.sum(np.array(seq)[:, None] * z_powers_search, axis=0)
        current_peak = get_peak(P_vals)
        
        T = T_max
        local_best_peak = current_peak
        local_best_seq = list(seq)
        
        while time.time() < deadline - 5 and T > T_min:
            # Pick a random index to flip
            idx = random.randint(0, n - 1)
            
            # Calculate delta P_vals
            c = seq[idx]
            delta_val = -2 * c * z_powers_search[idx]
            
            new_P_vals = P_vals + delta_val
            new_peak = get_peak(new_P_vals)
            
            if new_peak < current_peak:
                # Accept
                P_vals = new_P_vals
                current_peak = new_peak
                seq[idx] = -seq[idx]
                
                if current_peak < local_best_peak:
                    local_best_peak = current_peak
                    local_best_seq = list(seq)
            else:
                # Metropolis criterion
                delta_peak = new_peak - current_peak
                if random.random() < exp(-delta_peak / T):
                    P_vals = new_P_vals
                    current_peak = new_peak
                    seq[idx] = -seq[idx]
            
            T *= alpha
        
        # Update global best
        if local_best_peak < best_peak:
            best_peak = local_best_peak
            best_seq_copy = list(local_best_seq)
    
    # Final high-precision polish pass
    # Use more sampling points and a tighter local search
    S_polish = 10000
    angles_polish = 2 * np.pi * np.arange(S_polish) / S_polish
    z_powers_polish = np.zeros((n, S_polish), dtype=complex)
    for k in range(n):
        z_powers_polish[k] = np.exp(1j * k * angles_polish)
    
    seq = list(best_seq_copy)
    P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
    current_peak_polish = get_peak(P_vals_polish)
    best_peak_polish = current_peak_polish
    best_seq_polish = list(seq)
    
    # Tight local search with smaller temperature
    T = 0.1
    T_min_polish = 1e-6
    alpha_polish = 0.999
    
    while time.time() < deadline - 2 and T > T_min_polish:
        idx = random.randint(0, n - 1)
        c = seq[idx]
        delta_val = -2 * c * z_powers_polish[idx]
        
        new_P_vals = P_vals_polish + delta_val
        new_peak = get_peak(new_P_vals)
        
        if new_peak < current_peak_polish:
            P_vals_polish = new_P_vals
            current_peak_polish = new_peak
            seq[idx] = -seq[idx]
            
            if current_peak_polish < best_peak_polish:
                best_peak_polish = current_peak_polish
                best_seq_polish = list(seq)
        else:
            delta_peak = new_peak - current_peak_polish
            if random.random() < exp(-delta_peak / T):
                P_vals_polish = new_P_vals
                current_peak_polish = new_peak
                seq[idx] = -seq[idx]
        
        T *= alpha_polish
    
    # Return the best sequence found
    return best_seq_polish
