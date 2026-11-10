import numpy as np
import time
import random
from math import exp, log

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 24
    S_search = 4096  # Power of 2 for fast FFT-like operations, good resolution
    
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
    
    # Helper to evaluate peak from P_vals
    def get_peak(P_vals):
        return np.max(np.abs(P_vals))
    
    # Vectorized scoring: given current P_vals and seq, score all 24 single flips at once
    # For flip at index i: new_P = P - 2*c_i*z^i, so delta = -2*c_i*z^i
    # We can compute |P - 2*c_i*z^i| for all i simultaneously
    def score_all_flips(P_vals, seq, z_powers):
        """Score all n single-flip moves. Returns array of n peaks."""
        # delta_vals[i] = -2 * seq[i] * z_powers[i]  shape (n, S)
        delta_vals = -2.0 * np.array(seq)[:, None] * z_powers  # (n, S)
        # new_P[i] = P_vals + delta_vals[i]  shape (n, S)
        new_P = P_vals[None, :] + delta_vals  # (n, S)
        # peak for each flip
        peaks = np.max(np.abs(new_P), axis=1)  # (n,)
        return peaks
    
    # Vectorized pair-flip scoring: for all pairs (i,j), compute peak
    # This is O(n^2 * S) but done in one vectorized op
    def score_pair_flips(P_vals, seq, z_powers):
        """Score all pair-flip moves. Returns array of n*(n-1)/2 peaks."""
        # For flips at i and j: delta = -2*c_i*z^i - 2*c_j*z^j
        # We can compute all pairs
        n_local = len(seq)
        # delta_single[i] = -2*c_i*z^i
        delta_single = -2.0 * np.array(seq)[:, None] * z_powers  # (n, S)
        
        # For pair (i,j): new_P = P + delta_single[i] + delta_single[j]
        # We want max over all (i,j) with i < j
        # Use broadcasting: delta_single[:, None, :] + delta_single[None, :, :] gives (n, n, S)
        # But that's n^2 * S which for n=24, S=4096 is 24*24*4096 = ~2.4M complex = ~38MB
        # That's fine!
        pair_deltas = delta_single[:, None, :] + delta_single[None, :, :]  # (n, n, S)
        new_P_pairs = P_vals[None, None, :] + pair_deltas  # (n, n, S)
        peaks_pairs = np.max(np.abs(new_P_pairs), axis=2)  # (n, n)
        return peaks_pairs
    
    # Main search: use vectorized scoring for speed
    best_peak = np.inf
    best_seq_copy = list(best_seq)
    
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Diversified restart
        if restart == 0:
            seq = list(best_seq_copy)
        elif restart < 3:
            seq = list(best_seq_copy)
            num_flips = random.randint(2, 4)
            for _ in range(num_flips):
                idx = random.randint(0, n - 1)
                seq[idx] = -seq[idx]
        elif restart < 6:
            seq = list(best_seq_copy)
            num_flips = random.randint(4, 8)
            for _ in range(num_flips):
                idx = random.randint(0, n - 1)
                seq[idx] = -seq[idx]
        else:
            seq = [random.choice([1, -1]) for _ in range(n)]
        
        P_vals = np.sum(np.array(seq)[:, None] * z_powers_search, axis=0)
        current_peak = get_peak(P_vals)
        local_best_peak = current_peak
        local_best_seq = list(seq)
        
        # Simulated annealing with vectorized scoring
        T = 1.0
        T_min = 1e-5
        alpha = 0.9995
        
        while time.time() < deadline - 8 and T > T_min:
            # Score all single flips at once
            peaks = score_all_flips(P_vals, seq, z_powers_search)
            
            # Find the best flip
            best_flip_idx = int(np.argmin(peaks))
            best_flip_peak = peaks[best_flip_idx]
            
            if best_flip_peak < current_peak:
                # Accept the best flip
                idx = best_flip_idx
                c = seq[idx]
                delta_val = -2 * c * z_powers_search[idx]
                P_vals = P_vals + delta_val
                current_peak = best_flip_peak
                seq[idx] = -seq[idx]
                
                if current_peak < local_best_peak:
                    local_best_peak = current_peak
                    local_best_seq = list(seq)
            else:
                # Metropolis: accept with probability exp(-delta/T)
                # Pick a random flip (not just the best) for exploration
                idx = random.randint(0, n - 1)
                c = seq[idx]
                delta_val = -2 * c * z_powers_search[idx]
                new_P_vals = P_vals + delta_val
                new_peak = get_peak(new_P_vals)
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
    
    # Final high-precision polish with more sampling points
    S_polish = 8192
    angles_polish = 2 * np.pi * np.arange(S_polish) / S_polish
    z_powers_polish = np.zeros((n, S_polish), dtype=complex)
    for k in range(n):
        z_powers_polish[k] = np.exp(1j * k * angles_polish)
    
    seq = list(best_seq_copy)
    P_vals_polish = np.sum(np.array(seq)[:, None] * z_powers_polish, axis=0)
    current_peak_polish = get_peak(P_vals_polish)
    best_peak_polish = current_peak_polish
    best_seq_polish = list(seq)
    
    # Tight local search with vectorized scoring
    T = 0.03
    T_min_polish = 1e-6
    alpha_polish = 0.9998
    
    while time.time() < deadline - 3 and T > T_min_polish:
        # Score all single flips
        peaks = score_all_flips(P_vals_polish, seq, z_powers_polish)
        best_flip_idx = int(np.argmin(peaks))
        best_flip_peak = peaks[best_flip_idx]
        
        if best_flip_peak < current_peak_polish:
            idx = best_flip_idx
            c = seq[idx]
            delta_val = -2 * c * z_powers_polish[idx]
            P_vals_polish = P_vals_polish + delta_val
            current_peak_polish = best_flip_peak
            seq[idx] = -seq[idx]
            
            if current_peak_polish < best_peak_polish:
                best_peak_polish = current_peak_polish
                best_seq_polish = list(seq)
        else:
            # Try a random flip for exploration
            idx = random.randint(0, n - 1)
            c = seq[idx]
            delta_val = -2 * c * z_powers_polish[idx]
            new_P_vals = P_vals_polish + delta_val
            new_peak = get_peak(new_P_vals)
            delta_peak = new_peak - current_peak_polish
            if random.random() < exp(-delta_peak / T):
                P_vals_polish = new_P_vals
                current_peak_polish = new_peak
                seq[idx] = -seq[idx]
        
        T *= alpha_polish
    
    # Try pair flips in final polish
    while time.time() < deadline - 1:
        peaks_pairs = score_pair_flips(P_vals_polish, seq, z_powers_polish)
        # Find best pair
        best_i, best_j = np.unravel_index(np.argmin(peaks_pairs), peaks_pairs.shape)
        best_pair_peak = peaks_pairs[best_i, best_j]
        
        if best_pair_peak < current_peak_polish:
            # Apply both flips
            for idx in [best_i, best_j]:
                c = seq[idx]
                delta_val = -2 * c * z_powers_polish[idx]
                P_vals_polish = P_vals_polish + delta_val
                seq[idx] = -seq[idx]
            current_peak_polish = best_pair_peak
            
            if current_peak_polish < best_peak_polish:
                best_peak_polish = current_peak_polish
                best_seq_polish = list(seq)
        else:
            break
    
    return best_seq_polish
