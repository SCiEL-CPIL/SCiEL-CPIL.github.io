"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def construct():
    """
    Construct a signed step sequence minimizing the maximum absolute autoconvolution
    divided by the squared integral.
    
    Strategy:
    1. Initialize with a random sign pattern (or a structured one).
    2. Use local search (single-flip and maybe double-flip) to minimize the objective.
    3. The objective is max(|autocorr|) / (sum(x))^2.
       Note: if sum(x) is 0, the denominator is 0, which is problematic. 
       The task says "divided by squared integral". If integral is 0, it's undefined.
       Usually, these tasks assume non-zero integral or handle it.
       Let's check the seed: np.ones(N) -> sum=N, autocorr is triangular.
       Max abs autocorr for ones: N (at lag 0). Sum^2 = N^2. Ratio = 1/N.
       
       We want to minimize this ratio.
       
       Objective: J = max_k |C_k| / (sum x_i)^2
       
       We can scale x_i by +/-1. The sum changes sign/magnitude.
       If sum is close to 0, the ratio blows up. So we want large |sum| and small max autocorr.
       
       Local search:
       - Compute autocorr C = x * x (full convolution).
       - Current max_abs = max(|C|).
       - Current sum_sq = (sum(x))^2.
       - Current score = max_abs / sum_sq.
       
       Flip a bit x_i -> -x_i.
       New sum = sum - 2*x_i.
       New autocorr: C' = C - 2*x_i * (x * delta_i) ?
       Actually, C_k = sum_j x_j x_{j+k}.
       If we flip x_i to -x_i:
       C'_k = C_k - 2*x_i * (x_{i+k} + x_{i-k} - x_i*delta_{k0}?) 
       Let's derive:
       C_k = sum_j x_j x_{j+k}.
       Term involving x_i: when j=i, term is x_i x_{i+k}. When j+k=i, j=i-k, term is x_{i-k} x_i.
       If k=0, the term is x_i^2 = 1. It appears once in the sum? No, sum_j x_j x_{j}. j runs 0..N-1.
       For k=0, C_0 = sum x_j^2 = N.
       When flipping x_i, x_i^2 stays 1.
       For k != 0:
       The terms are x_i x_{i+k} and x_{i-k} x_i.
       New terms: (-x_i) x_{i+k} and x_{i-k} (-x_i).
       Change in C_k: -2 x_i x_{i+k} - 2 x_{i-k} x_i.
       So C'_k = C_k - 2 x_i (x_{i+k} + x_{i-k}).
       Note: indices must be valid.
       
       This update is O(N) for a single flip if we update the whole C array.
       But we only need to know if max(|C'|) decreases sufficiently to offset the change in denominator.
       
       However, updating the whole C array for each candidate flip is O(N^2) per pass if we check all N flips.
       N=400. N^2 = 160,000. That's fast.
       
       Let's implement a simple local search:
       1. Init x.
       2. Compute C.
       3. Loop:
          - For each i in 0..N-1:
            - Calculate new_score if we flip i.
            - Keep track of best flip.
          - If best flip improves score, apply it.
          - Else, break or random restart.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Objective function
    def get_score(x):
        # x is int array of +/-1
        s = np.sum(x)
        if s == 0:
            return np.inf
        # Autocorrelation
        # Using np.correlate or fftconvolve
        # np.correlate(x, x, mode='full') gives lags from -(N-1) to N-1
        # We want max absolute value.
        c = np.correlate(x, x, mode='full')
        max_abs = np.max(np.abs(c))
        return max_abs / (s * s)
    
    # Initialization: Random signs
    # Maybe start with a pattern that has high sum?
    # Or random.
    x = np.random.choice([-1, 1], size=N)
    
    # Ensure non-zero sum initially?
    # If sum is 0, score is inf. Local search will try to fix it.
    
    best_score = get_score(x)
    best_x = x.copy()
    
    # Local Search
    # To speed up, we can precompute C and update incrementally.
    # But N=400 is small. Let's just recompute or use the O(N) update for the candidate.
    
    # Let's use the O(N) update for C to evaluate flips quickly.
    # C = correlate(x, x, 'full')
    # C has length 2N-1. Index k corresponds to lag k - (N-1).
    # C[k] = sum_j x[j] x[j + (k - (N-1))]
    
    # For a flip at i:
    # Delta C[m] = -2 * x[i] * (x[i + lag] + x[i - lag])
    # where lag = m - (N-1).
    
    # Let's implement the search with incremental updates.
    
    def compute_C(x):
        return np.correlate(x, x, mode='full')
    
    C = compute_C(x)
    current_score = best_score
    
    # Precompute indices for faster access?
    # x is numpy array.
    
    improved = True
    no_improve_count = 0
    max_no_improve = 20
    
    while improved and time.time() < deadline:
        improved = False
        best_flip_idx = -1
        best_new_score = current_score
        best_new_C = None
        best_new_sum = None
        
        # Evaluate all flips
        for i in range(N):
            # Flip x[i]
            # New sum
            new_sum = np.sum(x) - 2 * x[i]
            if new_sum == 0:
                continue
            
            # New C
            # C_new = C - 2 * x[i] * (x shifted by lag + x shifted by -lag)
            # Let's compute the delta vector.
            # lag = m - (N-1)
            # term1: x[i + lag] -> x[i + m - (N-1)]
            # term2: x[i - lag] -> x[i - m + (N-1)]
            
            # We can compute this vectorized.
            # lags = np.arange(-(N-1), N)
            # This is expensive to do inside the loop for N=400?
            # 400 * 400 operations = 160k. Very fast.
            
            # Let's just recompute C for the candidate?
            # x_flip = x.copy()
            # x_flip[i] *= -1
            # C_new = compute_C(x_flip)
            # This is O(N log N) or O(N^2) depending on correlate impl.
            # np.correlate is O(N^2) for small N usually?
            # Let's try the incremental update.
            
            # lags = np.arange(-(N-1), N)
            # idx1 = i + lags
            # idx2 = i - lags
            # valid1 = (idx1 >= 0) & (idx1 < N)
            # valid2 = (idx2 >= 0) & (idx2 < N)
            # delta = -2 * x[i] * (np.where(valid1, x[idx1[valid1]], 0) + np.where(valid2, x[idx2[valid2]], 0))
            # This creates arrays.
            
            # Simpler: just compute the new C via convolution of the delta?
            # x_new = x - 2*x[i]*e_i
            # C_new = (x - 2*x[i]*e_i) * (x - 2*x[i]*e_i)
            # = C - 4*x[i]*(x*e_i) + 4*x[i]^2*(e_i*e_i)
            # = C - 4*x[i]*(x shifted) + 4*delta_{lag,0}
            # Wait, e_i * e_i is delta at lag 0.
            # x * e_i is x shifted such that the i-th element is at lag 0?
            # Correlation C[m] = <x, shift^m x>.
            # <x, shift^m e_i> = x[i-m].
            # <e_i, shift^m x> = x[i+m].
            # So C_new[m] = C[m] - 4*x[i]*(x[i-m] + x[i+m]) + 4*delta_{m,0}
            # Note: x[i]^2 = 1, so 4*1*delta.
            
            # Let's use this formula.
            # lags = np.arange(-(N-1), N)
            # This is constant. Precompute it.
            
            pass
        
        # Precompute lags outside loop
        lags = np.arange(-(N-1), N)
        idx_center = N - 1
        
        for i in range(N):
            new_sum = np.sum(x) - 2 * x[i]
            if new_sum == 0:
                continue
            
            # Calculate delta C
            # x[i-m] corresponds to x[i - lags]
            # x[i+m] corresponds to x[i + lags]
            
            idx_minus = i - lags
            idx_plus = i + lags
            
            valid_minus = (idx_minus >= 0) & (idx_minus < N)
            valid_plus = (idx_plus >= 0) & (idx_plus < N)
            
            # Initialize delta terms to 0
            term_minus = np.zeros(N, dtype=int) # Wait, C has length 2N-1
            # lags has length 2N-1
            term_minus = np.zeros(2*N-1, dtype=int)
            term_plus = np.zeros(2*N-1, dtype=int)
            
            term_minus[valid_minus] = x[idx_minus[valid_minus]]
            term_plus[valid_plus] = x[idx_plus[valid_plus]]
            
            delta_C = -4 * x[i] * (term_minus + term_plus)
            # Add the 4*delta_{m,0} term
            delta_C[idx_center] += 4
            
            C_new = C + delta_C
            
            max_abs = np.max(np.abs(C_new))
            new_score = max_abs / (new_sum * new_sum)
            
            if new_score < best_new_score:
                best_new_score = new_score
                best_flip_idx = i
                best_new_C = C_new
                best_new_sum = new_sum
        
        if best_flip_idx != -1:
            # Apply flip
            x[best_flip_idx] *= -1
            C = best_new_C
            current_score = best_new_score
            improved = True
            no_improve_count = 0
            if current_score < best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Random restart or perturb
                # Perturb: flip a few random bits
                num_flips = np.random.randint(1, 5)
                for _ in range(num_flips):
                    idx = np.random.randint(N)
                    x[idx] *= -1
                C = compute_C(x)
                current_score = get_score(x)
                no_improve_count = 0
                # Check if better
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                # Maybe just break to save time? Or continue?
                # If we can't improve, we are in a local min.
                # Let's try a larger perturbation.
                # Flip random half?
                # For now, just break if no improvement for a while.
                pass

    # Return the best sequence found
    return best_x

# EVOLVE-BLOCK-END
