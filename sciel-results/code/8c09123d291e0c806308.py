"""Optimal ±1 polynomials with 24 coefficients minimizing sup norm.

Strategy: Multi-Start Batch Local Search with Vectorized Evaluation.
1. Initialize a batch of random candidates.
2. Use a dense DFT grid to evaluate all candidates in parallel.
3. Select the best candidates and perform local search by evaluating all 
   single-bit flip neighbors in parallel batches.
4. Refine the top candidates with a higher-resolution grid and golden-section 
   peak finding to achieve high accuracy.
"""
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 24
    sqrtN = math.sqrt(N)
    
    # Grid for main search
    M_main = 4096
    # Grid for refinement
    M_refine = 16384
    
    # Precompute DFT matrices
    # W[j, k] = exp(i * 2*pi * j * k / M)
    angles_main = 2 * np.pi * np.arange(M_main) / M_main
    W_main = np.exp(1j * np.outer(angles_main, np.arange(N)))
    
    angles_refine = 2 * np.pi * np.arange(M_refine) / M_refine
    W_refine = np.exp(1j * np.outer(angles_refine, np.arange(N)))
    
    def eval_batch(coeffs_batch):
        """
        coeffs_batch: shape (B, N) of +/-1 floats
        Returns: shape (B,) array of max |P(z)| / sqrt(N)
        """
        # P(z_j) = sum_k c_k z_j^k
        # vals = W @ coeffs_batch.T  -> shape (M, B)
        vals = W_main @ coeffs_batch.T
        max_vals = np.max(np.abs(vals), axis=0)
        return max_vals / sqrtN

    def refine_peak(c, n_peaks=5, grid_size=M_refine, golden_iters=40):
        """
        Refine the sup norm for a single candidate c using a finer grid 
        and golden section search on the top peaks.
        Returns the refined max |P(z)| / sqrt(N).
        """
        angles = 2 * np.pi * np.arange(grid_size) / grid_size
        zs = np.exp(1j * angles)
        
        # Evaluate on fine grid
        vals = W_refine @ c
        abs_vals = np.abs(vals)
        
        # Find top peaks
        top_indices = np.argsort(abs_vals)[-n_peaks:][::-1]
        
        max_peak = 0.0
        gr = (math.sqrt(5) - 1) / 2
        
        for idx in top_indices:
            theta_c = angles[idx]
            window = 2 * np.pi / grid_size
            a = theta_c - window
            b = theta_c + window
            
            def f(theta):
                z = np.exp(1j * theta)
                # Horner's method
                val = c[N-1]
                for k in range(N-2, -1, -1):
                    val = val * z + c[k]
                return abs(val)
            
            c_pt = b - gr * (b - a)
            d_pt = a + gr * (b - a)
            fc = f(c_pt)
            fd = f(d_pt)
            
            for _ in range(golden_iters):
                if fc < fd:
                    b, d_pt, fd = d_pt, c_pt, fc
                    c_pt = b - gr * (b - a)
                    fc = f(c_pt)
                else:
                    a, c_pt, fc = c_pt, d_pt, fd
                    d_pt = a + gr * (b - a)
                    fd = f(d_pt)
            
            max_peak = max(max_peak, fc, fd)
        
        return max_peak / sqrtN

    # Phase 1: Random Initialization and Batch Evaluation
    n_init = 1024
    # Generate random +/-1 vectors
    # Use 0/1 then map to -1/1
    random_bits = np.random.randint(0, 2, size=(n_init, N))
    init_coeffs = 2.0 * random_bits - 1.0
    
    scores = eval_batch(init_coeffs)
    
    # Phase 2: Select best candidates and perform Local Search
    # Sort by score (lower is better)
    sorted_indices = np.argsort(scores)
    n_local = 64
    local_indices = sorted_indices[:n_local]
    local_coeffs = init_coeffs[local_indices].copy()
    local_scores = scores[local_indices].copy()
    
    # Local Search Loop
    improved = True
    while improved and time.time() < deadline - 30.0:
        improved = False
        
        # For each local candidate, evaluate all 24 neighbors
        # We can do this in batches of candidates? 
        # Let's process one candidate at a time but vectorize the neighbor evaluation
        for i in range(n_local):
            if time.time() > deadline - 30.0:
                break
                
            c = local_coeffs[i]
            cur_score = local_scores[i]
            
            # Generate all 24 neighbors
            neighbors = np.array([c.copy() for _ in range(N)])
            for k in range(N):
                neighbors[k][k] = -c[k]
            
            # Evaluate neighbors in batch
            neighbor_scores = eval_batch(neighbors)
            
            # Find the best neighbor
            best_neighbor_idx = np.argmin(neighbor_scores)
            best_neighbor_score = neighbor_scores[best_neighbor_idx]
            
            if best_neighbor_score < cur_score:
                local_coeffs[i] = neighbors[best_neighbor_idx]
                local_scores[i] = best_neighbor_score
                improved = True
    
    # Phase 3: Refine the best candidates
    # Select top 10 for high-accuracy refinement
    n_refine = 10
    final_sorted_indices = np.argsort(local_scores)
    refine_indices = final_sorted_indices[:n_refine]
    
    best_score = float('inf')
    best_coeffs = None
    
    for idx in refine_indices:
        if time.time() > deadline - 5.0:
            break
            
        c = local_coeffs[idx]
        refined_score = refine_peak(c)
        
        if refined_score < best_score:
            best_score = refined_score
            best_coeffs = c.copy()
    
    # Fallback if no refinement happened or failed
    if best_coeffs is None:
        # Return the best from local search
        best_idx = np.argmin(local_scores)
        best_coeffs = local_coeffs[best_idx]
    
    # Convert to integer list
    return [int(x) for x in best_coeffs]
