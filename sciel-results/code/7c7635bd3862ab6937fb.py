# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    """
    Discrete 3-D Kakeya set: union of 64 sheared tubes (N = 8).
    
    Strategy:
    1. Exploit the separability of the volume: V = E[Ux(c)] * E[Uy(c)].
    2. Optimize the 1D base positions x_i and y_i independently.
    3. Use a vectorized, high-resolution grid search over a parameterized 
       family of configurations, followed by a local refinement using 
       Nelder-Mead on the exact 1D union length integral (computed via 
       dense MC sampling).
    """
    N = 8
    deadline = time.time() + 600.0  # Safety margin

    # 1. Analytical Initialization
    # The centered shear alignment x_k = -k/2N is a strong starting point.
    # It minimizes the spread of the intervals around the origin.
    k = np.arange(N, dtype=np.float64)
    x_init = -k / (2.0 * N)
    y_init = -k / (2.0 * N)

    # 2. Monte Carlo Setup for 1D Union Length Estimation
    # We need to estimate E[Ux(c)] = integral_0^1 Ux(c) dc.
    # Ux(c) is the length of the union of intervals [x_i + c*i/N, x_i + c*i/N + 1/N].
    
    # To make the estimation fast and accurate, we use a large number of samples.
    # Since N=8, we can compute the union length exactly for each c by sorting 
    # the 8 intervals. We can vectorize this over a batch of c values.
    
    M = 500_000  # Large sample size for stable gradient/signal
    rng = np.random.default_rng(12345)
    c_samples = rng.uniform(0, 1, M)
    
    # Precompute the shear coefficients
    i_vals = np.arange(N, dtype=np.float64)
    shear_coeffs = i_vals / N  # shape (N,)
    
    def compute_1d_union_length(x, c_samples):
        """
        Computes the average union length of the intervals for given x and c_samples.
        x: shape (N,)
        c_samples: shape (M,)
        Returns: scalar (average union length)
        """
        # Intervals for each i and each c:
        # L_i(c) = x_i + c * (i/N)
        # R_i(c) = L_i(c) + 1/N
        
        # We want to compute the union length for each c in c_samples.
        # Vectorization strategy:
        # 1. Compute L and R matrices of shape (N, M).
        # 2. For each column m (each c), sort the intervals by L.
        # 3. Merge intervals to find the total length.
        
        # Since N is small (8), we can use a loop over M if vectorized merging is too complex,
        # but let's try to vectorize the sorting and merging as much as possible.
        
        # L: (N, M)
        L = x[:, None] + c_samples[None, :] * shear_coeffs[:, None]
        R = L + 1.0 / N
        
        # Transpose to (M, N) for easier row-wise operations
        L_t = L.T  # (M, N)
        R_t = R.T  # (M, N)
        
        # Sort intervals by their start point L
        # Order: (M, N)
        order = np.argsort(L_t, axis=1)
        L_sorted = np.take_along_axis(L_t, order, axis=1)
        R_sorted = np.take_along_axis(R_t, order, axis=1)
        
        # Now merge intervals row by row.
        # We can do this with a loop over the N intervals (8 iterations) 
        # but vectorized over M rows.
        
        # Initialize current end and total length
        cur_end = R_sorted[:, 0].copy()
        total_len = np.zeros(M)
        
        # Iterate through the sorted intervals (skip the first one which is already added to cur_end)
        # Actually, the standard merge algorithm:
        # current_merged_end = R[0]
        # for k in 1..N-1:
        #   if L[k] <= current_merged_end:
        #      current_merged_end = max(current_merged_end, R[k])
        #   else:
        #      total_len += current_merged_end - start_of_current_merged
        #      start_of_current_merged = L[k]
        #      current_merged_end = R[k]
        # total_len += current_merged_end - start_of_current_merged
        
        # We need to track the start of the current merged segment.
        cur_start = L_sorted[:, 0].copy()
        
        for k in range(1, N):
            L_k = L_sorted[:, k]
            R_k = R_sorted[:, k]
            
            # Check if current interval overlaps with the merged segment
            # Overlap condition: L_k <= cur_end
            overlap_mask = (L_k <= cur_end)
            
            # Update cur_end for overlapping intervals
            # If overlap, cur_end = max(cur_end, R_k)
            # If no overlap, we finalize the previous segment and start a new one.
            
            # This logic is tricky to vectorize perfectly with the "start" tracking.
            # Let's use a simpler, robust vectorized approach for small N:
            # Compute the union length by summing the lengths of all intervals 
            # and subtracting the overlaps? No, inclusion-exclusion is complex for N=8.
            
            # Alternative: The union length is the integral of the indicator function.
            # We can estimate it by sampling points? No, we want exact for the MC samples.
            
            # Let's stick to the loop but optimize it.
            # Since N=8, we can just loop over M in Python? 500k * 8 = 4M ops. 
            # In Python, this might take ~1-2 seconds. Acceptable for a few evaluations.
            # But we want to do many evaluations.
            
            # Let's try a fully vectorized merge using the property that N is small.
            # We can compute the union length as:
            # Sum_{k=0}^{N-1} max(0, R_sorted[:, k] - max(L_sorted[:, k], max_{j<k} R_sorted[:, j]))
            # This is O(N^2) per sample. N=8 -> 64 ops. 500k * 64 = 32M ops. 
            # Vectorized NumPy can do this very fast.
            
            pass # We will implement the O(N^2) vectorized method below
            
        # Let's implement the O(N^2) vectorized union length calculation.
        # Union Length = Sum_{k=0}^{N-1} ( R_k - max(L_k, PrevMaxR) )_+
        # Where PrevMaxR for k is max(R_0, ..., R_{k-1}).
        
        # We can compute PrevMaxR for all k in a vectorized way?
        # prev_max_r = np.maximum.accumulate(R_sorted, axis=1) # This gives max up to k, inclusive.
        # We need max up to k-1.
        # Let's create a shifted max array.
        
        # R_sorted shape (M, N)
        # max_r_up_to_k = np.maximum.accumulate(R_sorted, axis=1)
        # max_r_before_k: for k=0, it's -inf. For k>0, it's max_r_up_to_k[:, k-1].
        
        max_r_up_to = np.maximum.accumulate(R_sorted, axis=1)
        
        # Create array of max R before current index
        # Shape (M, N)
        # col 0: -inf
        # col k: max_r_up_to[:, k-1]
        
        max_r_before = np.empty_like(R_sorted)
        max_r_before[:, 0] = -np.inf
        max_r_before[:, 1:] = max_r_up_to[:, :-1]
        
        # The contribution of the k-th sorted interval to the union length is:
        # max(0, R_sorted[:, k] - max(L_sorted[:, k], max_r_before[:, k]))
        
        effective_start = np.maximum(L_sorted, max_r_before)
        contributions = np.maximum(0, R_sorted - effective_start)
        
        union_lens = np.sum(contributions, axis=1)
        
        return np.mean(union_lens)

    def optimize_1d(x_init):
        """
        Optimizes the 1D base positions x to minimize the average union length.
        """
        # Phase 1: Coarse Grid Search
        # The optimal x_i are likely close to the centered configuration.
        # Let's search in a small box around the initial guess.
        # Range: [-0.5, 0.5] is sufficient.
        
        # We will perform a coordinate-wise grid search.
        # For each dimension, we sweep a grid of values and pick the best.
        # This is slow if done naively, but we can vectorize the evaluation 
        # of a grid of candidates for one dimension at a time.
        
        x = x_init.copy()
        
        # Number of grid points per dimension
        grid_size = 50
        # Step size for the grid
        # Initial range for each x_i: x_init[i] +/- 0.1
        # Let's use a global range [-0.5, 0.5] for all i, but focus on local refinement.
        
        # Actually, a better approach for 8 dimensions is to use a multivariate 
        # grid search? 50^8 is too big.
        # Coordinate descent with a fine grid is better.
        
        # Let's use a more efficient optimizer: Nelder-Mead on the MC estimate.
        # The function is smooth enough for Nelder-Mead if the MC noise is low.
        # With M=500k, the noise is small.
        
        def objective(x):
            return compute_1d_union_length(x, c_samples)
        
        # Initial simplex
        x0 = x_init.copy()
        
        # Run Nelder-Mead
        # maxiter should be sufficient to converge.
        # time.time() check inside the objective? 
        # Nelder-Mead doesn't easily support early stopping via time in the standard scipy call 
        # unless we wrap the objective to raise an exception or return a large value.
        
        # Let's add a time check to the objective.
        def objective_with_time(x):
            if time.time() > deadline - 10:
                return 1e9
            return compute_1d_union_length(x, c_samples)
        
        res = minimize(objective_with_time, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        # If Nelder-Mead fails or times out, we might have a bad result.
        # Let's verify the result is finite and valid.
        if res.fun < 1e8:
            x_opt = res.x
        else:
            x_opt = x0
            
        # Phase 2: Local Refinement with a finer grid or more MC samples?
        # Let's do a final check with a larger MC sample size to ensure robustness.
        # But we are near the deadline.
        
        return x_opt

    # Optimize x and y independently
    x_opt = optimize_1d(x_init)
    y_opt = optimize_1d(y_init)
    
    # Construct the 64x2 array
    # Row k is tube (i, j) with i = k // 8 and j = k % 8.
    # Base position is (x_i, y_j).
    
    positions = np.zeros((64, 2), dtype=np.float64)
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = x_opt[i]
        positions[k, 1] = y_opt[j]
        
    return positions

# EVOLVE-BLOCK-END
