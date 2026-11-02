import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a signed step function with 400 intervals to minimize the 
    upper bound on the third autocorrelation coefficient C3.
    
    The strategy is to use a local search (steepest descent) on the signs 
    to minimize the L2 norm of the autocorrelation sidelobes.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Initialize with random signs
    s = np.random.choice([-1, 1], size=N)
    
    # Compute initial autocorrelation A_k for k in [-N+1, N-1]
    # A_k = sum_{i} s_i s_{i+k}
    # We store A in an array of length 2N-1, centered at index N-1
    # A[0] corresponds to k = -(N-1)
    # A[N-1] corresponds to k = 0
    A = np.zeros(2*N - 1)
    
    # Efficient calculation of autocorrelation using FFT
    # Pad to next power of 2 for efficiency, though N=400 is small
    size = 2 * N
    s_fft = np.fft.fft(s, n=size)
    A_fft = np.fft.ifft(s_fft * np.conj(s_fft)).real
    # The circular autocorrelation is A_fft.
    # Since we padded with zeros, the first N values are the linear autocorrelation for lags 0 to N-1.
    # A_fft[0] = A_0
    # A_fft[k] = A_k
    # A_fft[size-k] = A_{-k} = A_k
    A_lin = A_fft[:N]
    
    # Construct full A array
    # k = 0 is at index N-1 in our 2N-1 array
    A[N-1] = A_lin[0]
    for k in range(1, N):
        A[N-1+k] = A_lin[k]
        A[N-1-k] = A_lin[k]
        
    # Calculate current cost: Sum of squares of sidelobes
    # Exclude k=0 (index N-1)
    cost = np.sum(A**2) - A[N-1]**2
    
    best_cost = cost
    best_s = s.copy()
    best_A = A.copy()
    
    # Local Search
    # We iterate through all indices, find the best flip, apply it, repeat.
    # To speed up, we can just do a single pass of greedy flips, then restart.
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            # Calculate change in A if we flip s[i]
            # Delta A_k = -2 * s[i] * (s[i+k] + s[i-k])
            # We need to be careful with boundaries
            
            # Vectorized calculation of delta_A for all k
            # k ranges from -(N-1) to N-1
            # Let's map k to index in A: idx = N-1+k
            
            # s[i+k] is valid if 0 <= i+k < N
            # s[i-k] is valid if 0 <= i-k < N
            
            # Instead of explicit loops for delta, we can update A directly
            # But we need to evaluate the cost change first.
            
            # Let's compute the new cost for flipping i
            # Delta A_k = -2 * s[i] * (s[i+k] + s[i-k])
            
            # We can compute the contribution to the cost change
            # Delta Cost = sum_{k != 0} (A_k + Delta A_k)^2 - A_k^2
            #            = sum_{k != 0} (2 A_k Delta A_k + Delta A_k^2)
            
            # We need Delta A_k for all k.
            # This is effectively the convolution of the "impulse" at i with the sequence s.
            # Flipping s[i] is equivalent to adding -2*s[i]*delta_i to the sequence.
            # The change in autocorrelation is the autocorrelation of the perturbation.
            # Perturbation p: p[i] = -2*s[i], others 0.
            # Delta A_k = (p * p)_k? No.
            # A(s+p) = A(s) + A(s,p) + A(p,s) + A(p,p).
            # Since p is sparse, A(p,p) is only non-zero at k=0.
            # A(s,p)_k = sum s_j p_{j+k} = s_{i-k} p_i = -2 s[i] s_{i-k}.
            # A(p,s)_k = sum p_j s_{j+k} = p_i s_{i+k} = -2 s[i] s_{i+k}.
            # So Delta A_k = -2 s[i] (s_{i-k} + s_{i+k}).
            
            # We can compute Delta A_k for all k using slicing.
            # k is the lag.
            # For a given i, s[i+k] is a shifted version of s.
            
            # Let's create an array for s_i_plus_k and s_i_minus_k
            # This is tricky with boundaries in a vectorized way without padding.
            
            # Alternative: Just update A directly if we decide to flip.
            # To decide, we can estimate the cost change.
            
            # Given N=400, a simple loop over k for the cost evaluation is O(N).
            # Total O(N^2) per pass.
            
            s_i = s[i]
            
            # Calculate Delta A_k for all k
            # k ranges from -(N-1) to N-1
            # We'll iterate k from 1 to N-1 (symmetric)
            
            delta_cost = 0.0
            
            # Optimization: Precompute A_sq = A**2
            # We need sum(2 A_k Delta A_k + Delta A_k^2) for k != 0
            
            # Let's just compute the new A for this flip and its cost?
            # That's O(N).
            
            # We can vectorize the delta calculation for a fixed i
            # k_pos = np.arange(1, N)
            # indices_plus = i + k_pos
            # indices_minus = i - k_pos
            
            # valid_plus = (indices_plus >= 0) & (indices_plus < N)
            # valid_minus = (indices_minus >= 0) & (indices_minus < N)
            
            # s_plus = np.where(valid_plus, s[indices_plus], 0)
            # s_minus = np.where(valid_minus, s[indices_minus], 0)
            
            # delta_A_pos = -2 * s_i * (s_plus + s_minus)
            
            # A_pos = A[N-1 + k_pos]
            
            # delta_cost_pos = np.sum(2 * A_pos * delta_A_pos + delta_A_pos**2)
            # total_delta_cost = 2 * delta_cost_pos (for negative k)
            
            # This is O(N) per i.
            
            k_pos = np.arange(1, N)
            idx_plus = i + k_pos
            idx_minus = i - k_pos
            
            valid_plus = (idx_plus < N)
            valid_minus = (idx_minus >= 0)
            
            # Note: idx_plus >= 0 is always true since i>=0, k_pos>=1
            # idx_minus < N is always true since i<N, k_pos>=1 -> idx_minus < N-1
            
            s_plus = np.where(valid_plus, s[idx_plus], 0.0)
            s_minus = np.where(valid_minus, s[idx_minus], 0.0)
            
            delta_A = -2.0 * s_i * (s_plus + s_minus)
            
            # A_k for k > 0
            A_pos = A[N-1 + k_pos]
            
            # Cost change for k > 0
            dc_pos = np.sum(2.0 * A_pos * delta_A + delta_A**2)
            
            # Total cost change (k < 0 is symmetric)
            total_dc = 2.0 * dc_pos
            
            if total_dc < -1e-9:
                # Apply flip
                s[i] = -s_i
                A[N-1 + k_pos] += delta_A
                A[N-1 - k_pos] += delta_A
                cost += total_dc
                improved = True
                if cost < best_cost:
                    best_cost = cost
                    best_s = s.copy()
                    best_A = A.copy()
                    
        if not improved:
            # No improvement in this pass, try random restart or break
            # To avoid local minima, we can do a few random flips
            # But steepest descent is usually good enough for this landscape
            break
            
    # Return the best sequence found
    # The task asks for "signed step heights".
    # Assuming the evaluator handles normalization or expects +/- 1.
    return best_s
# EVOLVE-BLOCK-END
