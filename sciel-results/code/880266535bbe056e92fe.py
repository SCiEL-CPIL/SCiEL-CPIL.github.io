import numpy as np
import time

N = 600

def _compute_max_autoconv(h):
    """
    Compute the maximum autoconvolution value excluding lag 0.
    h is a vector of length N.
    Returns the max value of the autoconvolution at non-zero lags.
    """
    if np.sum(h) < 1e-12:
        return 0.0
    # Efficiently compute autoconvolution using FFT
    # We need the full convolution to get all lags -N+1 to N-1
    # Pad to at least 2N-1, but FFT is faster if size is power of 2 or similar
    # N=600, 2N-1 = 1199. Next power of 2 is 2048.
    fft_size = 1 << (2 * N - 1).bit_length()
    h_fft = np.fft.fft(h, n=fft_size)
    conv_fft = h_fft * h_fft
    conv = np.fft.ifft(conv_fft).real
    
    # The full convolution in numpy correlate mode 'full' has length 2N-1.
    # Index mapping:
    # np.correlate(h, h, 'full')[i] corresponds to sum_j h[j] h[j - (i - (N-1))]
    # Let k be the lag. k = i - (N-1).
    # i ranges from 0 to 2N-2.
    # k ranges from -(N-1) to N-1.
    # We want max over k != 0.
    # k=0 is at i = N-1.
    # So we exclude index N-1.
    
    # Ensure numerical stability
    conv = np.maximum(conv, 0) # Autoconvolution of non-negative h is non-negative
    
    # Exclude center
    if N == 1:
        return 0.0
        
    max_val = np.max(np.concatenate([conv[:N-1], conv[N:]]))
    return max_val

def _compute_score(h):
    """
    Normalized score: max_autoconv / (sum h)^2
    """
    s = np.sum(h)
    if s < 1e-12:
        return np.inf
    return _compute_max_autoconv(h) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # Initialize with uniform distribution
    # The task asks for nonnegative step heights.
    # The score is normalized by (sum h)^2.
    # So scaling h does not change the score.
    # We can fix sum(h) = 1 to simplify the search space to the simplex.
    h = np.ones(N) / N
    best_h = h.copy()
    best_score = _compute_score(h)
    
    # Frank-Wolfe / Mirror Descent on the Simplex
    # We minimize f(h) = max_{k!=0} (h^T A_k h) subject to h >= 0, sum h = 1.
    # Since f is homogeneous of degree 2, minimizing f(h) on the simplex is equivalent to minimizing the normalized score.
    
    # Gradient of f(h) w.r.t h:
    # f(h) = max_k (sum_i sum_j h_i A_{k,ij} h_j)
    # Let g(h) = [ (h^T A_0 h), (h^T A_1 h), ... ]
    # f(h) = max(g(h))
    # The subgradient of f at h is the gradient of the active term(s).
    # If unique max at k*, grad_f = 2 A_{k*} h.
    
    # We will use a simple projected gradient descent or Frank-Wolfe.
    # Frank-Wolfe is often better for simplex constraints.
    # Step:
    # 1. Compute gradient g = 2 A_{k*} h where k* is the lag with max autoconv.
    # 2. Find s in simplex minimizing <g, s>. Since s is a vertex of simplex (e_i), 
    #    min <g, e_i> = min_i g_i. So s = e_{argmin g}.
    # 3. Update h <- h + gamma (s - h).
    #    Actually, standard FW: h_new = (1 - gamma) h + gamma s.
    #    We can optimize gamma via line search or fixed schedule.
    
    # However, computing A_k h for all k is expensive if done naively.
    # But A_k h is just the autoconvolution at lag k?
    # Let's check: (A_k h)_i = sum_j A_{k,ij} h_j.
    # For circulant matrices, A_k is the matrix of the autoconvolution kernel shifted by k?
    # Actually, (h^T A_k h) is the autoconvolution at lag k.
    # The gradient of the autoconvolution at lag k, C_k = sum_i h_i h_{i+k}, w.r.t h_i is:
    # d C_k / d h_i = h_{i+k} + h_{i-k} (with boundary handling, assuming periodic or zero padding).
    # Wait, the task says "step function on [-1/4, 1/4]". This implies a compact support, not periodic.
    # So we should use zero-padding for the convolution.
    # If we use zero padding, the gradient of C_k = sum_i h_i h_{i+k} is:
    # For i such that i+k is in range: h_{i+k}
    # For i such that i-k is in range: h_{i-k}
    # So grad_i = h_{i+k} (if i+k < N) + h_{i-k} (if i-k >= 0).
    
    # This allows O(N) gradient computation for a specific lag k.
    # We need to find k* (max lag) first, then compute gradient.
    
    # To speed up finding k*, we can compute the full autoconvolution via FFT in O(N log N).
    # Then find argmax k*.
    # Then compute gradient in O(N).
    
    # Let's implement a hybrid: 
    # 1. FFT to find max lag k* and value.
    # 2. Compute gradient for k*.
    # 3. Frank-Wolfe update.
    
    # We also need to handle the "max" nature. If multiple lags are near max, 
    # using just one might be unstable. But for simplicity, we use the argmax.
    
    # Optimization:
    # We want to minimize the peak. 
    # A better approach might be to minimize the sum of squared peaks or use a smoothed max.
    # But let's try plain FW on the max first.
    
    # To make it robust, we can use a "soft" max or just iterate.
    
    # Let's refine the FW step.
    # gamma_t = 2 / (t + 2) is a standard schedule.
    
    T_max = 2000 # Max iterations
    for t in range(T_max):
        if time.time() > deadline:
            break
            
        # 1. Compute autoconvolution and find max lag
        fft_size = 1 << (2 * N - 1).bit_length()
        h_fft = np.fft.fft(h, n=fft_size)
        conv_fft = h_fft * h_fft
        conv = np.fft.ifft(conv_fft).real
        conv = np.maximum(conv, 0)
        
        # Exclude center
        if N == 1:
            break
        conv_excl = np.concatenate([conv[:N-1], conv[N:]])
        # We need to map the index in conv_excl back to the lag k
        # conv indices: 0..N-2 correspond to lags -(N-1)..-1
        # conv indices: N..2N-2 correspond to lags 1..N-1
        # conv_excl is just the concatenation.
        # We need the lag k* to compute the gradient.
        # Let's find the max in the full conv, excluding N-1.
        conv_copy = conv.copy()
        conv_copy[N-1] = -np.inf
        max_idx = np.argmax(conv_copy)
        
        # Determine lag k*
        if max_idx < N-1:
            k_star = max_idx - (N-1) # Negative lag
        else:
            k_star = max_idx - (N-1) # Positive lag
            
        # 2. Compute gradient of C_k_star w.r.t h
        # C_k = sum_i h_i h_{i+k}
        # grad_i = h_{i+k} (if valid) + h_{i-k} (if valid)
        grad = np.zeros(N)
        
        if k_star > 0:
            # i+k < N => i < N-k
            grad[:N-k_star] += h[k_star:]
            # i-k >= 0 => i >= k
            grad[k_star:] += h[:N-k_star]
        elif k_star < 0:
            # Symmetric to k > 0
            k_abs = -k_star
            grad[:N-k_abs] += h[k_abs:]
            grad[k_abs:] += h[:N-k_abs]
        else:
            # k=0 excluded, but if it happens, grad = 2h
            grad = 2 * h
            
        # 3. Frank-Wolfe update
        # min <grad, s> s.t. s in simplex
        # min is at s = e_{argmin grad}
        min_idx = np.argmin(grad)
        s = np.zeros(N)
        s[min_idx] = 1.0
        
        gamma = 2.0 / (t + 2.0)
        
        h_new = (1 - gamma) * h + gamma * s
        
        # Check score
        # We can evaluate score periodically to save time, or every step.
        # FFT is O(N log N) ~ 600 * 10 = 6000 ops. Very fast.
        # Let's evaluate every step.
        s_new = _compute_score(h_new)
        
        if s_new < best_score:
            best_score = s_new
            best_h = h_new.copy()
            h = h_new
        else:
            # If it didn't improve, we might still want to move if we are in a plateau,
            # but FW is a descent method for convex functions.
            # f(h) is NOT convex in h? 
            # f(h) = max_k (h^T A_k h).
            # Each h^T A_k h is convex if A_k is PSD.
            # A_k is the matrix for autoconvolution at lag k.
            # Is A_k PSD?
            # C_k = sum h_i h_{i+k}.
            # This is a quadratic form.
            # For k=0, it's sum h_i^2, which is convex.
            # For k!=0, it's not necessarily convex.
            # Example: N=2, k=1. C_1 = h0 h1 + h1 h0 = 2 h0 h1.
            # Matrix [[0, 1], [1, 0]]. Eigenvalues 1, -1. Not PSD.
            # So f(h) is not convex. FW may not converge to global min.
            # But it's a good heuristic.
            
            # If no improvement, we can try a smaller step or random restart.
            # To avoid getting stuck, we can accept non-improving steps with low probability (Simulated Annealing style)
            # Or just stick to greedy.
            # Given the non-convexity, let's add a small random perturbation if stuck.
            
            # For now, just update h to h_new to keep moving, even if score increases slightly?
            # No, that's bad.
            # Let's try to force improvement by checking if we are stuck.
            # If stuck for a while, restart from a random point or uniform.
            
            # Let's just update h to h_new. The score might go up, but we track best_h.
            # However, if we move away from the local min, we might escape.
            h = h_new
            
    # Final check
    # Sometimes the best found during iterations is better than the final h
    # We already track best_h.
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    # Normalize? The score is scale invariant.
    # But the output is "sequence of nonnegative step heights".
    # The evaluator likely normalizes or expects raw heights.
    # "score is the upper bound on C1".
    # "minimize the normalized maximum".
    # So the value returned by construct() is the heights.
    # The evaluator computes the score.
    # So we just return the heights.
    
    # One more optimization: Local search on the continuous h?
    # Maybe try a few random restarts if time permits.
    
    if time.time() < deadline - 20:
        # Try 5 random restarts
        for _ in range(5):
            if time.time() > deadline:
                break
            h_init = np.random.dirichlet(np.ones(N))
            h = h_init
            # Run a short FW loop
            for t in range(200):
                if time.time() > deadline:
                    break
                fft_size = 1 << (2 * N - 1).bit_length()
                h_fft = np.fft.fft(h, n=fft_size)
                conv_fft = h_fft * h_fft
                conv = np.fft.ifft(conv_fft).real
                conv = np.maximum(conv, 0)
                
                if N == 1:
                    break
                conv_copy = conv.copy()
                conv_copy[N-1] = -np.inf
                max_idx = np.argmax(conv_copy)
                
                if max_idx < N-1:
                    k_star = max_idx - (N-1)
                else:
                    k_star = max_idx - (N-1)
                    
                grad = np.zeros(N)
                if k_star > 0:
                    grad[:N-k_star] += h[k_star:]
                    grad[k_star:] += h[:N-k_star]
                elif k_star < 0:
                    k_abs = -k_star
                    grad[:N-k_abs] += h[k_abs:]
                    grad[k_abs:] += h[:N-k_abs]
                else:
                    grad = 2 * h
                    
                min_idx = np.argmin(grad)
                s = np.zeros(N)
                s[min_idx] = 1.0
                gamma = 2.0 / (t + 2.0)
                h_new = (1 - gamma) * h + gamma * s
                s_new = _compute_score(h_new)
                
                if s_new < best_score:
                    best_score = s_new
                    best_h = h_new.copy()
                h = h_new

    return list(best_h)
