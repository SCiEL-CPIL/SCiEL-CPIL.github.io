import time
import numpy as np
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """
    Compute the normalized maximum of the autoconvolution.
    h: array of nonnegative values.
    Returns max_k (h*h)[k] / (sum h)^2
    """
    s = np.sum(h)
    if s == 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    return max_conv / (s * s)

def fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * f, size)[:2*n-1]
    return conv

def smooth_max_and_grad(h, beta=100.0):
    """
    Smooth approximation of max(conv) / (sum h)^2 with analytical gradient.
    Returns (score, grad_score) where grad_score has shape (N,).
    """
    s = np.sum(h)
    if s == 0:
        return 0.0, np.zeros_like(h)
    
    conv = fft_convolve(h)
    # Scale for numerical stability in log-sum-exp
    conv_scaled = conv / beta
    max_conv_scaled = np.max(conv_scaled)
    exp_terms = np.exp(conv_scaled - max_conv_scaled)
    log_sum_exp = max_conv_scaled + np.log(np.sum(exp_terms))
    smooth_max = beta * log_sum_exp
    
    score = smooth_max / (s * s)
    
    # Weight vector w[j] = exp(conv[j]/beta - max) / sum(exp(...))
    w = exp_terms / np.sum(exp_terms)
    
    # grad_smooth_max[i] = sum_j w[j] * (h[j-i] if 0<=j-i<N else 0 + h[i-j] if 0<=i-j<N else 0)
    # This equals 2 * cross-correlation of w and h, computed via FFT
    # cross_corr[i] = sum_k w[i+k] * h[k] for valid indices
    # Using FFT: ifft(fft(w_padded) * conj(fft(h_padded))) gives cross-correlation
    
    size = 2 * N
    w_pad = np.zeros(size)
    w_pad[:2*N-1] = w
    h_pad = np.zeros(size)
    h_pad[:N] = h
    
    # Cross-correlation: sum_k w[i+k] * h[k] 
    # = ifft(fft(w) * conj(fft(h))) but we need to be careful with indexing
    # Actually, let's compute it directly using the convolution structure
    # grad_sm[i] = sum_{k=0}^{N-1} h[k] * (w[i+k] + w[i-k]) for valid indices
    
    # Use FFT to compute this efficiently
    # Let's compute two convolutions:
    # A[i] = sum_k h[k] * w[i+k]  (this is a cross-correlation)
    # B[i] = sum_k h[k] * w[i-k]  (this is also related to cross-correlation)
    
    # For A: pad w to length 3N-2, take the first N elements starting at i
    # Actually, let's use the fact that:
    # sum_k h[k] * w[i+k] is the i-th element of the cross-correlation of h and w
    
    # Cross-correlation of h and w: (h ⊛ w)[i] = sum_k h[k] * conj(w[k-i])
    # We want sum_k h[k] * w[i+k]
    
    # Let's define u = reverse of w, then sum_k h[k] * w[i+k] relates to convolution
    # This is getting complex. Let's use a different approach.
    
    # Since N=600, let's compute the gradient using matrix-free FFT approach:
    # grad_sm[i] = sum_{j=0}^{2N-2} w[j] * M[j,i]
    # where M[j,i] = h[j-i] if 0<=j-i<N else 0, plus h[i-j] if 0<=i-j<N else 0
    
    # Note that M[j,i] = h[j-i] + h[i-j] (with boundary conditions)
    # This is symmetric, so M[j,i] = M[i,j] if we consider the right structure
    
    # Actually, let's compute it as:
    # grad_sm = convolve(w, h) + convolve(w, h_reversed) with appropriate shifts
    
    # Let me think again. For autoconvolution conv[j] = sum_k h[k]h[j-k]
    # d(conv[j])/dh[i] = h[j-i] (if 0<=j-i<N) + h[i-j] (if 0<=i-j<N)
    
    # So grad_sm[i] = sum_j w[j] * (h[j-i] * I(0<=j-i<N) + h[i-j] * I(0<=i-j<N))
    
    # First term: sum_j w[j] * h[j-i] for j-i in [0,N-1]
    # Let k = j-i, j = i+k: sum_k w[i+k] * h[k] for k in [0,N-1] and i+k in [0,2N-2]
    
    # Second term: sum_j w[j] * h[i-j] for i-j in [0,N-1]  
    # Let k = i-j, j = i-k: sum_k w[i-k] * h[k] for k in [0,N-1] and i-k in [0,2N-2]
    
    # First term is like a "shifted dot product": for each i, dot(h, w[i:i+N]) with padding
    # Second term is dot(h, w[max(0,i-N+1):i+1]) reversed... 
    
    # Let's compute both using FFT:
    # Define v = w of length 2N-1, pad to size
    # First term[i] = sum_{k=0}^{N-1} h[k] * w[i+k] if i+k < 2N-1 else 0
    # This is the i-th element of the cross-correlation of h and w (with appropriate definition)
    
    # Cross-correlation (h ⊛ w)[m] = sum_n h[n] * conj(w[n-m])
    # We want sum_k h[k] * w[i+k] = sum_k h[k] * conj(conj(w[i+k]))
    # Let u = conj(w), then we want sum_k h[k] * conj(u[i+k]) which is (h ⊛ u)[i] if we define properly
    
    # Actually, let's just compute it directly with numpy for N=600 since it's manageable
    # But let's try to vectorize better.
    
    # Create a Toeplitz-like structure using FFT
    # Let's compute first_term[i] = sum_{k=0}^{min(N-1, 2N-2-i)} h[k] * w[i+k]
    # This is a sliding dot product. We can compute all at once using FFT.
    
    # Define w_ext of length 3N-2: w_ext[i] = w[i] for i in [0, 2N-2], else 0
    # Then first_term[i] = sum_{k=0}^{N-1} h[k] * w_ext[i+k]
    # This is the (i)-th element of the convolution of h with reverse(w_ext)?
    
    # Let me try a different approach: use the fact that this is a linear operator
    # and compute it via FFT.
    
    # For now, let's compute it in O(N^2) but vectorized in numpy (no Python loop)
    # Since N=600, N^2 = 360000 which is fine.
    
    # First term: for each i in [0,N-1], sum over k in [0,N-1] of h[k]*w[i+k] if i+k < 2N-1
    # We can create a matrix W1 where W1[i,k] = w[i+k] if i+k < 2N-1 else 0
    # Then first_term = W1 @ h
    
    # But creating a 600x600 matrix is fine.
    
    # Actually, let's use a smarter approach with FFT for the full gradient.
    # The gradient is a linear function of h, so we can precompute the linear operator.
    
    # For now, let's do it with a vectorized approach without explicit matrix.
    
    # first_term[i] = dot(h, w[i:i+N]) with zero padding
    # We can compute this as: for each i, take the slice w[i:i+N] and dot with h
    # Let's create an array of all such slices using stride tricks or just compute via FFT
    
    # Let's use FFT: 
    # sum_k h[k] * w[i+k] = ifft(fft(h) * fft(w_shifted))[i] where w_shifted is appropriately defined
    
    # This is complex. Let's just do it with a loop over i but vectorize over k.
    # For N=600, 600 iterations of dot products of length 600 is fast in numpy.
    
    grad_sm = np.zeros(N)
    for i in range(N):
        # First term: sum_k h[k] * w[i+k] for valid k
        k_vals = np.arange(N)
        j_vals = i + k_vals
        valid = (j_vals >= 0) & (j_vals < 2*N-1)
        if np.any(valid):
            term1 = np.dot(h[k_vals[valid]], w[j_vals[valid]])
        else:
            term1 = 0.0
        
        # Second term: sum_k h[k] * w[i-k] for valid k
        j_vals2 = i - k_vals
        valid2 = (j_vals2 >= 0) & (j_vals2 < 2*N-1)
        if np.any(valid2):
            term2 = np.dot(h[k_vals[valid2]], w[j_vals2[valid2]])
        else:
            term2 = 0.0
        
        grad_sm[i] = term1 + term2
    
    # Now, score = smooth_max / s^2
    # d_score/dh_i = (1/s^2) * grad_sm[i] - 2*smooth_max/s^3 * 1
    grad_score = grad_sm / (s * s) - 2 * smooth_max / (s * s * s)
    
    return score, grad_score

def objective_and_grad(x, beta):
    """Objective and gradient for L-BFGS-B."""
    h = np.maximum(x, 0)
    score, grad = smooth_max_and_grad(h, beta)
    # Chain rule: if x_i < 0, h_i = 0, so dh_i/dx_i = 0
    mask = x >= 0
    grad_out = np.where(mask, grad, 0)
    return score, grad_out

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Initialize with uniform distribution
    h_uniform = np.full(N, 1.0)
    
    best_h = h_uniform.copy()
    best_score = compute_score(h_uniform)
    
    # Coordinate descent with smooth max surrogate
    # Use decreasing beta for annealing
    
    np.random.seed(42)
    
    # Start from uniform
    h = h_uniform.copy()
    
    # Phase 1: Coordinate descent with moderate beta
    for beta in [20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
        if time.time() > deadline - 20:
            break
        
        # Multiple passes of coordinate descent
        num_passes = 3 if beta < 100 else 2
        
        for pass_num in range(num_passes):
            if time.time() > deadline - 10:
                break
            
            # Compute score and gradient once per pass
            score, grad = smooth_max_and_grad(h, beta)
            
            # Coordinate descent: for each coordinate, find optimal value
            # For coordinate i, the score as a function of h_i is:
            # f(t) = smooth_max(h with h_i=t) / (s + t - h_i_old)^2
            # This is a 1D problem. We can use the gradient to do a line search.
            
            # Simpler: use gradient-based update with step size
            # Update: h_i <- max(0, h_i - lr * grad_i)
            # But we need to handle the non-negativity constraint
            
            # Let's do a simple projected gradient step
            lr = 0.01 / beta  # Learning rate scaled by beta
            
            # Compute new h
            h_new = h - lr * grad
            h_new = np.maximum(h_new, 0)
            
            # Check if this improves the actual score
            new_score = compute_score(h_new)
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
            
            h = h_new
        
        # After each beta, check actual score
        actual_score = compute_score(h)
        if actual_score < best_score:
            best_score = actual_score
            best_h = h.copy()
    
    # Phase 2: Refine with coordinate descent on the actual objective
    # For each coordinate, optimize h_i given all other h_j fixed
    # The autoconvolution max is a piecewise quadratic function of h_i
    
    if time.time() < deadline - 15:
        h = best_h.copy()
        
        # Coordinate descent on actual objective
        for coord_pass in range(5):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                
                # Save current value
                old_val = h[i]
                
                # The score as a function of h_i = t:
                # conv(t) = conv_old + t * (h shifted) + t^2 * delta
                # Specifically, conv[k] = conv_old[k] + t * (h[k-i] + h[i-k]) + t^2 * delta[k]
                # where delta[k] = 1 if k == i (from h[i]*h[i] term)
                
                # Actually, conv_new[k] = conv_old[k] + t*(h[k-i] + h[i-k]) - old_val*(h[k-i] + h[i-k]) + (t^2 - old_val^2)*delta_{k,i}
                # Wait, let me redo this.
                
                # conv_old[k] = sum_j h[j] h[k-j]
                # conv_new[k] = sum_j h_new[j] h_new[k-j]
                # h_new[i] = t, h_new[j] = h[j] for j != i
                # conv_new[k] = conv_old[k] - h[i]*h[k-i] - h[i]*h[i-k] + h[i]^2*delta_{k,i}
                #             + t*h[k-i] + t*h[i-k] - t^2*delta_{k,i}
                # Wait, that's not right either. Let me be more careful.
                
                # conv_old[k] = sum_{j=0}^{N-1} h[j] h[k-j] (with h[x]=0 if x not in [0,N-1])
                # conv_new[k] = sum_{j=0}^{N-1} h_new[j] h_new[k-j]
                # = sum_{j != i} h[j] h[k-j] + h_new[i] h_new[k-i] + h_new[k-i] h_new[i] (if k-i != i)
                # Actually, the term j=i gives h_new[i] * h_new[k-i]
                # The term j=k-i gives h_new[k-i] * h_new[i] (if k-i != i)
                # If k-i == i, i.e., k=2i, then it's h_new[i]^2
                # If k-i != i, then we have two terms: h_new[i]*h_new[k-i] + h_new[k-i]*h_new[i] = 2*h_new[i]*h_new[k-i]
                
                # This is getting complex. Let's just compute the score for a few candidate values of t.
                
                # Try a few values: 0, old_val, and a few around old_val
                candidates = [0.0, old_val]
                if old_val > 0:
                    candidates.extend([old_val * 0.5, old_val * 1.5, old_val * 0.1, old_val * 5.0])
                
                best_t = old_val
                best_local_score = compute_score(h)
                
                for t in candidates:
                    h[i] = t
                    s = np.sum(h)
                    if s == 0:
                        continue
                    conv = fft_convolve(h)
                    local_score = np.max(conv) / (s * s)
                    if local_score < best_local_score:
                        best_local_score = local_score
                        best_t = t
                
                h[i] = best_t
                if h[i] != old_val:
                    improved = True
            
            # Check overall score
            actual_score = compute_score(h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = h.copy()
            
            if not improved:
                break
    
    # Final check: ensure non-negativity
    best_h = np.maximum(best_h, 0)
    
    return best_h.tolist()
# EVOLVE-BLOCK-END
