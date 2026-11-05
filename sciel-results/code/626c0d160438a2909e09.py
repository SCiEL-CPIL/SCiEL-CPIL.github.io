"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    c = np.convolve(h, h)
    return abs(2.0 * N * np.max(c) / (s * s))

def _autoconvolve(h):
    return np.convolve(h, h)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # Should be 2.0
    
    # Strategy: minimize max(convolve(h,h)) subject to sum(h) = 1
    # Scale invariance means we can fix sum(h) = 1
    
    def project_sum1(h):
        return h - (np.sum(h) - 1.0) / N
    
    def objective_and_grad(h):
        h_proj = project_sum1(h)
        c = np.convolve(h_proj, h_proj)
        k_max = np.argmax(c)
        val = c[k_max]
        
        # Gradient of c[k_max] with respect to h
        # c[k] = sum_i h[i] h[i+k] for full convolution
        # For full convolution of length 2N-1, c[k] = sum_{i=max(0,k-N+1)}^{min(N-1, N-1+k-N+1)} ...
        # Actually, for np.convolve(h, h) with h of length N:
        # c[k] = sum_{i} h[i] * h[i+k-N+1] ... this is getting confusing.
        
        # Let me think about np.convolve(h, h) more carefully.
        # np.convolve(a, b) = sum_k a[k] * b[n-k] (cross-correlation style but with flip)
        # Actually np.convolve does: c[n] = sum_k a[k] * b[n-k]
        # For autoconvolution: c[n] = sum_k h[k] * h[n-k]
        # Valid n range: 0 to 2N-2
        
        # dc[n]/dh[j] = h[n-j] (if 0 <= n-j <= N-1) + h[j-n] (if 0 <= j-n <= N-1)
        # Wait, that's not right either. Let me be more careful.
        
        # c[n] = sum_k h[k] * h[n-k] where k ranges so that both indices are valid.
        # dc[n]/dh[j] = h[n-j] (from the first h[k] term when k=j) + h[j-n] (from the second h[n-k] term when n-k=j, i.e., k=n-j)
        
        # Actually: dc[n]/dh[j] = h[n-j] if 0<=n-j<N, plus h[j-n] if 0<=j-n<N
        # But these are the same term when n-j = j-n, i.e., n=2j.
        
        # Let me just compute it numerically or use a different approach.
        
        # Alternative: use the fact that c = convolve(h,h), and the gradient can be computed
        # via the adjoint of the convolution operation.
        
        # For now, let's use finite differences or a simpler approach.
        # Actually, let's use a different strategy: random search + local optimization.
        
        return val, None  # Placeholder
    
    # Better approach: use random initialization and local refinement
    # Try different random signals
    
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with sum = 1
        if restart == 0:
            h = np.ones(N) / N  # sum = 1
        else:
            # Random signal with various structures
            mode = restart % 5
            if mode == 0:
                h = np.random.randn(N)
            elif mode == 1:
                h = np.sin(2 * np.pi * np.random.randint(1, 20) * np.arange(N) / N)
                h = h + np.random.randn(N) * 0.1
            elif mode == 2:
                h = np.cos(2 * np.pi * np.random.randint(1, 20) * np.arange(N) / N)
                h = h + np.random.randn(N) * 0.1
            elif mode == 3:
                # Block structure
                block_size = np.random.randint(5, 40)
                n_blocks = N // block_size
                h = np.repeat(np.random.randn(n_blocks), block_size)
                if N % block_size != 0:
                    h = np.concatenate([h, np.random.randn(N - n_blocks * block_size)])
            else:
                # Sparse-like
                h = np.zeros(N)
                n_nonzero = np.random.randint(N//4, N)
                idx = np.random.choice(N, n_nonzero, replace=False)
                h[idx] = np.random.randn(n_nonzero)
            
            h = project_sum1(h)
        
        # Local optimization: gradient descent on max(convolve(h,h))
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            c = np.convolve(h, h)
            k_max = np.argmax(c)
            val = c[k_max]
            
            # Compute gradient of c[k_max] with respect to h
            # c[k] = sum_i h[i] * h[i + k - (N-1)] ... 
            # Let me use the definition: c[n] = sum_k h[k] * h[n-k] for n=0,...,2N-2
            # where h is zero-padded outside [0, N-1]
            
            # dc[n]/dh[j] = h[n-j] + h[j-n] (with h[i]=0 for i outside [0,N-1])
            # But wait, this double counts when n-j = j-n, i.e., n = 2j
            
            # Actually: c[n] = sum_k h[k]*h[n-k]
            # dc[n]/dh[j] = h[n-j] (derivative w.r.t. h[k] at k=j) + h[j-n] (derivative w.r.t. h[n-k] at n-k=j, k=n-j)
            # These are equal when n-j = j-n, i.e., n=2j, so:
            # dc[n]/dh[j] = 2*h[n-j] if n=2j, else h[n-j] + h[j-n]
            # But h[n-j] and h[j-n] are the same value if we define h symmetrically... no, h is not symmetric.
            
            # Let me just be explicit:
            # dc[n]/dh[j] = h[n-j] if 0<=n-j<N else 0, PLUS h[j-n] if 0<=j-n<N else 0
            # Note: h[n-j] and h[j-n] are different unless n=j or we're in a symmetric case.
            
            grad = np.zeros(N)
            # For each j, dc[k_max]/dh[j] = h[k_max - j] (if valid) + h[j - k_max] (if valid)
            for j in range(N):
                idx1 = k_max - j
                idx2 = j - k_max
                g = 0.0
                if 0 <= idx1 < N:
                    g += h[idx1]
                if 0 <= idx2 < N:
                    g += h[idx2]
                grad[j] = g
            
            # Project gradient onto sum=0 hyperplane (since we're constrained to sum(h)=1)
            grad = grad - np.mean(grad)
            
            # Gradient descent step
            lr = 0.01 / (np.linalg.norm(grad) + 1e-10)
            h_new = h - lr * grad
            h_new = project_sum1(h_new)
            
            # Check if it improved
            c_new = np.convolve(h_new, h_new)
            val_new = np.max(c_new)
            
            if val_new < val:
                h = h_new
            else:
                # Try smaller step
                lr *= 0.5
                h_new = h - lr * grad
                h_new = project_sum1(h_new)
                c_new = np.convolve(h_new, h_new)
                val_new = np.max(c_new)
                if val_new < val:
                    h = h_new
                else:
                    break
        
        # Evaluate
        val_final = _compute_value(h)
        if val_final < best_val:
            best_val = val_final
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
