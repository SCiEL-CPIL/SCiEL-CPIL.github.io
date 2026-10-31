import numpy as np
import time

N = 600
DEADLINE = 140.0

def construct():
    start_time = time.time()
    
    # Initial guess: U-shaped
    x = np.linspace(-1.0, 1.0, N)
    h = 1.0 + 0.5 * x**2
    h = np.maximum(h, 0.0)
    
    best_h = h.copy()
    best_score = np.inf
    
    def compute_ac(h):
        n = 2 * N
        h_pad = np.zeros(n)
        h_pad[:N] = h
        H = np.fft.fft(h_pad)
        ac = np.fft.ifft(H * H).real
        return ac
    
    def score(h):
        ac = compute_ac(h)
        s = np.sum(h)
        if s < 1e-10:
            return 0.0
        return np.max(ac) / (s * s)
    
    best_score = score(best_h)
    
    lr = 0.05
    for iteration in range(3000):
        if time.time() - start_time > DEADLINE:
            break
        
        ac = compute_ac(h)
        max_val = np.max(ac)
        
        # Find indices where ac is near max
        threshold = max_val * 0.99
        mask = ac >= threshold
        
        # Compute subgradient vectorized
        # grad[j] = sum_{k: mask[k]} (h[j+k] if 0<=j+k<N else 0 + h[j-k] if 0<=j-k<N else 0)
        
        # For each k in mask, the contribution to grad is:
        # For j in [0,N): h[j+k] (if j+k < N) + h[j-k] (if j-k >= 0)
        
        grad = np.zeros(N)
        k_indices = np.where(mask)[0]
        
        for k in k_indices:
            # h[j+k] for j in [0, N-k) if k >= 0
            if k >= 0:
                if k < N:
                    grad[:N-k] += h[k:]
            # h[j-k] for j in [k, N) if k >= 0, i.e., j-k in [0, N-k)
            if k >= 0:
                if k < N:
                    grad[k:] += h[:N-k]
            # For negative k, but ac is even so k and -k give same
            # Actually in our FFT setup, k ranges 0 to 2N-1
            # ac[k] for k >= N corresponds to negative lags
            # Let me handle it properly
        
        # Actually, let me think again. ac[k] = sum_i h_pad[i] h_pad[i+k]
        # For k in [0, N-1]: ac[k] = sum_{i=0}^{N-1-k} h[i] h[i+k]
        # For k in [N, 2N-1]: let k' = k - N, ac[k] = sum_{i=k'}^{N-1} h[i] h[i-k']... 
        # No, h_pad[i+k] for i+k >= N is 0, so for k >= N, i+k >= N means i >= N-k, 
        # but i < N, so i in [N-k, N-1], and i+k in [N, 2N-1] where h_pad is 0.
        # Wait, h_pad has length 2N, with h_pad[0:N] = h, h_pad[N:2N] = 0.
        # So h_pad[i+k] is nonzero only if i+k < N.
        # For k >= N: i+k >= N for all i >= 0, so ac[k] = 0.
        # That's not right. Let me re-examine.
        
        # Actually, the linear convolution of two length-N sequences has length 2N-1.
        # h_pad has length 2N. The convolution via FFT of length 2N gives circular convolution.
        # Since h_pad[N:] = 0, the circular convolution equals linear convolution for lags 0 to N-1,
        # and for lags N to 2N-1, it wraps around, but since the second half is 0, it should be 0.
        
        # Let me just use a larger FFT to be safe.
        
        # Recompute with proper linear convolution
        n_fft = 4 * N
        h_pad = np.zeros(n_fft)
        h_pad[:N] = h
        H = np.fft.fft(h_pad)
        ac = np.fft.ifft(H * H).real[:2*N-1]  # linear autoconvolution
        
        max_val = np.max(ac)
        threshold = max_val * 0.99
        mask = ac >= threshold
        k_indices = np.where(mask)[0]
        
        grad = np.zeros(N)
        for k in k_indices:
            # ac[k] = sum_{i=max(0,k-N+1)}^{min(N-1, N-1-k+N-1)} ... 
            # Simpler: ac[k] = sum_i h[i] h[i+k] where both indices in [0,N-1]
            # So i in [max(0, -k), min(N-1, N-1-k)]
            i_min = max(0, -k) if k < 0 else 0
            i_max = min(N-1, N-1-k) if k >= 0 else min(N-1, N-1+k)
            # For k >= 0: i in [0, N-1-k]
            # For k < 0: i in [-k, N-1]
            
            if k >= 0:
                i_min = 0
                i_max = N - 1 - k
                if i_max >= i_min:
                    # d(ac[k])/d(h[j]) for j in [i_min, i_max]: h[j+k]
                    # for j in [i_min+k, i_max+k]: h[j-k]  (from the other term)
                    # Actually, ac[k] = sum_i h[i] h[i+k]
                    # d/dh[j] = h[j+k] (if j in [0, N-1-k]) + h[j-k] (if j in [k, N-1])
                    if k < N:
                        # h[j+k] term: j in [0, N-1-k]
                        grad[:N-k] += h[k:]
                        # h[j-k] term: j in [k, N-1]
                        grad[k:] += h[:N-k]
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        h_new = h - lr * grad
        h_new = np.maximum(h_new, 0.0)
        
        score_new = score(h_new)
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
        
        h = h_new
    
    return list(best_h)
