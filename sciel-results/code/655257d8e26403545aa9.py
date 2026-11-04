import numpy as np
from scipy.signal import fftconvolve

N = 400
w = 1.0 / 200.0

def build_haar_basis(N):
    W = np.zeros((N, N))
    W[0, :] = 1.0 / np.sqrt(N)
    
    idx = 1
    block_size = 2
    while block_size <= N:
        num_blocks = N // block_size
        for k in range(num_blocks):
            start = k * block_size
            end = start + block_size
            vec = np.zeros(N)
            vec[start:start+block_size//2] = 1.0
            vec[start+block_size//2:end] = -1.0
            vec /= np.sqrt(block_size)
            W[idx, :] = vec
            idx += 1
        block_size *= 2
        
    # Complete the basis with random orthogonal vectors
    if idx < N:
        # Generate random vectors
        for _ in range(N - idx):
            v = np.random.randn(N)
            # Orthogonalize against existing basis
            for j in range(idx):
                proj = np.dot(v, W[j, :])
                v -= proj * W[j, :]
            norm = np.linalg.norm(v)
            if norm < 1e-8:
                # Retry if degenerate
                v = np.random.randn(N)
                for j in range(idx):
                    proj = np.dot(v, W[j, :])
                    v -= proj * W[j, :]
                norm = np.linalg.norm(v)
            v /= norm
            W[idx, :] = v
            idx += 1
            
    return W

def construct():
    W = build_haar_basis(N)
    
    # Initialize c such that s = W c is close to 1
    s_init = np.ones(N)
    c = W.T @ s_init
    
    best_s = s_init.copy()
    best_val = np.inf
    
    # Hyperparameters
    lr = 0.01
    mu = 10.0 # Penalty weight
    
    def compute_score_and_grad(s):
        # Compute autoconvolution
        n_fft = 2 * N
        S = np.fft.rfft(s, n_fft)
        r = np.fft.irfft(S * np.conj(S), n_fft)[:N]
        
        # Score is max abs value
        max_val = np.max(np.abs(r))
        k_max = np.argmax(np.abs(r))
        r_max = r[k_max]
        
        # Subgradient of |r[k_max]|
        # r[k] = sum_i s[i] s[i+k]
        # d r[k_max] / d s[i] = s[i+k_max] (if i < N-k_max) + s[i-k_max] (if i >= k_max)
        sg_auto = np.zeros(N)
        sign_r = np.sign(r_max)
        
        # Vectorized subgradient calculation
        # Term 1: s[i+k]
        if k_max > 0:
            sg_auto[:N-k_max] += s[k_max:]
        # Term 2: s[i-k]
        if k_max > 0:
            sg_auto[k_max:] += s[:N-k_max]
        else:
            # k=0, r[0] = sum s[i]^2
            # d/ds[i] = 2 s[i]
            sg_auto = 2 * s
            # Note: The formula above for k=0 gives s[i]+s[i] = 2s[i]. Correct.
            
        sg_auto *= sign_r
        
        # Penalty gradient: d/ds [ sum (s^2 - 1)^2 ] = 4 s (s^2 - 1)
        sg_pen = 4.0 * s * (s**2 - 1.0)
        
        total_sg = sg_auto + mu * sg_pen
        return max_val, total_sg

    for iteration in range(1000):
        s = W @ c
        val, grad_s = compute_score_and_grad(s)
        
        if val < best_val:
            best_val = val
            best_s = s.copy()
            
        # Gradient in c space
        grad_c = W.T @ grad_s
        
        # Update
        c -= lr * grad_c
        
        # Optional: Decay learning rate
        if iteration % 100 == 0:
            lr *= 0.9

    s_final = np.sign(W @ c)
    s_final[s_final == 0] = 1
    return s_final
