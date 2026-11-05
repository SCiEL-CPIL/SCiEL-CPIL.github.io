import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes for fast convolution
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    def fast_convolve(h):
        """Fast convolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def smooth_objective_and_grad(x, beta=80.0):
        """
        Objective: 2*N * LSE(conv) / S^2
        Gradient derived analytically.
        """
        h = x
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = fast_convolve(h)
        max_c = np.max(conv)
        shifted = conv - max_c
        max_shifted = np.max(shifted)
        
        # Clip to prevent overflow in exp
        if max_shifted > 500:
            return 1e10, np.zeros_like(h)
            
        exp_terms = np.exp(beta * (shifted - max_shifted))
        sum_weights = np.sum(exp_terms)
        
        if sum_weights < 1e-300:
            return 1e10, np.zeros_like(h)
        
        # Log-Sum-Exp approximation of max
        lse = max_c + max_shifted + np.log(sum_weights) / beta
        f = 2.0 * N * lse / (s * s)
        
        if not np.isfinite(f):
            return 1e10, np.zeros_like(h)
        
        # Gradient of LSE: sum_j w_j * (conv[j] - lse) is not quite right for d/dh
        # d/dh_j LSE = sum_k w_k * d conv[k] / d h_j
        # conv[k] = sum_i h[i] h[k-i]
        # d conv[k] / d h_j = h[k-j] + h[j-k] (if in range)
        
        # We need G[j] = sum_k w_k * (h[k-j] + h[j-k])
        # This is cross-correlation of w and h, plus reverse.
        # Since w is symmetric (conv is symmetric for real h), and h is real:
        # G[j] = 2 * sum_k w_k * h[k-j] ? No, careful with indices.
        
        # Let's use the matrix approach for gradient as N=400 is small.
        # M[j, k] = d conv[k] / d h_j
        # conv[k] = sum_i h[i] h[k-i]
        # d/dh_j conv[k] = h[k-j] + h[j-k] (with 0 if index out of bounds)
        
        # We want grad_lse[j] = sum_k w_k * (h[k-j] + h[j-k])
        
        # Let W be weights.
        # Term 1: sum_k W[k] h[k-j]
        # Term 2: sum_k W[k] h[j-k]
        
        # These are correlations.
        # We can compute them via FFT or direct loop.
        # Direct loop for N=400 is 160k ops, fast.
        
        w = exp_terms / sum_weights
        grad_lse = np.zeros(N)
        
        # Vectorized approach using matrix if memory allows, or loop.
        # k ranges 0..2N-2
        # j ranges 0..N-1
        # We need sum over k where k-j is in [0, N-1] and j-k is in [0, N-1]
        
        # Using FFT for speed
        # corr1[j] = sum_k w[k] h[k-j]
        # This is cross-correlation.
        
        # Simpler: Build the matrix A where A[j,k] = d conv[k]/d h_j
        # Then grad_lse = A.T @ w
        
        # Construct A efficiently
        # A[j, k] = 1 if k-j in [0, N-1] -> h[k-j]
        #           + 1 if j-k in [0, N-1] -> h[j-k]
        
        # We can compute G = A.T @ w
        # G[j] = sum_k A[j,k] w[k]
        
        # Let's just use the direct sum with numpy broadcasting for the gradient step,
        # as 400x800 is small.
        
        # k indices
        k_idx = np.arange(2 * N - 1)
        # j indices
        j_idx = np.arange(N)
        
        # Create matrix of indices
        # k_j = k - j
        # k_minus_j = k - j
        # j_minus_k = j - k
        
        # This creates a 400x799 matrix, ~320k elements. Fine.
        k_mat = k_idx[None, :]
        j_mat = j_idx[:, None]
        
        kmj = k_mat - j_mat
        jmk = j_mat - k_mat
        
        # Mask valid indices
        mask_kmj = (kmj >= 0) & (kmj < N)
        mask_jmk = (jmk >= 0) & (jmk < N)
        
        # Extract h values
        # h[kmj] where mask_kmj
        # We can't just index h with a matrix of mixed values easily without flattening.
        
        # Alternative:
        # G1[j] = sum_{k: 0<=k-j<N} w[k] h[k-j]
        # G2[j] = sum_{k: 0<=j-k<N} w[k] h[j-k]
        
        # G1 is cross-correlation of w and h.
        # G2 is cross-correlation of w and h reversed?
        
        # Let's use the property that conv is symmetric.
        # Actually, let's just use the gradient from the current working code's logic
        # but ensure it's correct. The current code uses fast_cross_corr.
        
        # Re-implement fast_cross_corr correctly.
        # G1[j] = sum_{m=0}^{N-1} w[j+m] h[m]  (if w is long enough)
        # w has length 2N-1.
        # For j in 0..N-1:
        # Sum over m such that j+m is in [0, 2N-2]
        
        # We can compute the full cross-correlation of w and h using FFT.
        # cross_corr = ifft(fft(w) * conj(fft(h)))
        # w length 2N-1, h length N.
        # Pad both to 3N.
        
        size_fft = 1
        while size_fft < 3 * N:
            size_fft *= 2
            
        W_fft = np.fft.rfft(w, n=size_fft)
        H_fft = np.fft.rfft(h, n=size_fft)
        
        # Cross-correlation r[k] = sum_j w[j] h[j-k]
        # We want G1[j] = sum_k w[k] h[k-j]
        # Let m = k-j. k = j+m.
        # G1[j] = sum_m w[j+m] h[m]
        
        # This is the cross-correlation of w and h at lag j.
        # r[j] = sum_k w[k+j] h[k] ??
        
        # Let's stick to the matrix A.T @ w which is robust.
        # 400x800 matrix is small.
        
        # We need to sum w[k] * h[k-j] for valid k-j
        # and w[k] * h[j-k] for valid j-k
        
        # Loop over j might be slow in Python, but we can vectorize the sum.
        
        # Let's try the FFT approach from the parent which seemed to work.
        # grad_lse = 2.0 * fast_cross_corr(w, h)
        
        def fast_cross_corr(weights, h):
            # Compute C[j] = sum_m weights[j+m] h[m]
            # Pad weights and h
            w_pad = np.zeros(3 * N)
            w_pad[:len(weights)] = weights
            h_pad = np.zeros(3 * N)
            h_pad[:len(h)] = h
            
            W = np.fft.rfft(w_pad, n=4 * N)
            H = np.fft.rfft(h_pad, n=4 * N)
            
            # Correlation
            # ifft(fft(w) * conj(fft(h))) gives sum_k w[k] h[k-j] ??
            # Let's check documentation.
            # scipy.signal.correlate(w, h)
            # We want sum_m w[j+m] h[m]
            # This is correlate(w, h)[j + len(h) - 1] ?
            
            # Let's just use the matrix multiplication for safety, it's fast enough.
            # 400 * 800 = 320,000 multiplications.
            
            M1 = np.zeros((N, 2*N-1))
            # M1[j, k] = 1 if k-j in [0, N-1]
            # We can construct this by shifting h.
            
            # Actually, just use the explicit sum with numpy
            # G1 = np.zeros(N)
            # for j in range(N):
            #    idx = np.arange(max(0, -j), min(N, 2*N-1-j)) # k indices
            #    # Too slow
            
            # Use the FFT from the parent, it was likely correct.
            # Parent:
            # M = weights[idx[:, None] + idx[None, :]] # 400x400
            # return M @ h
            
            # This computes sum_m weights[j+m] h[m]
            # j in 0..N-1, m in 0..N-1.
            # weights length 2N-1. j+m up to 2N-2. Valid.
            
            idx = np.arange(N)
            M = weights[idx[:, None] + idx[None, :]]
            return M @ h

        grad_lse = 2.0 * fast_cross_corr(w, h)
        
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        
        if f >= 0:
            grad = df_dh
        else:
            grad = -df_dh
            
        return f, grad

    def smooth_objective(x, beta=80.0):
        f, _ = smooth_objective_and_grad(x, beta)
        return f

    def smooth_grad(x, beta=80.0):
        _, grad = smooth_objective_and_grad(x, beta)
        return grad

    rng = np.random.RandomState(42)

    # --- Generate diverse initial points ---
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Random
    for _ in range(20):
        initial_points.append(rng.uniform(-1, 1, N))

    # Symmetric/Asymmetric patterns
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 10]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))
        
    # Random Fourier
    for _ in range(20):
        freqs = rng.randint(1, 50, 10)
        amps = rng.uniform(0.5, 2.0, 10)
        phases = rng.uniform(0, 2 * np.pi, 10)
        h = np.zeros(N)
        for f, a, p in zip(freqs, amps, phases):
            h += a * np.cos(2 * np.pi * f * t + p)
        initial_points.append(h)

    # Sort by value
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # --- Optimize ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break
        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            
        try:
            result = minimize(
                smooth_objective,
                init_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-7}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Final polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                smooth_objective,
                best_h,
                jac=smooth_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-9}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    return best_h
