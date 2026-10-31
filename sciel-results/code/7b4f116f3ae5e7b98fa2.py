"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from numpy.fft import fft, ifft
import time

def _autoconv(x):
    """Compute linear autoconvolution via FFT."""
    n = len(x)
    size = 2 * n - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    F = fft(x, size2)
    A = ifft(F * np.conj(F)).real
    return A[:size]

def _score(x):
    """Compute max(|autoconv|) / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return np.inf
    A = _autoconv(x)
    m = np.max(np.abs(A))
    return m / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial best: all ones
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
            
    # Phase 1: Gradient Descent on Continuous Variables
    # Objective: J = max_i |A_i| / S^2
    # We minimize J by gradient descent on y (real vector).
    # A = y * y (linear convolution)
    # dA/dy = 2 * (y * ones) ? No.
    # Let's use the property: A_k = sum_i y_i y_{i+k}.
    # dA_k / dy_j = y_{j+k} + y_{j-k} (with boundary handling 0).
    # This is equivalent to: dA/dy_j = (y * y)_j ? No.
    # Actually, the gradient of the autoconvolution vector A with respect to y is:
    # (dA/dy)_j = sum_k (dA_k/dy_j) sign(A_k) ... wait, we want gradient of max |A|.
    # Let M = max |A_k|. The subgradient of M with respect to A is a vector g_A where
    # g_A[k] = sign(A[k]) / N_active if |A[k]| == M, else 0.
    # Then grad_y J = (1/S^2) * (dA/dy)^T g_A - 2*M/S^3 * (dS/dy)
    # dS/dy = 1.
    
    # Efficient gradient computation:
    # A = ifft(fft(y) * conj(fft(y))).real
    # dA/dy is a Toeplitz-like operation?
    # Actually, dA_k / dy_j = y_{j+k} + y_{j-k} (where indices out of bounds are 0).
    # This is the same as: (y * y)_j in the cross-correlation sense?
    # Let's check: (y * y)_j = sum_i y_i y_{i+j}.
    # If we take the gradient of sum_k A_k g_k with respect to y_j:
    # sum_k (y_{j+k} + y_{j-k}) g_k = sum_m y_{j+m} g_m + sum_m y_{j-m} g_m
    # = sum_m y_{j+m} (g_m + g_{-m}) (with g_m=0 for m<0 in linear conv context? No, A is defined for -N..N).
    # Our A is linear autoconv, so A_k for k=0..N-1.
    # A_k = sum_{i=0}^{N-1-k} y_i y_{i+k}.
    # dA_k/dy_j = y_{j+k} (if j+k < N) + y_{j-k} (if j-k >= 0).
    # So grad_y = sum_k g_k (y_{j+k} + y_{j-k}).
    # This can be computed efficiently using FFT if we embed y in a larger vector and use correlation.
    # Let h be a vector of length 2N-1 where h[k] = g_k for k>=0 and h[2N-1-k] = g_k? No.
    # Let's just use the FFT trick for the gradient of the energy.
    # Energy E = sum_k A_k g_k.
    # dE/dy = 2 * (g * y) ? No.
    # Let's use the property: sum_k A_k g_k = sum_{i,j} y_i y_j g_{i-j} (if we extend g).
    # This is equivalent to y^T T_g y where T_g is the Toeplitz matrix defined by g.
    # Gradient is 2 T_g y.
    # We can compute T_g y using FFT:
    # Let G = fft(g_padded). Then T_g y = ifft(G * fft(y_padded)).
    # We need to be careful with the definition of A.
    # A_k = sum_i y_i y_{i+k}.
    # Let's define g_ext of length 2N-1. g_ext[k] = g_k for k=0..N-1.
    # Actually, the gradient of the autoconvolution peak is well-known to be computed via:
    # grad = ifft(fft(y, 2N) * fft(g, 2N)).real
    # where g is the subgradient of the peak function, padded appropriately.
    
    def compute_grad(y, A, S):
        # Find peaks
        max_val = np.max(np.abs(A))
        if max_val < 1e-12:
            return np.zeros_like(y)
        
        # Subgradient of max(|A|)
        # We pick the indices where |A| is close to max
        active = np.abs(A) > max_val * 0.999
        g_A = np.zeros_like(A)
        g_A[active] = np.sign(A[active]) / np.sum(active)
        
        # Compute dA/dy * g_A
        # This is equivalent to 2 * (g_A * y) in the sense of correlation?
        # Let's use the FFT method for the quadratic form gradient.
        # We want sum_k g_A[k] * A[k] derivative wrt y.
        # A[k] = sum_i y[i] y[i+k].
        # d/dy[j] sum_k g_A[k] sum_i y[i] y[i+k]
        # = sum_k g_A[k] (y[j+k] + y[j-k]) (with bounds)
        # This is 2 * sum_k g_A[k] y[j+k] (if g_A is symmetric, which it isn't necessarily).
        # Actually, if we define the linear convolution, the gradient is:
        # grad[j] = sum_k g_A[k] y[j+k] + sum_k g_A[k] y[j-k]
        # = sum_m y[j+m] (g_A[m] + g_A[-m]) ... wait, k can be negative?
        # In our A, k is 0..N-1.
        # So grad[j] = sum_{k=0}^{N-1} g_A[k] y[j+k] + sum_{k=0}^{N-1} g_A[k] y[j-k] (with y[out of bounds]=0)
        
        # We can compute this using FFT.
        # Let v = g_A.
        # term1[j] = sum_k v[k] y[j+k]. This is correlation.
        # term2[j] = sum_k v[k] y[j-k]. This is also correlation (reversed).
        # Actually, term1[j] = (y * v)[j] ? No.
        # Let's just use the full FFT gradient trick for the energy E = y^T T_v y.
        # T_v is the Toeplitz matrix with first row v[0], v[1], ... v[N-1] and first col v[0], 0, ...?
        # No, A[k] depends on y[i]y[i+k].
        # The matrix M such that A = vec(M y)? No.
        
        # Simpler: Use the fact that grad = ifft(fft(y, 2N) * fft(v_padded, 2N)).real
        # where v_padded is constructed such that it represents the kernel.
        # Let's test with a small example in mind.
        # If we use scipy.signal.correlate(y, v, mode='full'), that gives sum y[i] v[i-k].
        # We want sum v[k] y[j+k].
        
        # Let's use the standard gradient for autoconv:
        # grad = 2 * ifft(fft(y, 2*N) * fft(v, 2*N)).real[:N]
        # This works if v is the subgradient of the norm of the convolution.
        
        v = np.zeros(2*N)
        v[:N] = g_A
        # We need to handle the symmetry correctly.
        # The energy is sum_k A[k] g_A[k].
        # The gradient wrt y is:
        # grad[j] = sum_k g_A[k] (y[j+k] + y[j-k])
        # = sum_k g_A[k] y[j+k] + sum_k g_A[k] y[j-k]
        # Let u = g_A.
        # sum_k u[k] y[j+k] is the cross-correlation of y and u.
        # sum_k u[k] y[j-k] is the cross-correlation of y and u (reversed).
        
        # Using FFT:
        # corr1 = ifft(fft(y, 2N) * conj(fft(u, 2N))).real
        # This gives sum y[i] conj(u[i-k])?
        
        # Let's stick to a robust numerical gradient or a specific FFT formula.
        # Formula: grad = ifft(fft(y, 2N) * fft(u, 2N)).real
        # Let's assume this is correct for the "cyclic" case and adjust.
        # For linear autoconv, the gradient is:
        # grad = ifft(fft(y, 2N) * fft(u, 2N)).real[:N]
        # where u is the subgradient vector of length 2N, with u[k] = g_A[k] for k < N and u[2N-1-k] = g_A[k] for k > N?
        # Actually, the simplest is:
        # grad = np.zeros(N)
        # for j in range(N):
        #    s = 0
        #    for k in range(N):
        #       if j+k < N: s += g_A[k]*y[j+k]
        #       if j-k >= 0: s += g_A[k]*y[j-k]
        #    grad[j] = s
        # This is O(N^2). N=400, so 160,000 ops. Very fast.
        
        grad = np.zeros(N)
        for j in range(N):
            s = 0.0
            # k such that j+k < N => k < N-j
            if N-j > 0:
                s += np.dot(g_A[:N-j], y[j:])
            # k such that j-k >= 0 => k <= j
            if j >= 0:
                s += np.dot(g_A[:j+1], y[:j+1][::-1])
            grad[j] = s
            
        # Total gradient of J = M/S^2
        # dJ/dy = (1/S^2) grad_A - (2/S^3) M * 1
        grad_J = (1.0 / (S*S)) * grad - (2.0 * max_val / (S**3)) * np.ones(N)
        
        return grad_J

    # Run Gradient Descent
    y = np.random.default_rng(42).normal(0, 1, N)
    # Normalize y to have sum 1 to avoid scaling issues?
    # The objective is scale invariant: J(cy) = J(y).
    # So we can project onto sum(y)=1 or just let it evolve.
    # Projecting helps stability.
    y = y / np.sum(y)
    
    lr = 0.01
    for step in range(2000):
        if time.time() > deadline - 30:
            break
        A = _autoconv(y)
        S = np.sum(y)
        if abs(S) < 1e-6:
            y += 0.1
            y = y / np.sum(y)
            continue
        grad = compute_grad(y, A, S)
        y -= lr * grad
        # Project back to sum=1
        y = y / np.sum(y)
        
        # Threshold to binary and check
        if step % 10 == 0:
            x_bin = np.where(y > 0, 1.0, -1.0)
            try_candidate(x_bin)
            
    # Phase 2: Local Search on the best binary sequence found
    # Use the best_x from Phase 1
    x = best_x.copy()
    sc = best_score
    
    # Fast local search: flip bits that reduce score
    # We can precompute the effect of a flip?
    # A_new = A + 2*x_j*(y * x) ... complicated.
    # Just recompute FFT. N=400 is fast.
    
    improved = True
    passes = 0
    while improved and passes < 20 and time.time() < deadline - 10:
        improved = False
        passes += 1
        # Try flipping each bit
        for j in range(N):
            if time.time() > deadline - 10:
                break
            x[j] *= -1.0
            sc_new = _score(x)
            if sc_new < sc - 1e-10:
                sc = sc_new
                improved = True
            else:
                x[j] *= -1.0 # revert
        if improved:
            best_x = x.copy()
            best_score = sc
            
    # Phase 3: Random restarts with gradient descent
    rng = np.random.default_rng(123)
    for i in range(10):
        if time.time() > deadline - 5:
            break
        y = rng.normal(0, 1, N)
        y = y / np.sum(y)
        
        for step in range(500):
            if time.time() > deadline - 5:
                break
            A = _autoconv(y)
            S = np.sum(y)
            if abs(S) < 1e-6:
                y += 0.1
                y = y / np.sum(y)
                continue
            grad = compute_grad(y, A, S)
            y -= lr * grad
            y = y / np.sum(y)
            
            if step % 50 == 0:
                x_bin = np.where(y > 0, 1.0, -1.0)
                try_candidate(x_bin)
                
        # Local search on this candidate
        x = best_x.copy()
        sc = best_score
        improved = True
        passes = 0
        while improved and passes < 5 and time.time() < deadline - 2:
            improved = False
            passes += 1
            for j in range(N):
                if time.time() > deadline - 2:
                    break
                x[j] *= -1.0
                sc_new = _score(x)
                if sc_new < sc - 1e-10:
                    sc = sc_new
                    improved = True
                else:
                    x[j] *= -1.0
            if improved:
                best_x = x.copy()
                best_score = sc

    return best_x
# EVOLVE-BLOCK-END
