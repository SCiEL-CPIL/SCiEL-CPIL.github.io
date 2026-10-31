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
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    # Pre-compute FFT size
    size = 2 * N - 1
    size2 = 1
    while size2 < size:
        size2 *= 2
    
    best_x = None
    best_score = np.inf
    
    def try_candidate(x):
        nonlocal best_x, best_score
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()
    
    # Phase 1: Structured Candidates
    # 1. All ones
    try_candidate(np.ones(N))
    
    # 2. Alternating
    alt = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    try_candidate(alt)
    
    # 3. Kronecker sequences (fractional parts of sqrt(2)*n)
    for alpha in [np.sqrt(2) - 1, np.sqrt(3) - 1, (np.sqrt(5) - 1) / 2, np.pi - 3]:
        if time.time() > deadline:
            break
        n = np.arange(N)
        frac = np.mod(alpha * n, 1.0)
        x = np.where(frac < 0.5, 1.0, -1.0)
        try_candidate(x)
    
    # 4. Random +/- 1 with various biases
    rng = np.random.default_rng(42)
    for bias in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        p = bias
        x = rng.choice([-1.0, 1.0], size=N, p=[1-p, p])
        try_candidate(x)
    
    # 5. Block structures
    for block_len in [2, 4, 8, 16, 32, 50, 100]:
        if time.time() > deadline:
            break
        n_blocks = N // block_len
        x = np.zeros(N)
        for b in range(n_blocks):
            sign = 1.0 if b % 2 == 0 else -1.0
            x[b*block_len:(b+1)*block_len] = sign
        if n_blocks * block_len < N:
            x[n_blocks*block_len:] = 1.0
        try_candidate(x)
    
    # 6. Sine wave thresholded
    for freq in [1, 2, 3, 5, 8, 13, 21]:
        if time.time() > deadline:
            break
        n = np.arange(N)
        x = np.where(np.sin(2 * np.pi * freq * n / N) > 0, 1.0, -1.0)
        try_candidate(x)
    
    # Phase 2: Gradient-based optimization on continuous relaxation, then threshold
    # This is a structural change: use gradient descent on a smooth proxy
    # f(x) = max_k |A_k|^p / (sum x)^2 with p=2 for smoothness
    # Then project to +/- 1
    
    def _smooth_score(x_cont):
        """Smooth proxy: sum(|A|^2) / (sum x)^2"""
        s = np.sum(x_cont)
        if abs(s) < 1e-12:
            return np.inf
        F = fft(x_cont, size2)
        A = ifft(F * np.conj(F)).real[:size]
        return np.sum(A * A) / (s * s)
    
    def _gradient_smooth(x_cont):
        """Gradient of sum(|A|^2) / (sum x)^2"""
        s = np.sum(x_cont)
        if abs(s) < 1e-12:
            return np.zeros_like(x_cont)
        F = fft(x_cont, size2)
        A = ifft(F * np.conj(F)).real[:size]
        # d(A_k)/dx_j = x_{k-j} for linear autoconv
        # Gradient of sum(A^2) = 2 * A * dA/dx
        # This is complex; use numerical gradient or FFT-based
        # Actually, for autoconv A = x * x, dA/dx = 2x (shifted)
        # Let's use: grad_j = 2 * sum_k A_k * x_{k-j} / s^2 - 2 * sum(A^2) * x_j / s^3
        # This is O(N^2) which is 160000 ops, acceptable
        grad = np.zeros(N)
        # A_k = sum_j x_j x_{k-j}
        # d(A_k)/dx_j = x_{k-j} + x_{k-j} = 2*x_{k-j} (if k-j in range)
        # sum_k A_k * dA_k/dx_j = 2 * sum_k A_k * x_{k-j}
        # This is 2 * (A * x)_j where * is cross-correlation
        # (A * x)_j = sum_k A_k x_{k-j}
        # We can compute this via FFT
        Ax = ifft(fft(A, size2) * fft(x_cont, size2)).real[:N]
        # Actually need to be careful with indexing
        # Let me just use finite differences for simplicity in gradient
        eps = 1e-6
        grad = np.zeros(N)
        for i in range(0, N, 4):  # Sample every 4th for speed
            xp = x_cont.copy()
            xp[i] += eps
            xm = x_cont.copy()
            xm[i] -= eps
            grad[i] = (_smooth_score(xp) - _smooth_score(xm)) / (2 * eps)
        # Interpolate
        for i in range(1, N, 4):
            grad[i] = grad[i-1]
        for i in range(2, N, 4):
            grad[i] = grad[i-1]
        for i in range(3, N, 4):
            grad[i] = grad[i-1]
        return grad
    
    # Actually, let's do a simpler approach: use the fact that for binary sequences,
    # we can do coordinate descent more efficiently
    
    # Phase 2: Iterated Local Search with restarts
    # Key insight: the objective is not convex, so we need multiple restarts
    
    def _local_search(x_init, max_iter=2000):
        """Hill climbing with single-bit flips."""
        x = x_init.copy()
        F = fft(x, size2)
        current_sum = np.sum(x)
        current_score = _score(x)
        
        improved = True
        iterations = 0
        while improved and iterations < max_iter and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            # Try flipping each bit
            for i in range(N):
                old_val = x[i]
                new_val = -old_val
                delta_sum = -2.0 * old_val
                new_sum = current_sum + delta_sum
                
                if abs(new_sum) < 1e-12:
                    continue
                
                # Update F incrementally
                delta_x = np.zeros(size2)
                delta_x[i] = -2.0 * old_val
                F_new = F + fft(delta_x, size2)
                
                A = ifft(F_new * np.conj(F_new)).real[:size]
                m_new = np.max(np.abs(A))
                score_new = m_new / (new_sum * new_sum)
                
                if score_new < current_score - 1e-12:
                    x[i] = new_val
                    F = F_new
                    current_sum = new_sum
                    current_score = score_new
                    improved = True
                else:
                    # Revert
                    pass
        
        return x, current_score
    
    # Run multiple restarts of local search
    rng2 = np.random.default_rng(123)
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate diverse starting points
        if r == 0:
            x_start = best_x.copy()
        elif r % 3 == 1:
            # Random with bias
            bias = 0.5 + 0.3 * rng2.random()
            x_start = rng2.choice([-1.0, 1.0], size=N, p=[1-bias, bias])
        elif r % 3 == 2:
            # Perturb best
            x_start = best_x.copy()
            n_flips = rng2.integers(10, 50)
            flip_idx = rng2.choice(N, size=n_flips, replace=False)
            x_start[flip_idx] *= -1
        else:
            # Block structure with random block sizes
            x_start = np.zeros(N)
            pos = 0
            while pos < N:
                blen = rng2.integers(1, 20)
                sign = rng2.choice([-1.0, 1.0])
                x_start[pos:pos+blen] = sign
                pos += blen
        
        x_ls, sc_ls = _local_search(x_start)
        try_candidate(x_ls)
    
    # Phase 3: Simulated Annealing from best
    x = best_x.copy()
    current_score = best_score
    F = fft(x, size2)
    current_sum = np.sum(x)
    
    T = 0.05
    T_min = 0.001
    cooling_rate = 0.999
    
    sa_best_x = x.copy()
    sa_best_score = current_score
    
    while T > T_min and time.time() < deadline - 5.0:
        for _ in range(20):
            if time.time() > deadline - 5.0:
                break
            
            n_flips = rng2.integers(1, 4)
            flip_idx = rng2.choice(N, size=n_flips, replace=False)
            
            old_vals = x[flip_idx]
            delta_sum = -2.0 * np.sum(old_vals)
            new_sum = current_sum + delta_sum
            
            if abs(new_sum) < 1e-12:
                continue
            
            x[flip_idx] *= -1.0
            
            delta_x = np.zeros(size2)
            delta_x[flip_idx] = -2.0 * old_vals
            F += fft(delta_x, size2)
            
            A = ifft(F * np.conj(F)).real[:size]
            m_new = np.max(np.abs(A))
            score_new = m_new / (new_sum * new_sum)
            
            if score_new < current_score:
                current_score = score_new
                current_sum = new_sum
                if score_new < sa_best_score:
                    sa_best_score = score_new
                    sa_best_x = x.copy()
            else:
                delta_E = score_new - current_score
                accept = np.exp(-delta_E / T) > rng2.random()
                if accept:
                    current_score = score_new
                    current_sum = new_sum
                else:
                    x[flip_idx] *= -1.0
                    F -= fft(delta_x, size2)
                    current_sum = new_sum - delta_sum
        
        T *= cooling_rate
    
    if sa_best_score < best_score:
        best_x = sa_best_x
        best_score = sa_best_score
    
    return best_x
# EVOLVE-BLOCK-END
