"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _overlap_max(h, N):
    """Compute max over shifts of sum_i h[i] * (1 - h[(i+t) % N]) / N."""
    h = np.asarray(h, dtype=np.float64)
    comp = 1.0 - h
    H = np.fft.fft(h)
    C = np.fft.fft(comp)
    overlap = np.real(np.fft.ifft(np.conj(H) * C))
    return np.max(overlap)


def _project(h, N):
    """Project h to [0,1] with sum exactly N/2."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    h += diff / N
    for _ in range(100):
        h = np.clip(h, 0.0, 1.0)
        diff = N / 2.0 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        if abs(diff) < 1e-12:
            break
    return h


def construct():
    N = 100
    
    # Multi-start optimization with different initializations
    best_h = None
    best_score = np.inf
    
    for seed in range(20):
        np.random.seed(seed)
        
        # Initialize with various patterns
        if seed % 3 == 0:
            # Binary random
            h = np.zeros(N)
            ones = np.random.choice(N, N // 2, replace=False)
            h[ones] = 1.0
        elif seed % 3 == 1:
            # Smooth harmonic
            h = np.full(N, 0.5)
            for k in [1, 2, 3, 5, 7]:
                amp = 0.2 / k
                phase = np.random.uniform(0, 2 * np.pi)
                h += amp * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
            h = _project(h, N)
        else:
            # Block pattern with random phase
            h = np.zeros(N)
            block_size = N // 4
            phase = np.random.randint(0, N)
            for i in range(N):
                if (i - phase) % (2 * block_size) < block_size:
                    h[i] = 1.0
            h = _project(h, N)
        
        h = _project(h, N)
        
        # Optimize using L-BFGS-B with projection
        def objective(x):
            h = _project(x, N)
            return _overlap_max(h, N)
        
        try:
            res = minimize(objective, h, method='L-BFGS-B',
                          bounds=[(0.0, 1.0)] * N,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            h = _project(res.x, N)
            score = _overlap_max(h, N)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
        except Exception:
            pass
    
    # Final refinement with Nelder-Mead on the best solution
    if best_h is not None:
        def objective(x):
            h = _project(x, N)
            return _overlap_max(h, N)
        
        try:
            res = minimize(objective, best_h, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            best_h = _project(res.x, N)
        except Exception:
            pass
    
    # Ensure exact constraints
    best_h = _project(best_h, N)
    return best_h
# EVOLVE-BLOCK-END
