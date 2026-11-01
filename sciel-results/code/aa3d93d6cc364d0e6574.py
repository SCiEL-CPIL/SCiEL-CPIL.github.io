"""Uncertainty bound optimization via stochastic local search on a reparameterized sphere."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 200.0

# Precompute integral table I[n] = integral x^{2n} e^{-x^2} dx = (2n)!/(4^n n!) * sqrt(pi)
_I_TABLE = np.zeros(28)
_sp = np.sqrt(np.pi)
_I_TABLE[0] = _sp
for _n in range(1, 28):
    _I_TABLE[_n] = _I_TABLE[_n - 1] * (2 * _n - 1) / (4 * _n)

_SIGNS = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

def _compute_p(params):
    """Compute polynomial coefficients p_k from (a0, a4, a8)."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0
    p = np.zeros(7)
    p[0] = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p[1] = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p[2] = a4 + 210.0 * a8 + 13860.0 * a12
    p[3] = -28.0 * a8 - 6930.0 * a12
    p[4] = a8 + 1155.0 * a12
    p[5] = -66.0 * a12
    p[6] = a12
    return p

def _bound(params):
    """Compute the uncertainty product bound for given coefficients."""
    p = _compute_p(params)
    
    # Compute moments m[k] = sum_{i,j} p_i p_j I[k+i+j] for k=0,1,2
    m = np.zeros(3)
    for k in range(3):
        m[k] = 0.0
        for i in range(7):
            pi = p[i]
            if pi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 28:
                    m[k] += pi * p[j] * _I_TABLE[idx]
    
    # Fourier: q_k = p_k * (-1)^k
    q = p * _SIGNS
    
    # n[k] = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for k in range(3):
        n[k] = 0.0
        for i in range(7):
            qi = q[i]
            if qi == 0.0:
                continue
            for j in range(7):
                idx = k + i + j
                if idx < 28:
                    n[k] += qi * q[j] * _I_TABLE[idx]
        n[k] *= 2.0 * np.pi
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    bound = np.sqrt(vf * vF)
    return bound

def construct():
    # Seed point from the current best
    seed = np.array([-1.13, 0.04, 0.000308642])
    best_params = seed.copy()
    best_val = _bound(best_params)
    
    # Generate diverse starting points
    rng = np.random.default_rng(12345)
    
    # Start with the seed and nearby points
    starts = [seed.copy()]
    
    # Add points around the seed with varying scales
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        for _ in range(5):
            starts.append(seed + rng.normal(0, scale, size=3))
    
    # Add some broader exploration
    for _ in range(10):
        starts.append(rng.normal([-1.0, 0.05, 0.0003], [0.5, 0.05, 0.0005]))
    
    # Evaluate all starts and keep the best
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        val = _bound(start)
        if val < best_val:
            best_val = val
            best_params = start.copy()
    
    # Stochastic local search with adaptive step sizes
    step_sizes = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    current_step_idx = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < _DEADLINE - 10:
        if no_improve_count >= max_no_improve:
            # Decrease step size
            if current_step_idx < len(step_sizes) - 1:
                current_step_idx += 1
            no_improve_count = 0
            continue
        
        step = step_sizes[current_step_idx]
        # Generate a random perturbation
        delta = rng.normal(0, step, size=3)
        candidate = best_params + delta
        
        val = _bound(candidate)
        if val < best_val:
            best_val = val
            best_params = candidate.copy()
            no_improve_count = 0
            # Occasionally increase step size if we've been improving
            if no_improve_count == 0 and current_step_idx > 0 and rng.random() < 0.1:
                current_step_idx -= 1
        else:
            no_improve_count += 1
    
    # Final polish with very small steps
    if time.time() < _DEADLINE - 5:
        for _ in range(10000):
            if time.time() > _DEADLINE - 5:
                break
            delta = rng.normal(0, 1e-8, size=3)
            candidate = best_params + delta
            val = _bound(candidate)
            if val < best_val:
                best_val = val
                best_params = candidate.copy()
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
