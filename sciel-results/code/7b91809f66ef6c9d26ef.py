"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 18.0  # seconds to spare before 150s kill


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0,N-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds.
    Uses iterative water-filling style correction for better bound handling."""
    if target is None:
        target = N / 2.0
    h = h.copy()
    for _ in range(10):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        step = diff / len(h)
        h += step
        h = np.clip(h, 0.0, 1.0)
    # Final correction on non-saturated entries
    diff = target - h.sum()
    if abs(diff) > 1e-14:
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _sinusoidal_h(A, phi, A2=0.0, phi2=0.0):
    """Generate step heights from sinusoidal model with optional 2nd harmonic.
    h_i = 0.5 + A*sin(2*pi*i/N + phi) + A2*sin(4*pi*i/N + phi2)"""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
    if abs(A2) > 1e-12:
        h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
    return h


def _optimize_sinusoidal():
    """
    Optimize sinusoidal parameters (A, phi, A2, phi2) via:
    1. Coarse grid search over (A, phi)
    2. Local Nelder-Mead refinement on 4 params
    3. Multiple starting points
    """
    best_score = float('inf')
    best_h = None

    # Phase 1: Coarse grid search (1st harmonic only for speed)
    A_grid = np.linspace(0.05, 0.45, 25)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 50)

    best_A = 0.25
    best_phi = 0.0
    best_grid_score = float('inf')

    for A in A_grid:
        for phi in phi_grid:
            h = _sinusoidal_h(A, phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_grid_score:
                best_grid_score = s
                best_A = A
                best_phi = phi

    # Phase 2: Local refinement with Nelder-Mead on 4 params
    def obj_sin(params):
        A, phi, A2, phi2 = params
        A = np.clip(A, 0.0, 0.5)
        A2 = np.clip(A2, -0.3, 0.3)
        h = _sinusoidal_h(A, phi, A2, phi2)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    x0 = np.array([best_A, best_phi, 0.0, 0.0])
    try:
        res = minimize(obj_sin, x0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_grid_score:
            best_grid_score = res.fun
            best_A = float(res.x[0])
            best_phi = float(res.x[1])
            best_A2 = float(res.x[2])
            best_phi2 = float(res.x[3])
    except Exception:
        best_A2 = 0.0
        best_phi2 = 0.0

    # Phase 3: Try multiple starting points for local refinement
    for A_init in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for phi_init in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            for A2_init in [0.0, 0.05, -0.05]:
                x0 = np.array([A_init, phi_init, A2_init, 0.0])
                try:
                    res = minimize(obj_sin, x0, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                    if res.fun < best_grid_score:
                        best_grid_score = res.fun
                        best_A = float(res.x[0])
                        best_phi = float(res.x[1])
                        best_A2 = float(res.x[2])
                        best_phi2 = float(res.x[3])
                except Exception:
                    pass

    # Build best sinusoidal h
    h_sin = _sinusoidal_h(best_A, best_phi, best_A2, best_phi2)
    h_sin = np.clip(h_sin, 0.0, 1.0)
    h_sin = _fix_sum(h_sin)
    best_score = _score(h_sin)
    best_h = h_sin.copy()

    return best_score, best_h, (best_A, best_phi, best_A2, best_phi2)


def _smooth_max_and_grad(h, beta=500.0):
    """
    Compute smooth max of overlap values and its gradient.
    overlap[i] = h[i] * (1 - h[i+1]) for i in [0, N-2], 0 for i = N-1
    Uses log-sum-exp for numerical stability.
    """
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    
    m = np.max(overlap)
    # log-sum-exp for numerical stability
    exp_terms = np.exp(beta * (overlap - m))
    log_sum = np.log(np.sum(exp_terms))
    smooth_val = m + log_sum / beta
    
    # Gradient: d(smooth_max)/d(overlap[i]) = exp_terms[i] / sum(exp_terms)
    weights = exp_terms / np.sum(exp_terms)
    
    # d(overlap[i])/d(h[j]) for i in [0, N-2]
    # overlap[i] = h[i] * (1 - h[i+1])
    # d(overlap[i])/d(h[i]) = 1 - h[i+1]
    # d(overlap[i])/d(h[i+1]) = -h[i]
    grad = np.zeros(n)
    for i in range(n - 1):
        w = weights[i]
        grad[i] += w * (1.0 - h[i+1])
        grad[i+1] += w * (-h[i])
    # overlap[N-1] = 0, no contribution
    
    return smooth_val, grad


def _optimize_direct_lbfgsb(h0, max_iter=3000, beta=500.0):
    """
    Optimize the full vector h directly using SLSQP with analytical gradients.
    Uses log-sum-exp approximation to make the objective differentiable.
    SLSQP handles the sum constraint properly.
    """
    def obj_and_grad(h):
        h = np.clip(h, 0.0, 1.0)
        val, grad = _smooth_max_and_grad(h, beta)
        return val, grad

    def obj(h):
        return obj_and_grad(h)[0]

    def grad(h):
        return obj_and_grad(h)[1]

    constraints = [{'type': 'eq', 'fun': lambda h: h.sum() - N / 2.0,
                    'jac': lambda h: np.ones(N)}]

    try:
        res = minimize(obj, h0, jac=grad, method='SLSQP',
                       bounds=[(0.0, 1.0)] * N,
                       constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'maxls': 200})
        h_opt = np.clip(res.x, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return _score(h_opt), h_opt
    except Exception:
        return float('inf'), h0


def _optimize_multiblock(num_blocks=3):
    """
    Optimize a multi-block step function with num_blocks equal-width segments.
    Each segment has a height, and we optimize those heights.
    """
    best_score = float('inf')
    best_h = None

    # Equal-width blocks
    block_width = N / num_blocks
    # Initial guess: alternating high/low
    heights = np.zeros(num_blocks)
    for i in range(num_blocks):
        heights[i] = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / num_blocks)

    def obj_heights(hs):
        hs = np.clip(hs, 0.0, 1.0)
        # Build full h vector
        h = np.zeros(N)
        for i in range(num_blocks):
            start = int(i * block_width)
            end = int((i + 1) * block_width)
            h[start:end] = hs[i]
        h = _fix_sum(h)
        return _score(h)

    # Try multiple initializations
    for init_type in range(5):
        if init_type == 0:
            h0 = np.zeros(num_blocks)
            for i in range(num_blocks):
                h0[i] = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / num_blocks)
        elif init_type == 1:
            h0 = np.full(num_blocks, 0.5)
            h0[0] = 0.7
            h0[-1] = 0.3
        elif init_type == 2:
            h0 = np.linspace(0.3, 0.7, num_blocks)
        elif init_type == 3:
            h0 = np.full(num_blocks, 0.5)
            h0[::2] = 0.6
            h0[1::2] = 0.4
        else:
            h0 = np.random.uniform(0.2, 0.8, num_blocks)

        try:
            res = minimize(obj_heights, h0, method='Nelder-Mead',
                           bounds=[(0.0, 1.0)] * num_blocks,
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            hs_opt = np.clip(res.x, 0.0, 1.0)
            h = np.zeros(N)
            for i in range(num_blocks):
                start = int(i * block_width)
                end = int((i + 1) * block_width)
                h[start:end] = hs_opt[i]
            h = _fix_sum(h)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            pass

    if best_h is None:
        # Fallback
        h = np.full(N, 0.5)
        return _score(h), h

    return best_score, best_h


def _local_search_polish(h, max_iters=5000):
    """
    Direct local search: try adjusting individual entries or swapping adjacent pairs.
    Uses adaptive step sizes that shrink over time for fine-tuning.
    """
    h = h.copy()
    current_score = _score(h)
    improved = True
    iters = 0
    # Start with larger steps, shrink over time
    base_deltas = [0.02, -0.02, 0.01, -0.01, 0.005, -0.005, 0.002, -0.002, 0.001, -0.001, 0.0005, -0.0005]

    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Adaptive scaling: shrink steps as we iterate
        scale = 1.0 if iters < 100 else max(0.1, 1.0 - (iters - 100) / 1000.0)
        deltas = [d * scale for d in base_deltas]

        # Try small adjustments to each entry
        for i in range(N):
            for delta in deltas:
                new_val = h[i] + delta
                if new_val < 0 or new_val > 1:
                    continue
                h_trial = h.copy()
                h_trial[i] = new_val
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-14:
                    h = h_trial
                    current_score = s
                    improved = True
                    break

        # Try swapping adjacent entries
        for i in range(N - 1):
            if abs(h[i] - h[i+1]) < 1e-10:
                continue
            h_trial = h.copy()
            h_trial[i], h_trial[i+1] = h_trial[i+1], h_trial[i]
            h_trial = _fix_sum(h_trial)
            s = _score(h_trial)
            if s < current_score - 1e-14:
                h = h_trial
                current_score = s
                improved = True

    return current_score, h


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)

    # Phase 1: Sinusoidal optimization
    best_score = float('inf')
    best_h = None

    if time.time() < deadline:
        s_sin, h_sin, _ = _optimize_sinusoidal()
        if s_sin < best_score:
            best_score = s_sin
            best_h = h_sin.copy()

    # Phase 2: Multi-block optimization (3 and 4 blocks)
    for nb in [3, 4]:
        if time.time() > deadline - 10.0:
            break
        s_mb, h_mb = _optimize_multiblock(nb)
        if s_mb < best_score:
            best_score = s_mb
            best_h = h_mb.copy()

    # Phase 3: Direct SLSQP optimization from best candidate with analytical gradients
    if best_h is not None and time.time() < deadline - 20.0:
        s_direct, h_direct = _optimize_direct_lbfgsb(best_h, max_iter=3000, beta=500.0)
        if s_direct < best_score:
            best_score = s_direct
            best_h = h_direct.copy()

    # Phase 4: Reduced local search polish
    if best_h is not None and time.time() < deadline - 10.0:
        s_polish, h_polish = _local_search_polish(best_h, max_iters=2000)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Phase 5: Final high-precision SLSQP polish with tighter tolerance
    if best_h is not None and time.time() < deadline - 3.0:
        s_final, h_final = _optimize_direct_lbfgsb(best_h, max_iter=5000, beta=1000.0)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Phase 6: Ultra-high-beta final polish if time permits
    if best_h is not None and time.time() < deadline - 1.0:
        s_ultra, h_ultra = _optimize_direct_lbfgsb(best_h, max_iter=3000, beta=2000.0)
        if s_ultra < best_score:
            best_score = s_ultra
            best_h = h_ultra.copy()

    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
