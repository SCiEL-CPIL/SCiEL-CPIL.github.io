"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 140.0  # wall-clock safety margin before 150s kill


def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    return val if np.isfinite(val) else np.inf


def _build_chirp(params):
    """
    Build h from chirp parameters:
    h[n] = offset + A * cos(2*pi*(f*n + beta*n^2/N))
    params = [A, f, beta, offset]
    """
    A, f, beta, offset = params
    n = np.arange(N, dtype=np.float64)
    phase = 2.0 * np.pi * (f * n + beta * n * n / N)
    h = offset + A * np.cos(phase)
    return h


def _chirp_objective(params):
    h = _build_chirp(params)
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.isfinite(s):
        return 1e10
    conv = np.convolve(h, h)
    max_conv = np.max(conv)
    val = abs(2.0 * N * max_conv / (s * s))
    return val if np.isfinite(val) else 1e10


def _zadoff_chu_init(n=400, root=3, offset=0.0):
    """
    Generate a Zadoff-Chu based initialization.
    ZC sequences have near-constant amplitude and low sidelobe autocorrelation.
    We use the real part of a modulated ZC sequence as a starting point.
    """
    N = n
    # Zadoff-Chu sequence: z[k] = exp(-j*pi*root*k*(k+1)/N) for k=0..N-1
    k = np.arange(N, dtype=np.float64)
    zc = np.exp(-1j * np.pi * root * k * (k + 1) / N)
    # Take real part and add a small chirp modulation to break symmetry
    h = np.real(zc)
    # Add a gentle linear phase ramp to spread energy
    h += 0.1 * np.cos(2 * np.pi * 0.5 * k / N)
    h += offset
    return h


def _subgradient_refine(h, deadline, max_iter=20000, eta0=0.05):
    """
    Subgradient refinement: directly optimize h using the subgradient of
    max(convolve(h,h)) with respect to h, while maintaining sum(h) = 1.
    
    The subgradient of max_k conv[k] w.r.t. h[j] is:
      sum over k where conv[k] is max: (h[k-j] if 0<=k-j<N else 0) + (h[j-k] if 0<=j-k<N else 0)
    """
    best_val = _evaluate(h)
    best_h = h.copy()
    
    # Normalize to sum=1 for stability
    s = np.sum(h)
    if abs(s) < 1e-10:
        return best_h, best_val
    h = h / s
    
    eta = eta0
    j = np.arange(N)
    no_improve = 0
    patience = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        conv = np.convolve(h, h)
        k_star = np.argmax(conv)
        
        # Compute subgradient
        idx1 = k_star - j  # h[k_star - j] for valid indices
        idx2 = j - k_star  # h[j - k_star] for valid indices
        mask1 = (idx1 >= 0) & (idx1 < N)
        mask2 = (idx2 >= 0) & (idx2 < N)
        
        g = np.zeros(N)
        g[mask1] += h[idx1[mask1]]
        g[mask2] += h[idx2[mask2]]
        
        # Normalize gradient
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-15:
            break
        g = g / g_norm
        
        # Update
        h_new = h - eta * g
        # Project back to sum=1
        h_new = h_new - (np.sum(h_new) - 1.0) / N
        
        val_new = _evaluate(h_new)
        
        if val_new < best_val - 1e-12:
            best_val = val_new
            best_h = h_new.copy()
            h = h_new.copy()
            no_improve = 0
            eta = min(eta * 1.02, 0.5)  # slight increase on success
        else:
            no_improve += 1
            eta *= 0.995
            if no_improve > patience:
                # Restart with smaller eta
                eta = max(eta * 0.5, 1e-6)
                no_improve = 0
                if eta < 1e-7:
                    break
    
    return best_h, best_val


def _local_perturb_refine(h, deadline, max_iter=30000):
    """
    Local refinement via random perturbations of individual elements.
    """
    best_val = _evaluate(h)
    best_h = h.copy()
    rng = np.random.default_rng(123)
    
    step_size = 0.3
    no_improve = 0
    patience = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        n_perturb = rng.integers(1, 4)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        val_new = _evaluate(h_new)
        
        if val_new < best_val - 1e-12:
            best_val = val_new
            best_h = h_new.copy()
            no_improve = 0
            step_size = min(step_size * 1.01, 2.0)
        else:
            no_improve += 1
            step_size *= 0.998
            if no_improve > patience:
                step_size *= 0.5
                no_improve = 0
                if step_size < 1e-4:
                    break
    
    return best_h, best_val


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = _evaluate(best_h)  # Should be 2.0
    
    # Phase 1: Chirp parameter optimization with multiple restarts
    bounds = [
        (0.05, 3.0),      # A: amplitude
        (0.0, 5.0),       # f: frequency
        (0.0, 15.0),      # beta: chirp rate
        (-1.0, 2.0),      # offset: DC level
    ]
    
    np.random.seed(42)
    
    # Try several chirp configurations
    chirp_starts = [
        [1.0, 1.0, 1.0, 0.0],
        [0.5, 2.0, 3.0, 0.5],
        [1.5, 0.5, 5.0, -0.3],
        [0.8, 3.0, 2.0, 0.2],
        [1.2, 1.5, 8.0, -0.5],
    ]
    
    for x0 in chirp_starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(
                _chirp_objective,
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_candidate = _build_chirp(res.x)
            val_candidate = _evaluate(h_candidate)
            if val_candidate < best_val:
                best_val = val_candidate
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Phase 2: Zadoff-Chu based initialization
    for root in [3, 5, 7, 9, 11]:
        if time.time() > deadline - 25:
            break
        for offset in [-0.5, 0.0, 0.5]:
            if time.time() > deadline - 20:
                break
            h_zc = _zadoff_chu_init(N, root=root, offset=offset)
            val_zc = _evaluate(h_zc)
            if val_zc < best_val:
                best_val = val_zc
                best_h = h_zc.copy()
    
    # Phase 3: Subgradient refinement on the best candidate
    if time.time() < deadline - 15:
        h_refined, val_refined = _subgradient_refine(
            best_h.copy(), deadline - 10, max_iter=30000, eta0=0.05
        )
        if val_refined < best_val:
            best_val = val_refined
            best_h = h_refined.copy()
    
    # Phase 4: Local perturbation refinement
    if time.time() < deadline - 5:
        h_perturbed, val_perturbed = _local_perturb_refine(
            best_h.copy(), deadline - 2, max_iter=50000
        )
        if val_perturbed < best_val:
            best_val = val_perturbed
            best_h = h_perturbed.copy()
    
    # Final safety check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
