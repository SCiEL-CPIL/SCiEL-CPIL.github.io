"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_c3_bound(h):
    """
    Compute the C3 autoconvolution upper bound.
    For a ±1 sequence, this is based on the third moment of the autocorrelation.
    """
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    
    sidelobes = acf[1:]
    c3 = np.sum(np.abs(sidelobes) ** 3) / (N ** 3)
    return c3


def _continuous_objective(x, n_fft):
    """
    Continuous objective: minimize C3-like bound for h = tanh(x).
    x is unconstrained real vector; h = tanh(x) ∈ (-1, 1).
    """
    h = np.tanh(x)
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft).real[:N]
    sidelobes = acf[1:]
    c3 = np.sum(np.abs(sidelobes) ** 3) / (N ** 3)
    return c3


def _continuous_gradient(x, n_fft):
    """
    Numerical gradient for the continuous objective.
    """
    eps = 1e-7
    grad = np.zeros_like(x)
    f0 = _continuous_objective(x, n_fft)
    for i in range(N):
        xp = x.copy()
        xp[i] += eps
        fp = _continuous_objective(xp, n_fft)
        xm = x.copy()
        xm[i] -= eps
        fm = _continuous_objective(xm, n_fft)
        grad[i] = (fp - fm) / (2 * eps)
    return grad


def _discrete_local_search(h, deadline):
    """Fast greedy local search on ±1 sequence."""
    best_h = h.copy()
    best_val = _compute_c3_bound(best_h)
    n_fft = 1 << (2 * N - 1).bit_length()
    
    while time.time() < deadline:
        improved = False
        for i in range(N):
            best_h[i] = -best_h[i]
            val = _compute_c3_bound(best_h)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_h[i] = -best_h[i]
        
        if not improved:
            break
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    n_fft = 1 << (2 * N - 1).bit_length()
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = float('inf')
    
    # Phase 1: Continuous relaxation with L-BFGS-B from multiple random starts
    n_continuous_starts = 8
    for restart in range(n_continuous_starts):
        if time.time() > deadline - 60:
            break
        
        # Random initial point in unconstrained space
        x0 = rng.normal(0, 1.0, size=N)
        
        # Use L-BFGS-B with numerical gradient (slow but globally better)
        # We'll do a limited number of iterations
        try:
            result = minimize(
                _continuous_objective,
                x0,
                args=(n_fft,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            h_cont = np.tanh(result.x)
            
            # Project to ±1
            h_disc = np.sign(h_cont)
            # Handle zeros: assign +1
            h_disc[h_disc == 0] = 1.0
            
            val = _compute_c3_bound(h_disc)
            if val < best_val:
                best_val = val
                best_h = h_disc.copy()
        except Exception:
            continue
    
    # Phase 2: Try some direct random ±1 sequences
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        h = rng.choice([-1.0, 1.0], size=N)
        val = _compute_c3_bound(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Phase 3: Discrete local search refinement
    if best_h is not None and time.time() < deadline - 5:
        result_h, result_val = _discrete_local_search(best_h, deadline)
        if result_val < best_val:
            best_h = result_h
            best_val = result_val
    
    if best_h is None:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
