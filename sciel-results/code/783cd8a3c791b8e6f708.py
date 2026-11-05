"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _value(h):
    """Compute the objective value (lower is better)."""
    s = np.sum(h)
    if s == 0 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))


def _objective(h):
    """Objective to minimize."""
    return _value(h)


def construct():
    """
    Construct a step height vector that minimizes:
        value = abs(2 * N * max(convolve(h, h)) / sum(h)^2)
    
    Strategy:
    1. Start with a structured initial guess that already gives improvement.
       Use a pattern that reduces the peak of the autoconvolution relative to sum^2.
       A good approach: use h[i] = 1 + a * sin(2*pi*k*i/N) with appropriate k,
       or use a two-level pattern.
    
    2. Perform local optimization using scipy or manual gradient-free search.
    
    Key insight: The autoconvolution max is dominated by the DC component (sum^2/N)
    when h is constant. To reduce the ratio, we want the autoconvolution peak to be
    as small as possible relative to sum^2. This means we want h to have significant
    high-frequency content that cancels in the autoconvolution peak.
    
    A two-level pattern: alternate between +1 and -1 gives sum=0 (invalid).
    Use +1 and -0.5: sum = 400*0.25 = 100, autoconvolution will have reduced peak.
    
    Let's try: h[i] = 1 if i even, -0.5 if i odd.
    sum = 200*1 + 200*(-0.5) = 100
    The autoconvolution of this pattern will have its max reduced.
    
    Actually, let's use optimization to find a better pattern.
    """
    deadline = time.time() + 110  # 10s safety margin
    
    # Start with a structured initial guess
    # Use a pattern with alternating signs to reduce autoconvolution peak
    best_h = np.ones(N)
    best_val = _value(best_h)  # Should be 2.0
    
    # Try several structured initial guesses
    for k in [1, 2, 3, 5, 8, 10, 16, 20, 32, 50, 100, 200]:
        h_try = 1.0 + 0.5 * np.sin(2 * np.pi * k * np.arange(N) / N)
        # Ensure sum is not zero
        s = np.sum(h_try)
        if abs(s) < 1e-6:
            continue
        val = _value(h_try)
        if val < best_val:
            best_val = val
            best_h = h_try.copy()
    
    # Also try two-level patterns
    for ratio in [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        h_try = np.ones(N)
        h_try[1::2] = -ratio
        s = np.sum(h_try)
        if abs(s) < 1e-6:
            continue
        val = _value(h_try)
        if val < best_val:
            best_val = val
            best_h = h_try.copy()
    
    # Try block patterns
    for block_size in [2, 4, 8, 16, 32]:
        for ratio in [0.5, 1.0, 2.0]:
            h_try = np.ones(N)
            h_try[1::block_size] = -ratio
            s = np.sum(h_try)
            if abs(s) < 1e-6:
                continue
            val = _value(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
    
    # Now do local optimization using Nelder-Mead on a reduced set of parameters
    # Represent h as base + sum of a few sinusoidal modes
    from scipy.optimize import minimize
    
    def param_to_h(params):
        """Convert parameters to height vector.
        params[0] = base offset
        params[1:] = amplitudes of sinusoidal modes
        """
        base = params[0]
        amps = params[1:]
        h = np.full(N, base)
        for i, amp in enumerate(amps):
            k = (i + 1) * 2  # Use even frequencies to avoid DC issues
            h += amp * np.sin(2 * np.pi * k * np.arange(N) / N)
        return h
    
    def objective_params(params):
        h = param_to_h(params)
        s = np.sum(h)
        if abs(s) < 1e-6:
            return 1e6
        return _value(h)
    
    # Start from best_h found so far
    # Fit a few sinusoidal modes to best_h
    base = np.mean(best_h)
    amps = np.zeros(5)
    for i in range(5):
        k = (i + 1) * 2
        amps[i] = 2.0 / N * np.sum((best_h - base) * np.sin(2 * np.pi * k * np.arange(N) / N))
    
    x0 = np.concatenate([[base], amps])
    
    try:
        result = minimize(objective_params, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if time.time() < deadline:
            h_opt = param_to_h(result.x)
            s = np.sum(h_opt)
            if abs(s) > 1e-6 and np.all(np.isfinite(h_opt)):
                val_opt = _value(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
    except Exception:
        pass
    
    # If time remains, do a few random perturbations
    if time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for _ in range(50):
            if time.time() > deadline:
                break
            # Random perturbation of best_h
            h_try = best_h + rng.normal(0, 0.1, N)
            s = np.sum(h_try)
            if abs(s) < 1e-6:
                continue
            val = _value(h_try)
            if val < best_val:
                best_val = val
                best_h = h_try.copy()
    
    return best_h
# EVOLVE-BLOCK-END
