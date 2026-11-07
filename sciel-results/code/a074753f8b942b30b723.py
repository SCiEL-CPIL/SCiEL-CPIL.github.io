"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 100
DEADLINE = 110.0  # seconds, safety margin before 120s limit


def _value(h):
    """Compute the task value: max correlation / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2


def _is_valid(h):
    """Check if h is a valid solution."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True


def _project_to_simplex(h, target_sum=N / 2):
    """Project h onto the simplex: 0 <= h_i <= 1, sum(h) = target_sum.
    Uses iterative water-filling."""
    h = h.copy()
    for _ in range(1000):
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        # Adjust to meet sum constraint
        s = np.sum(h)
        if abs(s - target_sum) < 1e-14:
            break
        # Find which elements are at bounds
        at_lower = h <= 1e-12
        at_upper = h >= 1 - 1e-12
        free = ~(at_lower | at_upper)
        
        if not np.any(free):
            # All at bounds, can't adjust further
            break
        
        diff = target_sum - s
        n_free = np.sum(free)
        # Adjust free elements
        h[free] += diff / n_free
        # Re-clip
        h = np.clip(h, 0.0, 1.0)
    # Final projection
    h = np.clip(h, 0.0, 1.0)
    # Final sum adjustment
    s = np.sum(h)
    if abs(s - target_sum) > 1e-15:
        # Distribute remaining error to elements not at bounds
        diff = target_sum - s
        free = (h > 1e-10) & (h < 1 - 1e-10)
        if np.any(free):
            h[free] += diff / np.sum(free)
            h = np.clip(h, 0.0, 1.0)
    return h


def construct():
    """Construct 100 step heights minimizing the Erdos overlap value."""
    start_time = time.time()
    
    # Seed: all 0.5
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _value(best_h)
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Try several optimization strategies
    
    # Strategy 1: Direct optimization with penalty method
    # Objective: maximize min(h_i, 1-h_i) in a sense, or minimize max correlation
    # The correlation max is achieved at some lag. We want to minimize it.
    
    # A good heuristic: make h oscillate between low and high values
    # to reduce correlation with 1-h.
    
    # Let's try: alternating pattern
    # h = [a, b, a, b, ...] where a+b=1 for sum constraint
    # If a=b=0.5, value is baseline.
    # If we alternate between 0 and 1: h = [0,1,0,1,...]
    # sum = 50, which is N/2 = 50. Good.
    # Let's check the value for this pattern.
    
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    if _is_valid(h_alt):
        v_alt = _value(h_alt)
        if v_alt < best_val:
            best_h = h_alt.copy()
            best_val = v_alt
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Try to optimize using scipy
    # Parameterize with a lower-dimensional representation:
    # Use a periodic pattern with adjustable parameters
    
    # Try: h_i = 0.5 + A * cos(2*pi*k*i/N + phi) for some k, A, phi
    # But this might go out of [0,1]. Clip and project.
    
    # Actually, let's try direct optimization of the 100 variables
    # with a smooth approximation of the max.
    
    def objective(params):
        """Objective: smooth approximation of max correlation."""
        h = _project_to_simplex(params)
        corr = np.correlate(h, 1 - h, mode="full")
        # Soft max via log-sum-exp
        m = np.max(corr)
        if m < 1e-10:
            return 0.0
        # Scale for numerical stability
        scaled = corr / (m + 1e-10)
        log_sum_exp = m + np.log(np.sum(np.exp(scaled - 1.0)))
        return log_sum_exp / N * 2
    
    # Try L-BFGS-B from the alternating pattern
    if best_val < _value(h0):
        x0 = best_h.copy()
    else:
        x0 = h0.copy()
    
    try:
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10},
            tol=1e-12
        )
        if _is_valid(res.x):
            v_res = _value(res.x)
            if v_res < best_val:
                best_h = res.x.copy()
                best_val = v_res
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_h
    
    # Try multiple random restarts with different initial patterns
    rng = np.random.default_rng(42)
    
    for restart in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Generate random initial pattern
        # Mix of structured and random
        if restart % 3 == 0:
            # Periodic pattern
            period = rng.integers(2, 20)
            phase = rng.random()
            amp = rng.random() * 0.5
            x = 0.5 + amp * np.sin(2 * np.pi * np.arange(N) / period + phase)
        elif restart % 3 == 1:
            # Bimodal: some values near 0, some near 1
            x = rng.random(N)
            x = np.where(x < 0.5, x * 0.4, 0.6 + (x - 0.5) * 0.8)
        else:
            # Random
            x = rng.random(N)
        
        x = _project_to_simplex(x)
        
        try:
            res = minimize(
                objective,
                x,
                method="L-BFGS-B",
                options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-10},
                tol=1e-12
            )
            if _is_valid(res.x):
                v_res = _value(res.x)
                if v_res < best_val:
                    best_h = res.x.copy()
                    best_val = v_res
        except Exception:
            pass
    
    # Final polish: try to fine-tune the best solution
    if time.time() - start_time < DEADLINE - 2:
        try:
            res = minimize(
                objective,
                best_h,
                method="L-BFGS-B",
                options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-12},
                tol=1e-15
            )
            if _is_valid(res.x):
                v_res = _value(res.x)
                if v_res < best_val:
                    best_h = res.x.copy()
                    best_val = v_res
        except Exception:
            pass
    
    # Ensure final validity
    best_h = _project_to_simplex(best_h)
    if not _is_valid(best_h):
        # Fallback to alternating pattern
        best_h = np.zeros(N)
        best_h[1::2] = 1.0
    
    return best_h
