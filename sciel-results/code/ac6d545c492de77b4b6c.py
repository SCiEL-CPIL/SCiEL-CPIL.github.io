"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv)) / (s ** 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _compute_value(best_h)  # should be 2.0

    # Strategy 1: Try various structured initializations
    candidates = []

    # Constant (seed)
    candidates.append(np.ones(N))

    # Triangular / parabolic concentration in center
    idx = np.arange(N)
    center = (N - 1) / 2.0
    tri = np.maximum(0, 1 - np.abs(idx - center) / center)
    if np.sum(tri) > 0:
        candidates.append(tri / np.sum(tri))

    # Gaussian centered
    gauss = np.exp(-((idx - center) ** 2) / (2 * (N / 4) ** 2))
    if np.sum(gauss) > 0:
        candidates.append(gauss / np.sum(gauss))

    # Alternating with small offset to avoid sum=0
    alt = np.array([(-1) ** i for i in range(N)], dtype=float)
    alt[0] += 1.0  # break symmetry to make sum nonzero
    if abs(np.sum(alt)) > 1e-9:
        candidates.append(alt / np.sum(alt))

    # Random perturbations of the constant
    rng = np.random.default_rng(42)
    for trial in range(20):
        h = np.ones(N) + rng.standard_normal(N) * 0.5
        if abs(np.sum(h)) > 1e-9:
            candidates.append(h / np.sum(h))

    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        val = _compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    # Strategy 2: Local optimization from the best candidate
    if time.time() < deadline - 5.0:
        def objective(h):
            s = np.sum(h)
            if abs(s) < 1e-6:
                return 1e10
            conv = np.convolve(h, h)
            k_max = np.max(conv)
            return abs(2.0 * N * k_max) / (s ** 2)

        # Try L-BFGS-B from best_h
        try:
            res = minimize(
                objective,
                best_h,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if time.time() < deadline:
                val = objective(res.x)
                if val < best_val:
                    best_val = val
                    best_h = res.x.copy()
        except Exception:
            pass

        # Try from a few other good candidates
        for h0 in candidates[:5]:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(
                    objective,
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if time.time() < deadline:
                    val = objective(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
            except Exception:
                pass

    # Strategy 3: Random search with refinement
    if time.time() < deadline - 5.0:
        rng2 = np.random.default_rng(123)
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            # Random direction perturbation of best_h
            direction = rng2.standard_normal(N)
            for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
                h_try = best_h + scale * direction
                val = _compute_value(h_try)
                if val < best_val:
                    best_val = val
                    best_h = h_try.copy()
                    break

    # Strategy 4: More aggressive random restarts with local optimization
    if time.time() < deadline - 10.0:
        rng3 = np.random.default_rng(777)
        for restart in range(10):
            if time.time() > deadline - 10.0:
                break
            # Random starting point
            h0 = rng3.standard_normal(N)
            if abs(np.sum(h0)) < 1e-9:
                h0[0] += 1.0
            h0 = h0 / np.sum(h0)
            
            try:
                res = minimize(
                    objective,
                    h0,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if time.time() < deadline:
                    val = objective(res.x)
                    if val < best_val:
                        best_val = val
                        best_h = res.x.copy()
            except Exception:
                pass

    # Strategy 5: Try sequences with negative values mixed in
    if time.time() < deadline - 5.0:
        rng4 = np.random.default_rng(999)
        for _ in range(30):
            if time.time() > deadline - 5.0:
                break
            # Mix of positive and negative values
            h = rng4.standard_normal(N)
            # Ensure sum is not zero
            if abs(np.sum(h)) < 1e-9:
                h[0] += 1.0
            h = h / np.sum(h)
            
            val = _compute_value(h)
            if val < best_val:
                best_val = val
                best_h = h.copy()
                
                # Refine locally
                try:
                    res = minimize(
                        objective,
                        best_h,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if time.time() < deadline:
                        val2 = objective(res.x)
                        if val2 < best_val:
                            best_val = val2
                            best_h = res.x.copy()
                except Exception:
                    pass

    return best_h
# EVOLVE-BLOCK-END
