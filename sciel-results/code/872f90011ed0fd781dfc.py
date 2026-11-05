"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_chebyt

N = 400
DEADLINE = 110  # safety margin before 120s limit

# Precompute Chebyshev basis on [-1, 1]
_x = np.linspace(-1, 1, N)
_basis = np.array([eval_chebyt(k, _x) for k in range(12)])  # 12 coefficients
_basis_T = _basis.T  # shape (N, 12)


def _coeffs_to_h(c):
    """Convert coefficient vector to step heights."""
    h = _basis_T @ c
    # Ensure sum is not zero (it won't be for reasonable c, but guard)
    s = np.sum(h)
    if abs(s) < 1e-10:
        h[0] += 1e-6
    return h


def _value_from_h(h):
    """Compute the task value."""
    s = np.sum(h)
    if abs(s) < 1e-15:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2 * N * np.max(conv) / (s ** 2))


def _softmax_surrogate(c, beta=50.0):
    """
    Smooth surrogate for the objective using log-sum-exp.
    We want to minimize abs(2*N*max(conv)/s^2).
    Since max(conv) can be positive or negative, and we take abs,
    we handle both: minimize max(conv) and minimize -max(conv) (i.e., maximize min(conv)).
    Actually the value is abs(2*N*max(conv)/s^2). Let's think carefully.
    
    value = abs(2*N*max(conv) / s^2)
    
    To use a smooth surrogate, we can use soft-max for max(conv) and soft-min for min(conv).
    But actually we just need max(conv) since np.max takes the maximum signed entry.
    
    Let M = max(conv). value = abs(2*N*M/s^2).
    
    We can minimize a surrogate: f(c) = 2*N/s^2 * soft_max(conv, beta)
    where soft_max is log-sum-exp based.
    
    But abs makes it tricky. Let's just minimize the soft-max version of the numerator
    and handle the sign.
    
    Actually, let's just directly minimize: 2*N * softmax(conv, beta) / s^2
    using log-sum-exp for softmax.
    """
    h = _coeffs_to_h(c)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    # Soft max via log-sum-exp
    m = np.max(conv)
    shifted = conv - m
    # For numerical stability, use a moderate beta
    logsumexp = m + np.log(np.sum(np.exp(beta * shifted))) / beta
    # Actually standard log-sum-exp: m + (1/beta)*log(sum(exp(beta*(x-m))))
    # This approximates max as beta->inf
    val = 2.0 * N * logsumexp / (s * s)
    # We want abs, but since we're minimizing and the seed gives 2.0 (positive max),
    # let's just use the value directly. If max(conv) is negative, abs would flip it.
    # For now, minimize the soft-max value; if it goes negative we'll take abs at the end.
    return abs(val)


def _hard_objective(c):
    """Hard objective for final evaluation."""
    h = _coeffs_to_h(c)
    return _value_from_h(h)


def _construct_flat_null_phase(u):
    """
    Construct step heights using flat-spectrum mask with deep null at zero frequency.
    
    Parameters:
    - u: array of parameters [null_depth, phase_curvature, amplitude_scale]
    
    The idea: create a spectrum that is flat (constant magnitude) except for a deep 
    null at zero frequency. The inverse FFT of this gives h with large negative DC offset,
    forcing autoconvolution peak to be negative and small in magnitude.
    """
    null_depth = u[0]  # How deep the null is (0 to 1, 1 = perfect null)
    phase_curv = u[1]  # Phase curvature parameter
    amp_scale = u[2]   # Overall amplitude scaling
    
    # Create frequency indices
    k = np.fft.fftfreq(N)
    
    # Create flat spectrum with null at DC
    # Spectrum magnitude: flat except at k=0 where it's reduced
    spec_mag = np.ones(N)
    spec_mag[0] = 1.0 - null_depth  # Deep null at zero frequency
    
    # Create phase with curvature
    # Phase: linear + quadratic term for dispersion
    phase = 2 * np.pi * phase_curv * k ** 2
    
    # Complex spectrum
    spec = spec_mag * np.exp(1j * phase)
    
    # Inverse FFT to get step heights
    h = np.fft.ifft(spec).real
    
    # Scale
    h *= amp_scale
    
    # Ensure finite
    if not np.all(np.isfinite(h)):
        return np.ones(N)
    
    return h


def _flat_null_objective(u):
    """Objective function for flat-null parameter optimization."""
    h = _construct_flat_null_phase(u)
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv) / (s * s))


def construct():
    t0 = time.time()
    
    # Start with seed: all ones
    # In Chebyshev basis, what coefficients give h = ones?
    # h = _basis_T @ c = ones
    # c = pinv(_basis_T) @ ones
    c_seed = np.linalg.pinv(_basis_T) @ np.ones(N)
    
    best_c = c_seed.copy()
    best_val = _hard_objective(best_c)
    
    # Phase 1: Explore flat-null parameter space
    if time.time() - t0 < DEADLINE - 40:
        # Initial random search over flat-null parameters
        np.random.seed(42)
        best_u = None
        best_flat_val = 1e10
        
        # Try various null depths, curvatures, and scales
        for null_depth in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            for phase_curv in [-0.5, -0.2, 0.0, 0.2, 0.5]:
                for amp_scale in [0.5, 1.0, 2.0, 5.0]:
                    if time.time() - t0 > DEADLINE - 30:
                        break
                    u = np.array([null_depth, phase_curv, amp_scale])
                    val = _flat_null_objective(u)
                    if val < best_flat_val:
                        best_flat_val = val
                        best_u = u.copy()
        
        if best_u is not None and best_flat_val < best_val:
            # Convert best flat-null h to Chebyshev coefficients
            h_best = _construct_flat_null_phase(best_u)
            best_c = np.linalg.pinv(_basis_T) @ h_best
            best_val = best_flat_val
    
    # Phase 2: Local refinement with L-BFGS-B using soft-max surrogate
    if time.time() - t0 < DEADLINE - 10:
        # Try multiple beta values for soft-max
        for beta in [20.0, 50.0, 100.0]:
            if time.time() - t0 > DEADLINE - 5:
                break
            try:
                res = minimize(
                    _softmax_surrogate,
                    best_c,
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    # Verify with hard objective
                    hard_val = _hard_objective(res.x)
                    if hard_val < best_val:
                        best_c = res.x.copy()
                        best_val = hard_val
            except Exception:
                pass
    
    # Phase 3: Additional local refinement with tighter tolerance
    if time.time() - t0 < DEADLINE - 5:
        try:
            res = minimize(
                _softmax_surrogate,
                best_c,
                args=(200.0,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            hard_val = _hard_objective(res.x)
            if hard_val < best_val:
                best_c = res.x.copy()
                best_val = hard_val
        except Exception:
            pass
    
    # Final: return the best h found
    h_best = _coeffs_to_h(best_c)
    
    # Safety check
    if not np.all(np.isfinite(h_best)) or abs(np.sum(h_best)) < 1e-15:
        return np.ones(N)
    
    return h_best


# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for structural compatibility)
# EVOLVE-BLOCK-END
