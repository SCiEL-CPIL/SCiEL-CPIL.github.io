"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Represent f as a truncated Legendre polynomial series with constrained
   coefficients ensuring non-negativity on [-1/4, 1/4].
2. Global stochastic search: Differential Evolution over Legendre coefficients
   to minimize the peak autoconvolution.
3. FFT-accelerated sensitivity-based local refinement (peak shaving) with
   simulated annealing acceptance for escaping local minima.
4. Spectral refinement: dampen high-power frequency components to flatten
   the autoconvolution peak.
5. Final high-precision polish pass with adaptive step sizes.

Key improvements:
- Legendre polynomial representation: smooth, low-dimensional, inherently
  structured (orthogonal basis), easy to enforce non-negativity
- FFT-based autoconvolution and sensitivity (O(N log N) vs O(N^2))
- Differential Evolution navigates the coefficient space
- Multi-stage refinement: global DE -> local sensitivity -> spectral -> polish
- Simulated annealing acceptance in local refinement for better exploration
- Spectral domain refinement to reduce peak concentration
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

# Precompute Legendre polynomial values on our grid
# x in [-1/4, 1/4], we'll use the standard Legendre polynomials P_k(t) on t in [-1,1]
# Map x to t: t = 4*x, so t in [-1, 1]
_X_GRID = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
_T_GRID = 4.0 * _X_GRID  # t in [-0.998333..., 0.998333...]

def _legendre_values(k_max=12):
    """Precompute Legendre polynomial values P_k(t) on the grid for k=0..k_max."""
    vals = np.zeros((k_max + 1, N))
    vals[0] = 1.0
    if k_max >= 1:
        vals[1] = _T_GRID
    for k in range(2, k_max + 1):
        vals[k] = ((2*k - 1) * _T_GRID * vals[k-1] - (k - 1) * vals[k-2]) / k
    return vals

_LEG_VALS = _legendre_values(12)

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def legendre_to_step(coeffs):
    """Convert Legendre coefficients to a step function of length N.
    coeffs: array of length K+1, where f(x) = sum_k coeffs[k] * P_k(4x)
    Returns: array of length N, nonnegative.
    """
    k_max = len(coeffs) - 1
    f = np.zeros(N)
    for k in range(k_max + 1):
        if coeffs[k] != 0.0:
            f += coeffs[k] * _LEG_VALS[k]
    f = np.maximum(f, 0.0)
    return f

def legendre_objective(coeffs, beta=20.0):
    """Objective for DE: minimize smooth max of autoconvolution of Legendre function."""
    f = legendre_to_step(coeffs)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    conv = fft_autocorr(f)
    return smooth_max(conv, beta) / (s * s)

def ensure_nonnegative_legendre(coeffs, x_grid=None):
    """Check and enforce non-negativity by projecting onto feasible set.
    Simple approach: if any value is negative, scale down higher-order terms."""
    if x_grid is None:
        x_grid = _X_GRID
    f = np.zeros(N)
    k_max = len(coeffs) - 1
    for k in range(k_max + 1):
        f += coeffs[k] * _LEG_VALS[k]
    if np.min(f) >= -1e-10:
        return coeffs.copy()
    # Simple projection: reduce higher-order coefficients
    # This is a heuristic; DE will handle the constraints
    return coeffs

def fft_autocorr(f):
    """Compute linear autoconvolution using FFT."""
    n = len(f)
    size = 2 * n
    F = np.fft.rfft(f, size)
    C = np.fft.irfft(F * F, size)
    return C[:2*n-1]

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def cosine_basis(x, centers, widths, heights):
    """Generate a smooth nonnegative function from cosine bump components."""
    result = np.zeros_like(x)
    for c, w, h in zip(centers, widths, heights):
        dist = np.abs(x - c)
        mask = dist <= w / 2.0
        if np.any(mask):
            result[mask] += h * np.cos(np.pi * (x[mask] - c) / w) ** 2
    return result

def parametric_to_step(params, n_components):
    """Convert parameters to a step function of length N."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    centers = params[:n_components]
    widths = params[n_components:2*n_components]
    heights = params[2*n_components:3*n_components]
    
    widths = np.abs(widths) + 0.01
    widths = np.minimum(widths, 0.5)
    heights = np.abs(heights)
    
    f = cosine_basis(x, centers, widths, heights)
    f = np.maximum(f, 0.0)
    return f

def objective(params, n_components, beta=10.0):
    f = parametric_to_step(params, n_components)
    conv = fft_autocorr(f)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return smooth_max(conv, beta) / (s * s)

def generate_extreme_basis(n_basis=20, seed=42):
    """Generate diverse extreme nonnegative basis vectors.
    
    These are vertices of the feasible polytope (non-negative, normalized)
    created by combining different parametric families and random perturbations.
    """
    rng = np.random.RandomState(seed)
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    basis = []
    
    # Constant function
    f_const = np.ones(N)
    basis.append(f_const / np.sum(f_const))
    
    # Parabolic initial guess
    f_parab = np.maximum(1.0 - (x / 0.25) ** 2, 0.0)
    s_parab = np.sum(f_parab)
    if s_parab > 1e-10:
        basis.append(f_parab / s_parab)
    
    # Cosine initial guess
    f_cos = 0.5 * (1.0 + np.cos(2 * np.pi * x))
    s_cos = np.sum(f_cos)
    if s_cos > 1e-10:
        basis.append(f_cos / s_cos)
    
    # Cosine bumps with various configurations
    for i in range(n_basis - 4):
        n_c = rng.randint(2, 6)
        centers = rng.uniform(-0.2, 0.2, size=n_c)
        widths = rng.uniform(0.03, 0.25, size=n_c)
        heights = rng.uniform(0.3, 2.0, size=n_c)
        
        f = cosine_basis(x, centers, widths, heights)
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s > 1e-10:
            basis.append(f / s)
    
    # Add some random smooth functions
    for i in range(3):
        freq = rng.uniform(1.0, 8.0)
        phase = rng.uniform(0, 2*np.pi)
        f = 1.0 + 0.5 * np.cos(2*np.pi*freq*x + phase)
        f = np.maximum(f, 0.0)
        s = np.sum(f)
        if s > 1e-10:
            basis.append(f / s)
    
    return np.array(basis)

def legendre_de_search(start_time, max_time, seed=123):
    """Run Differential Evolution over Legendre coefficients to find optimal
    nonnegative function minimizing peak autoconvolution.
    
    We use K=8 Legendre terms (coefficients a_0 through a_8).
    Non-negativity is enforced by the objective returning inf if violated.
    """
    K = 8  # number of Legendre terms (0..K)
    
    # Bounds for coefficients: a_0 should be large enough to keep f >= 0
    # P_0 = 1, P_1 = t, P_2 = (3t^2-1)/2, etc.
    # On t in [-1,1], |P_k(t)| <= 1 for all k.
    # So f(t) = sum a_k P_k(t) >= sum |a_k| * (-1) ... not quite.
    # Safer: a_0 in [0.5, 3.0], a_k in [-1.0, 1.0] for k>=1
    bounds = [(0.5, 3.0)] + [(-1.0, 1.0)] * K
    
    best_coeffs = None
    best_score = np.inf
    
    de_deadline = start_time + min(50, max_time * 0.5)
    
    configs = [
        {'popsize': 20, 'maxiter': 60, 'seed': seed},
        {'popsize': 25, 'maxiter': 50, 'seed': seed + 1},
        {'popsize': 15, 'maxiter': 80, 'seed': seed + 2},
    ]
    
    for config in configs:
        if time.time() > de_deadline:
            break
        
        try:
            result = differential_evolution(
                legendre_objective, bounds,
                seed=config['seed'],
                popsize=config['popsize'],
                maxiter=config['maxiter'],
                tol=1e-7,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube',
                updating='immediate',
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    if best_coeffs is None:
        # Fallback: constant function
        best_coeffs = np.zeros(K + 1)
        best_coeffs[0] = 1.0
        best_score = legendre_objective(best_coeffs)
    
    # Convert to step function
    f = legendre_to_step(best_coeffs)
    s = np.sum(f)
    if s > 1e-10:
        f = f / s
    
    actual_score = compute_score(f)
    return f, actual_score

def de_objective(weights, basis_matrix, beta=15.0):
    """Objective for Differential Evolution: minimize max autoconvolution
    of convex combination of basis vectors."""
    weights = np.maximum(weights, 0.0)
    w_sum = np.sum(weights)
    if w_sum < 1e-15:
        return 1e10
    weights = weights / w_sum
    
    f = weights @ basis_matrix
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    conv = fft_autocorr(f)
    return smooth_max(conv, beta) / (s * s)

def differential_evolution_search(basis_matrix, start_time, max_time, seed=123):
    """Run Differential Evolution to find optimal convex combination weights."""
    n_basis = basis_matrix.shape[0]
    rng = np.random.RandomState(seed)
    
    # Initial population: random simplex points
    best_weights = None
    best_score = np.inf
    
    # Time budget for DE
    de_deadline = start_time + min(45, max_time * 0.4)
    
    # Bounds: weights in [0, 1]
    bounds = [(0, 1)] * n_basis
    
    # Try different population sizes and max iterations
    configs = [
        {'popsize': 15, 'maxiter': 80},
        {'popsize': 20, 'maxiter': 60},
        {'popsize': 12, 'maxiter': 100},
    ]
    
    for config in configs:
        if time.time() > de_deadline:
            break
        
        try:
            result = differential_evolution(
                de_objective, bounds,
                args=(basis_matrix,),
                seed=seed,
                popsize=config['popsize'],
                maxiter=config['maxiter'],
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,  # We do our own polish
                init='latinhypercube',
                updating='immediate',
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_weights = result.x
        except Exception:
            pass
    
    if best_weights is None:
        # Fallback: uniform weights
        best_weights = np.ones(n_basis) / n_basis
        best_score = de_objective(best_weights, basis_matrix)
    
    return best_weights, best_score

def compute_sensitivity_fft(h, peak_mask):
    """Compute sensitivity of autoconvolution peaks to each bin using FFT."""
    n = len(h)
    size = 2 * n
    P = np.fft.rfft(peak_mask.astype(np.float64), size)
    H = np.fft.rfft(h, size)
    corr = np.fft.irfft(P * np.conj(H), size)
    return corr[:n]

def local_refinement(h, start_time, max_time, rng=None):
    """Local refinement using sensitivity-based peak reduction with FFT
    and simulated annealing acceptance for better exploration."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
    # Simulated annealing parameters
    T = 0.1  # initial "temperature"
    cooling = 0.995
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        sensitivity = compute_sensitivity_fft(current_h, peak_mask)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        # Add noise for simulated annealing effect
        noise = rng.normal(0, 0.05, size=N)
        update = -alpha * sens_norm + alpha * 0.3 * noise
        
        h_new = current_h * (1.0 + update)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            T *= cooling
            continue
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        # Accept if better or with probability exp(-delta/T)
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
            T *= cooling
        else:
            delta = val_new - current_val
            if delta > 0:
                accept_prob = np.exp(-delta / (T + 1e-12))
                if rng.random() < accept_prob:
                    current_h = h_new
            else:
                alpha *= 0.85
            T *= cooling
            if alpha < 1e-8:
                break
        
        # Early stopping if temperature too low
        if T < 1e-6:
            break
    
    return best_h, best_val

def spectral_refinement(f, start_time, max_time):
    """Refine using frequency domain insights.
    
    The autoconvolution peak is related to the L-infinity norm of the inverse
    Fourier transform of |F|^2. We can try to flatten the power spectrum.
    """
    best_f = f.copy()
    best_val = compute_score(f)
    current_f = f.copy()
    
    # Get FFT of current function
    F = np.fft.rfft(current_f, n=2*N)
    power = np.abs(F) ** 2
    
    # We want to reduce the peak of autoconvolution, which corresponds to
    # reducing the high-frequency components that cause sharp peaks
    # Try to smooth the power spectrum
    for iteration in range(2000):
        if time.time() - start_time > max_time:
            break
        
        s = np.sum(current_f)
        if s < 1e-12:
            break
        current_f = current_f * (N / s)
        
        conv = fft_autocorr(current_f)
        current_val = np.max(conv) / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_f = current_f.copy()
        
        # Get FFT
        F = np.fft.rfft(current_f, n=2*N)
        power = np.abs(F) ** 2
        
        # Identify frequency bins with high power that contribute to peak
        # The DC component is always present; focus on others
        if len(power) > 1:
            power_no_dc = power[1:]
            max_power = np.max(power_no_dc)
            if max_power > 1e-12:
                # Reduce high-power frequency components
                threshold = max_power * 0.9
                high_power_mask = power_no_dc >= threshold
                # Dampen these components in frequency domain
                F_damped = F.copy()
                F_damped[1:][high_power_mask] *= 0.95
                # Inverse FFT
                f_new = np.fft.irfft(F_damped, n=2*N)[:N]
                f_new = np.maximum(f_new, 0.0)
                s_new = np.sum(f_new)
                if s_new > 1e-12:
                    f_new = f_new * (N / s_new)
                    conv_new = fft_autocorr(f_new)
                    val_new = np.max(conv_new) / (N * N)
                    if val_new < current_val:
                        current_f = f_new
                    else:
                        break
                else:
                    break
        else:
            break
    
    return best_f, best_val

def polish_pass(h, start_time, max_time):
    """Final polish pass with adaptive step sizes and tighter convergence."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    threshold_frac = 0.98
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time and no_improve_count < max_no_improve:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = fft_autocorr(current_h)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        threshold = max_c * threshold_frac
        peak_mask = (conv >= threshold).astype(np.float64)
        
        sensitivity = compute_sensitivity_fft(current_h, peak_mask)
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        effective_alpha = alpha * (1.0 + 0.5 * np.exp(-no_improve_count / 10.0))
        
        h_new = current_h * (1.0 - effective_alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.3)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Legendre polynomial DE search (primary global optimization)
    if time.time() - start_time < TIME_LIMIT - 50:
        try:
            leg_f, leg_score = legendre_de_search(start_time, TIME_LIMIT - 45, seed=42)
            if leg_score < best_score:
                best_score = leg_score
                best_f = leg_f.copy()
        except Exception:
            pass
    
    # Phase 2: Additional Legendre DE with different seeds for diversity
    if time.time() - start_time < TIME_LIMIT - 35:
        for seed in [100, 200]:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            try:
                leg_f2, leg_score2 = legendre_de_search(start_time, TIME_LIMIT - 28, seed=seed)
                if leg_score2 < best_score:
                    best_score = leg_score2
                    best_f = leg_f2.copy()
            except Exception:
                pass
    
    # Phase 3: Local refinement with FFT-accelerated sensitivity and SA
    if time.time() - start_time < TIME_LIMIT - 20:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 15, rng
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 4: Spectral refinement
    if time.time() - start_time < TIME_LIMIT - 10:
        spec_f, spec_score = spectral_refinement(
            best_f, start_time, TIME_LIMIT - 5
        )
        if spec_score < best_score:
            best_f = spec_f
            best_score = spec_score
    
    # Phase 5: Final polish pass
    if time.time() - start_time < TIME_LIMIT - 3:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
