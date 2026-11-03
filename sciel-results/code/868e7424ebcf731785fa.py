"""
First autocorrelation inequality: minimize normalized max of autoconvolution
for a nonnegative step function with 600 equal intervals on [-1/4, 1/4].

Strategy:
1. Generate diverse initial candidates via cosine bump parametric family.
2. Global stochastic search: Differential Evolution over convex combinations
   of pre-computed extreme basis vectors (vertices of feasible polytope).
3. FFT-accelerated sensitivity-based local refinement (peak shaving).
4. Final high-precision polish pass with adaptive step sizes.

Key improvements:
- FFT-based autoconvolution and sensitivity (O(N log N) vs O(N^2))
- Differential Evolution navigates non-convex landscape of max autoconvolution
- Convex combination of extreme points ensures non-negativity and normalization
- Multi-stage refinement: global DE -> local sensitivity -> polish
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 600
TIME_LIMIT = 130  # safety margin before 150s kill

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
    
    # Cosine bumps with various configurations
    for i in range(n_basis - 1):
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

def local_refinement(h, start_time, max_time):
    """Local refinement using sensitivity-based peak reduction with FFT."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.05
    threshold_frac = 0.95
    
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
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = fft_autocorr(h_new)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.05, 0.5)
        else:
            alpha *= 0.85
            if alpha < 1e-8:
                break
    
    return best_h, best_val

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
    
    # Initial best: constant function
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Generate diverse initial candidates via parametric family
    candidate_list = [best_f.copy()]
    
    for n_c in [2, 3, 4, 5]:
        if time.time() - start_time > TIME_LIMIT - 60:
            break
        
        rng = np.random.RandomState(42 + n_c)
        
        for beta in [10.0, 20.0]:
            if time.time() - start_time > TIME_LIMIT - 45:
                break
            
            for restart in range(4):
                if time.time() - start_time > TIME_LIMIT - 40:
                    break
                
                centers = rng.uniform(-0.2, 0.2, size=n_c)
                widths = rng.uniform(0.05, 0.2, size=n_c)
                heights = rng.uniform(0.5, 2.0, size=n_c)
                
                x0 = np.concatenate([centers, widths, heights])
                
                try:
                    result = minimize(
                        objective, x0, args=(n_c, beta),
                        method='Nelder-Mead',
                        options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    f = parametric_to_step(result.x, n_c)
                    f = np.maximum(f, 0.0)
                    score = compute_score(f)
                    if score < best_score:
                        best_score = score
                        best_f = f.copy()
                    candidate_list.append(f)
                except Exception:
                    pass
    
    # Phase 2: Differential Evolution over convex combinations of extremes
    if time.time() - start_time < TIME_LIMIT - 40:
        # Build basis from best candidates + additional diverse ones
        n_basis = min(20, len(candidate_list))
        if n_basis >= 3:
            # Select most diverse candidates for basis
            # Use a simple greedy diversity selection
            selected = [candidate_list[0]]
            for cand in candidate_list[1:]:
                if len(selected) >= n_basis:
                    break
                # Check diversity: L2 distance from all selected
                min_dist = min(np.linalg.norm(cand - s) for s in selected)
                if min_dist > 0.1:
                    selected.append(cand)
            
            # Pad with random smooth functions if needed
            rng = np.random.RandomState(999)
            x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
            while len(selected) < n_basis:
                freq = rng.uniform(1.0, 6.0)
                phase = rng.uniform(0, 2*np.pi)
                f = 1.0 + 0.4 * np.cos(2*np.pi*freq*x + phase)
                f = np.maximum(f, 0.0)
                s = np.sum(f)
                if s > 1e-10:
                    selected.append(f / s)
            
            basis_matrix = np.array(selected)
            
            de_weights, de_score = differential_evolution_search(
                basis_matrix, start_time, TIME_LIMIT - 35, seed=777
            )
            
            # Reconstruct from DE result
            f_de = de_weights @ basis_matrix
            f_de = np.maximum(f_de, 0.0)
            s_de = np.sum(f_de)
            if s_de > 1e-10:
                f_de = f_de / s_de
                actual_score = compute_score(f_de)
                if actual_score < best_score:
                    best_score = actual_score
                    best_f = f_de.copy()
    
    # Phase 3: Local refinement with FFT-accelerated sensitivity
    if time.time() - start_time < TIME_LIMIT - 15:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 10
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score
    
    # Phase 4: Final polish pass
    if time.time() - start_time < TIME_LIMIT - 5:
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
