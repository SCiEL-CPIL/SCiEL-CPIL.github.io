import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
dx = 0.5 / N
_DEADLINE = 115.0  # Safety margin before 120s/150s kill

def _compute_ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        return -np.inf
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h * h)
    if l2_sq <= 0:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 0.0
    return l2_sq / (dx * l1 * max_conv)

def _smooth_max(x, sigma=0.005):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp

def _objective_complex(g_flat, sigma=0.005):
    """
    Objective to minimize: negative of the log ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0]**2 + g[:, 1]**2
    
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    
    # Normalize h to have L1 norm 1 to ensure scale invariance and stability
    # R is scale invariant, so we can fix L1=1
    h_norm = h / l1
    
    l2sq = np.sum(h_norm**2)
    if l2sq == 0:
        return 0.0
        
    conv = np.convolve(h_norm, h_norm, mode='full')
    # We want to maximize l2sq / (dx * max_conv).
    # Minimizing -log(l2sq) + log(dx) + log(max_conv)
    # Since dx is constant, we minimize -log(l2sq) + log(max_conv)
    
    peak = _smooth_max(conv, sigma)
    if peak <= 0:
        return 0.0
        
    # Objective: -log(l2sq) + log(peak)
    # Note: l2sq is small, peak is small.
    val = -math.log(l2sq) + math.log(peak)
    return val

def _optimize_complex(g0, time_limit):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    
    try:
        # We use a callback to stop if time runs out
        def callback(x):
            if time.time() - start_time > time_limit:
                return True
            return False

        result = minimize(
            _objective_complex,
            g0,
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 2000}
        )
        
        g_opt = result.x.reshape(N, 2)
        h_opt = g_opt[:, 0]**2 + g_opt[:, 1]**2
        return h_opt
    except Exception:
        g = g0.reshape(N, 2)
        h = g[:, 0]**2 + g[:, 1]**2
        return h

def _dft_parameterize(z):
    """
    Parameterize heights as |DFT(z)|^2 where z is a complex vector.
    This ensures non-negativity and provides a smooth manifold.
    """
    H = np.fft.fft(z)
    h = np.abs(H) ** 2
    return h

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    start_time = time.time()
    
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)
    
    # Generate diverse initializations for g (complex vector)
    initializations_g = []
    
    x = np.linspace(-1, 1, N)
    
    # 1. Uniform h -> g = sqrt(1) + 0j
    initializations_g.append(np.ones(N, dtype=complex))
    
    # 2. Gaussian-like h
    for width in [0.1, 0.5, 2.0]:
        h_gauss = np.exp(-x**2 / width)
        g_gauss = np.sqrt(h_gauss).astype(complex)
        initializations_g.append(g_gauss)
        
    # 3. Bimodal h
    for offset in [0.4, 0.6]:
        h_bimodal = np.exp(-((x - offset)**2) / 0.1) + np.exp(-((x + offset)**2) / 0.1)
        g_bimodal = np.sqrt(h_bimodal).astype(complex)
        initializations_g.append(g_bimodal)
        
    # 4. Triangle / tent
    h_tri = 1 - np.abs(x)
    h_tri = np.maximum(h_tri, 0)
    initializations_g.append(np.sqrt(h_tri).astype(complex))
    
    # 5. Cosine window
    h_cos = 0.5 + 0.5 * np.cos(np.pi * x)
    initializations_g.append(np.sqrt(h_cos).astype(complex))
    
    # 6. Random smooth h
    for seed in [1, 2, 3, 4, 5]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        g_smooth = np.sqrt(h_smooth).astype(complex)
        initializations_g.append(g_smooth)
        
    # 7. Peak in center
    h_center = np.zeros(N)
    h_center[N//2 - 2:N//2 + 3] = 1.0
    initializations_g.append(np.sqrt(h_center).astype(complex))
    
    # 8. Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:3] = 1.0
    h_edge[-3:] = 1.0
    initializations_g.append(np.sqrt(h_edge).astype(complex))
    
    # 9. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    initializations_g.append(np.sqrt(h_alt).astype(complex))
    
    # 10. Linear ramp
    h_ramp1 = np.linspace(0.1, 1.0, N)
    initializations_g.append(np.sqrt(h_ramp1).astype(complex))
    h_ramp2 = np.linspace(1.0, 0.1, N)
    initializations_g.append(np.sqrt(h_ramp2).astype(complex))
    
    # 11. DFT-based (binary mask approach)
    # Create h from |DFT(z)|^2 where z has sparse structure
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = np.zeros(N, dtype=complex)
        # Randomly select K active bins
        K = rng_local.randint(5, 15)
        active_bins = rng_local.choice(N, size=K, replace=False)
        z[active_bins] = rng_local.randn(K) + 1j * rng_local.randn(K)
        h_dft = _dft_parameterize(z)
        h_dft = h_dft / np.max(h_dft)
        g_dft = np.sqrt(h_dft).astype(complex)
        initializations_g.append(g_dft)
        
    # 12. Structured patterns
    h_sin = np.sin(np.pi * (x + 1) / 2)**2
    initializations_g.append(np.sqrt(h_sin).astype(complex))
    h_cos2 = np.cos(np.pi * x / 2)**2
    initializations_g.append(np.sqrt(h_cos2).astype(complex))

    # 13. Random complex vectors
    for seed in [100, 200, 300, 400, 500]:
        rng_local = np.random.RandomState(seed)
        g_rand = rng_local.randn(N) + 1j * rng_local.randn(N)
        initializations_g.append(g_rand)

    time_per_opt = 5.0  # seconds per optimization attempt

    for i, g0 in enumerate(initializations_g):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break
        
        # Convert g0 to flat real vector
        g0_real = g0.real
        g0_imag = g0.imag
        g0_flat = np.concatenate([g0_real, g0_imag])
        
        h_opt = _optimize_complex(g0_flat, time_per_opt)
        
        score = _compute_ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt
    
    # Final polish: run a few more iterations from the best solution with tighter sigma
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        # Try small perturbations around the best
        for pert_seed in [1000, 2000, 3000]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            g_best_real = np.sqrt(best_h)
            g_best_imag = np.zeros(N)
            
            pert_real = g_best_real + rng_local.randn(N) * 0.05
            pert_imag = rng_local.randn(N) * 0.05
            
            g_pert_flat = np.concatenate([pert_real, pert_imag])
            
            h_opt = _optimize_complex(g_pert_flat, 5.0)
            score = _compute_ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt
                
    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    
    return [float(x) for x in best_h]

if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _compute_ratio(h))
    print("time =", time.time() - t0)
