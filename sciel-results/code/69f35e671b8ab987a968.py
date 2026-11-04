import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
_DH = 1.0 / 20.0
_DEADLINE = 115.0  # seconds, safety margin before 120s kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _elapsed()


def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        h = np.maximum(h, 0.0)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = np.convolve(h, h, mode='full')
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = float(np.max(x))
    if x_max == 0.0:
        return 0.0
    x_shifted = x - x_max
    exp_sum = float(np.sum(np.exp(x_shifted / sigma)))
    if exp_sum <= 0.0:
        return 0.0
    return x_max + sigma * math.log(exp_sum)


def _objective_complex(g_flat, sigma=0.01):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0] ** 2 + g[:, 1] ** 2

    # Normalize h to have L1 norm 1 to ensure scale invariance and stability
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    h = h / l1

    l2sq = float(np.sum(h ** 2))
    if l2sq <= 0.0:
        return 0.0

    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma)

    if peak <= 0.0:
        return 0.0

    return -l2sq / peak


def _grad_objective_complex(g_flat, sigma=0.01):
    """
    Gradient of the objective with respect to g_flat.
    Uses finite differences for simplicity and robustness.
    """
    eps = 1e-6
    grad = np.zeros_like(g_flat)
    base_val = _objective_complex(g_flat, sigma)
    for i in range(len(g_flat)):
        g_plus = g_flat.copy()
        g_plus[i] += eps
        val_plus = _objective_complex(g_plus, sigma)
        grad[i] = (val_plus - base_val) / eps
    return grad


def _optimize_complex(g0, time_limit, sigma=0.01):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    best_g = g0.copy()
    best_val = _objective_complex(g0, sigma)

    # We will run L-BFGS-B with a callback to stop if time runs out
    # We also need to ensure we don't exceed time_limit

    def callback(x):
        nonlocal best_g, best_val
        if time.time() - start_time > time_limit * 0.9:
            raise RuntimeError("Time limit exceeded")
        val = _objective_complex(x, sigma)
        if val < best_val:
            best_val = val
            best_g = x.copy()
        return x

    try:
        # Use finite difference gradient initially, then maybe switch? 
        # For now, finite difference is safer for correctness.
        # To speed up, we could implement analytic gradient, but let's try FD first.
        # Actually, FD is slow (2N evals per step). 
        # Let's try to implement analytic gradient for the complex representation.
        
        # h = |g|^2. l1 = sum(h). h_norm = h/l1.
        # l2sq = sum(h_norm^2). peak = smooth_max(conv(h_norm)).
        # This is complicated. Let's stick to FD but limit iterations.
        
        res = minimize(
            _objective_complex,
            g0,
            method='L-BFGS-B',
            jac=_grad_objective_complex,
            args=(sigma,),
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5, 'maxfun': 200, 'disp': False},
            callback=callback
        )
        if res.x is not None:
            val_res = _objective_complex(res.x, sigma)
            if val_res < best_val:
                best_val = val_res
                best_g = res.x.copy()
    except RuntimeError:
        pass
    except Exception:
        pass

    g_opt = best_g
    h_opt = g_opt[:, 0] ** 2 + g_opt[:, 1] ** 2
    return h_opt


def _initializations():
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Random positive
    for _ in range(20):
        h = rng.random(N) + 0.1
        inits.append(h)
        
    # 2. Sparse spikes
    for k in [1, 2, 3, 5]:
        for _ in range(5):
            h = np.zeros(N)
            idx = rng.choice(N, size=k, replace=False)
            h[idx] = 1.0
            inits.append(h)
            
    # 3. Bumps
    for _ in range(10):
        center = rng.uniform(0, N)
        width = rng.uniform(1.0, 5.0)
        x = np.arange(N)
        h = np.exp(-0.5 * ((x - center) / width) ** 2)
        inits.append(h)
        
    # 4. Cosine window
    for _ in range(5):
        x = np.arange(N)
        h = 0.5 * (1 + np.cos(2 * np.pi * x / N))
        inits.append(h)
        
    # 5. Alternating
    for _ in range(3):
        h = np.ones(N)
        h[1::2] = 0.5
        inits.append(h)
        
    # 6. Blocky
    for _ in range(5):
        h = np.zeros(N)
        start = rng.integers(0, N-5)
        end = start + rng.integers(5, 15)
        h[start:end] = 1.0
        inits.append(h)
        
    # 7. Disjoint Gaussian Bumps
    for _ in range(20):
        c1 = rng.uniform(0, N/4)
        c2 = rng.uniform(3*N/4, N)
        s1 = rng.uniform(0.5, 2.0)
        s2 = rng.uniform(0.5, 2.0)
        a = rng.uniform(0.5, 2.0)
        b = rng.uniform(0.5, 2.0)
        
        x = np.arange(N)
        w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
        w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
        h = (a * w1 + b * w2) ** 2
        h = np.maximum(h, 0.0)
        if np.sum(h) > 0:
            h = h / np.sum(h) * 1.0
        inits.append(h)
        
    # 8. Tapered Plateau
    for _ in range(5):
        x = np.arange(N)
        h = 0.5 * (1 + np.cos(2 * np.pi * x / N))
        taper = np.exp(-0.1 * (x - N/2)**2)
        h = h * taper
        h = np.maximum(h, 0.0)
        if np.sum(h) > 0:
            h = h / np.sum(h) * 1.0
        inits.append(h)
        
    # 9. DFT-based initializations
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        h_dft = np.maximum(h_dft, 0.0)
        if np.sum(h_dft) > 0:
            h_dft = h_dft / np.sum(h_dft)
        inits.append(h_dft)
        
    # 10. Structured patterns
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-x**2 / 0.1))
    inits.append(np.exp(-x**2 / 0.5))
    inits.append(np.exp(-x**2 / 2.0))
    inits.append(np.exp(-((x - 0.4)**2) / 0.1) + np.exp(-((x + 0.4)**2) / 0.1))
    inits.append(np.exp(-((x - 0.6)**2) / 0.05) + np.exp(-((x + 0.6)**2) / 0.05))
    inits.append(1 - np.abs(x))
    inits.append(0.5 + 0.5 * np.cos(np.pi * x))
    
    return inits


def construct():
    global _start
    _start = time.time()
    
    best_h = None
    best_v = 0.0
    
    inits = _initializations()
    
    # Score initializations
    scored = []
    for h in inits:
        h = np.maximum(h, 0.0)
        v = _ratio(h)
        scored.append((v, h))
        if v > best_v:
            best_v = v
            best_h = h.copy()
            
    scored.sort(key=lambda x: -x[0])
    
    # Optimization loop
    # We use the complex parameterization g where h = |g|^2
    # This naturally enforces non-negativity.
    
    n_starts = len(scored)
    
    for i, (v0, h0) in enumerate(scored):
        if _time_left() <= 10.0:
            break
            
        # Skip if this start is much worse than current best and we have many starts left
        if i > 5 and v0 < best_v * 0.8 and best_v > 15.0:
            continue
            
        t0 = time.time()
        time_budget = min(_time_left() * 0.7, 15.0) # Cap per start to ensure diversity
        
        # Convert h0 to g0
        # h0 = |g0|^2 => g0 = sqrt(h0) * exp(i * phase)
        # We can just use real g0 = sqrt(h0), imag g0 = 0
        g0_real = np.sqrt(h0)
        g0_imag = np.zeros(N)
        g0_flat = np.concatenate([g0_real, g0_imag])
        
        # Phase 1: L-BFGS-B on complex parameters
        try:
            h_opt = _optimize_complex(g0_flat, time_budget, sigma=0.01)
            h_opt = np.maximum(h_opt, 0.0)
            v_opt = _ratio(h_opt)
            if v_opt > best_v:
                best_v = v_opt
                best_h = h_opt.copy()
        except Exception:
            h_opt = h0.copy()
            
        # Phase 2: Nelder-Mead polish on exact objective in h-space
        if _time_left() > 10.0:
            try:
                t1 = time.time()
                time_budget2 = min(_time_left() * 0.4, 8.0)
                
                def _neg_ratio_exact(h):
                    val = _ratio(h)
                    if val <= 0.0:
                        return 1e12
                    return -math.log(val)
                
                res2 = minimize(
                    _neg_ratio_exact,
                    h_opt,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False},
                    callback=lambda x: (np.nan if (time.time() - t1) > time_budget2 else x)
                )
                h_pol = np.maximum(res2.x, 0.0)
                v_pol = _ratio(h_pol)
                if v_pol > best_v:
                    best_v = v_pol
                    best_h = h_pol.copy()
            except Exception:
                pass
                
    # Final polish on the best solution found
    if best_h is not None and _time_left() > 10.0:
        try:
            t_final = time.time()
            time_budget_final = _time_left() * 0.8
            
            def _neg_ratio_exact(h):
                val = _ratio(h)
                if val <= 0.0:
                    return 1e12
                return -math.log(val)
                
            res_final = minimize(
                _neg_ratio_exact,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - t_final) > time_budget_final else x)
            )
            h_final = np.maximum(res_final.x, 0.0)
            v_final = _ratio(h_final)
            if v_final > best_v:
                best_v = v_final
                best_h = h_final.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
        
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
