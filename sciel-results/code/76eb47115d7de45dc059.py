import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = 145.0  # Safety margin before 150s kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _elapsed()


def _ratio_exact(h):
    """Compute the exact ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
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


def _smooth_max(x, sigma=0.1):
    """Smooth approximation of max using log-sum-exp."""
    x_max = float(np.max(x))
    if x_max == 0.0:
        return 0.0
    # Use shifted values for numerical stability
    x_shifted = x - x_max
    # exp(x_shifted / sigma)
    # If sigma is small, this is sharp.
    exp_vals = np.exp(x_shifted / sigma)
    exp_sum = float(np.sum(exp_vals))
    if exp_sum <= 0.0:
        return 0.0
    return x_max + sigma * math.log(exp_sum)


def _objective_complex(g_flat, sigma=0.1):
    """
    Objective to minimize: negative log of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0] ** 2 + g[:, 1] ** 2

    # Normalize h to have L1 norm 1
    l1 = float(np.sum(h))
    if l1 <= 1e-12:
        return 1e10
    h = h / l1

    l2sq = float(np.sum(h ** 2))
    if l2sq <= 1e-12:
        return 1e10

    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma)

    if peak <= 1e-12:
        return 1e10

    ratio = l2sq / peak
    if ratio <= 0.0:
        return 1e10
    return -math.log(ratio)


def _grad_objective_complex(g_flat, sigma=0.1):
    """
    Gradient using finite differences.
    """
    eps = 1e-5
    grad = np.zeros_like(g_flat)
    base_val = _objective_complex(g_flat, sigma)
    for i in range(len(g_flat)):
        g_plus = g_flat.copy()
        g_plus[i] += eps
        val_plus = _objective_complex(g_plus, sigma)
        grad[i] = (val_plus - base_val) / eps
    return grad


def _optimize_complex(g0, time_limit, sigma=0.1):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()
    best_g = g0.copy()
    best_val = _objective_complex(g0, sigma)

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
        res = minimize(
            _objective_complex,
            g0,
            method='L-BFGS-B',
            jac=_grad_objective_complex,
            args=(sigma,),
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 2000, 'disp': False},
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


def _find_sidon_set(n, k, rng):
    """Find a Sidon set of size k in {0, ..., n-1}."""
    # Greedy approach
    S = []
    used_sums = set()
    # Try to pick k elements
    # For simplicity, just pick random distinct and check, or use a known construction.
    # A simple greedy:
    candidates = list(range(n))
    rng.shuffle(candidates)
    
    for c in candidates:
        if len(S) >= k:
            break
        valid = True
        for s in S:
            if c + s in used_sums:
                valid = False
                break
        if valid:
            S.append(c)
            for s in S:
                used_sums.add(c + s)
                # Note: c+c is also a sum
                used_sums.add(2*c)
                
    if len(S) < k:
        # Fallback to random
        S = list(rng.choice(n, size=k, replace=False))
        
    return S


def _initializations():
    rng = np.random.default_rng(42)
    inits = []
    
    # 1. Random complex
    for _ in range(10):
        g_real = rng.randn(N)
        g_imag = rng.randn(N)
        g_flat = np.concatenate([g_real, g_imag])
        inits.append(g_flat)
        
    # 2. Sidon set based (Sparse Time)
    for k in [5, 8, 10, 12]:
        for _ in range(3):
            S = _find_sidon_set(N, k, rng)
            h = np.zeros(N)
            for idx in S:
                h[idx] = 1.0
            # Convert to g
            g_real = np.sqrt(h)
            g_imag = np.zeros(N)
            g_flat = np.concatenate([g_real, g_imag])
            inits.append(g_flat)
            
    # 3. Sparse Frequency (Sum of sinusoids)
    for k in [2, 3, 4, 5]:
        for _ in range(3):
            freqs = rng.choice(N, size=k, replace=False)
            phases = rng.uniform(0, 2*np.pi, size=k)
            g = np.zeros(N, dtype=complex)
            for i, f in enumerate(freqs):
                g += np.exp(1j * phases[i]) * np.exp(1j * 2 * np.pi * f * np.arange(N) / N)
            g_flat = np.concatenate([g.real, g.imag])
            inits.append(g_flat)
            
    # 4. Structured (Gaussian bumps in time)
    for _ in range(5):
        center = rng.uniform(0, N)
        width = rng.uniform(1.0, 5.0)
        x = np.arange(N)
        h = np.exp(-0.5 * ((x - center) / width) ** 2)
        g_real = np.sqrt(h)
        g_imag = np.zeros(N)
        g_flat = np.concatenate([g_real, g_imag])
        inits.append(g_flat)
        
    # 5. Chirp
    for _ in range(3):
        x = np.arange(N)
        chirp = np.exp(1j * 2 * np.pi * (x**2 / (2*N)))
        g_flat = np.concatenate([chirp.real, chirp.imag])
        inits.append(g_flat)

    return inits


def construct():
    global _start
    _start = time.time()
    
    best_h = None
    best_v = 0.0
    
    inits = _initializations()
    
    # Score initializations to prioritize
    scored = []
    for g_flat in inits:
        g = g_flat.reshape(N, 2)
        h = g[:, 0] ** 2 + g[:, 1] ** 2
        v = _ratio_exact(h)
        scored.append((v, g_flat))
        if v > best_v:
            best_v = v
            best_h = h.copy()
            
    scored.sort(key=lambda x: -x[0])
    
    # Optimization loop
    n_starts = len(scored)
    
    for i, (v0, g0) in enumerate(scored):
        if _time_left() <= 5.0:
            break
            
        # Skip if this start is much worse and we have time
        if i > 5 and v0 < best_v * 0.9 and best_v > 10.0:
            continue
            
        t0 = time.time()
        # Allocate time based on remaining budget and number of starts left
        # Give more time to better initializations
        time_budget = min(_time_left() * 0.5, 10.0 + (v0 / best_v if best_v > 0 else 0))
        
        try:
            h_opt = _optimize_complex(g0, time_budget, sigma=0.1)
            h_opt = np.maximum(h_opt, 0.0)
            v_opt = _ratio_exact(h_opt)
            if v_opt > best_v:
                best_v = v_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # Final polish on the best solution found
    if best_h is not None and _time_left() > 5.0:
        # Try to polish in h-space using Nelder-Mead with non-negativity constraint
        # Since h must be non-negative, we can use a penalty or just clip.
        # Clipping in Nelder-Mead can be problematic.
        # Instead, do a few more L-BFGS-B steps from the best g?
        # We don't have the best g stored explicitly, only h.
        # We can reconstruct g from h: g = sqrt(h).
        g_best_real = np.sqrt(best_h)
        g_best_imag = np.zeros(N)
        g_best_flat = np.concatenate([g_best_real, g_best_imag])
        
        t_final = time.time()
        time_budget_final = _time_left() * 0.8
        
        try:
            h_final = _optimize_complex(g_best_flat, time_budget_final, sigma=0.05)
            h_final = np.maximum(h_final, 0.0)
            v_final = _ratio_exact(h_final)
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
    print("ratio =", _ratio_exact(h))
    print("time =", time.time() - t0)
