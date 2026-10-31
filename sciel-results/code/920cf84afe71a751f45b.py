"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Fix sum of h to be exactly N/2, keeping elements in [0,1]."""
    if target is None:
        target = N / 2.0
    h = h.copy()
    diff = target - h.sum()
    if abs(diff) > 1e-15:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        # Second pass for elements not at boundary
        s = h.sum()
        diff2 = target - s
        if abs(diff2) > 1e-15:
            mask = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.any(mask):
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                s = h.sum()
                diff3 = target - s
                if abs(diff3) > 1e-15:
                    mask2 = (h > 1e-12) & (h < 1.0 - 1e-12)
                    if np.any(mask2):
                        h[mask2] += diff3 / np.sum(mask2)
                        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max(x, beta=100.0):
    """Soft max using log-sum-exp."""
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0
    shifted = x - max_x
    return max_x + math.log(np.sum(np.exp(beta * shifted))) / beta


def _overlap_smooth(h, beta=100.0):
    """Smooth surrogate for max overlap."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return _smooth_max(overlap, beta)


def _sinusoidal_overlap(params, n=N):
    """
    Evaluate overlap for a sinusoidal h.
    params: [amp, freq, phase, offset]
    h[i] = clip(offset + amp * sin(freq * i / n * 2*pi + phase), 0, 1)
    """
    amp, freq, phase, offset = params
    i = np.arange(n)
    h = offset + amp * np.sin(freq * i / n * 2.0 * math.pi + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _fix_sum(h)
    return _score(h), h


def _optimize_sinusoidal(deadline):
    """Optimize sinusoidal construction."""
    n = N
    best_score = float('inf')
    best_h = None
    
    # Try various frequency ranges
    for freq in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        if time.time() > deadline - 10:
            break
        
        # For each freq, optimize amp, phase, offset
        def obj_sinusoidal(params, freq=freq):
            amp, phase, offset = params
            if amp < 0 or amp > 1 or offset < -0.5 or offset > 1.5:
                return 1.0
            i = np.arange(n)
            h = offset + amp * np.sin(freq * i / n * 2.0 * math.pi + phase)
            h = np.clip(h, 0.0, 1.0)
            # Check if sum is fixable
            s = h.sum()
            if s < 0.1 * n or s > 0.9 * n:
                return 1.0 + abs(s - n/2.0)
            h = _fix_sum(h)
            return _overlap_smooth(h, beta=200.0)
        
        # Try multiple starting points
        for amp_init in [0.1, 0.3, 0.5, 0.7]:
            for phase_init in [0, math.pi/2, math.pi, 3*math.pi/2]:
                offset_init = 0.5
                try:
                    res = minimize(obj_sinusoidal, np.array([amp_init, phase_init, offset_init]),
                                  method='SLSQP',
                                  bounds=[(0.01, 1.0), (0, 2*math.pi), (0.0, 1.0)],
                                  options={'maxiter': 300, 'ftol': 1e-12})
                    if res.fun < 1.0:
                        amp, phase, offset = res.x
                        i = np.arange(n)
                        h = offset + amp * np.sin(freq * i / n * 2.0 * math.pi + phase)
                        h = np.clip(h, 0.0, 1.0)
                        h = _fix_sum(h)
                        actual = _score(h)
                        if actual < best_score:
                            best_score = actual
                            best_h = h.copy()
                except Exception:
                    pass
    
    return best_h, best_score


def _optimize_full_vector(deadline):
    """
    Optimize the full vector h directly using L-BFGS-B with smooth objective.
    This is a structural change: instead of block or sinusoidal parametric families,
    we optimize all N degrees of freedom directly.
    """
    n = N
    best_score = float('inf')
    best_h = None
    
    # Start from several initial guesses
    # 1. Golden ratio 2-block
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h_init1 = np.zeros(n)
    h_init1[:n//2] = A
    h_init1[n//2:] = B
    h_init1 = _fix_sum(h_init1)
    
    # 2. Sinusoidal
    i = np.arange(n)
    h_init2 = 0.5 + 0.3 * np.sin(2.0 * i / n * 2.0 * math.pi)
    h_init2 = np.clip(h_init2, 0.0, 1.0)
    h_init2 = _fix_sum(h_init2)
    
    # 3. Random
    np.random.seed(42)
    h_init3 = np.random.uniform(0.2, 0.8, n)
    h_init3 = _fix_sum(h_init3)
    
    # 4. Triangle wave
    h_init4 = np.zeros(n)
    for k in range(n):
        h_init4[k] = 0.5 + 0.4 * abs(2 * (k / n - 0.5) - 1)
    h_init4 = _fix_sum(h_init4)
    
    for h_init in [h_init1, h_init2, h_init3, h_init4]:
        if time.time() > deadline - 15:
            break
        
        def obj_full(x, beta=300.0):
            h = np.clip(x, 0.0, 1.0)
            s = h.sum()
            penalty = (s - n/2.0) ** 2 * 50.0
            return _overlap_smooth(h, beta) + penalty
        
        def grad_full(x, beta=300.0):
            h = np.clip(x, 0.0, 1.0)
            s = h.sum()
            # Gradient of penalty
            grad_penalty = 2.0 * (s - n/2.0) * 50.0 / n
            # Gradient of smooth max overlap
            shifted = np.zeros(n)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            max_x = np.max(overlap)
            if max_x < 1e-12:
                grad_smooth = np.zeros(n)
            else:
                shifted_x = overlap - max_x
                weights = np.exp(beta * shifted_x)
                weights /= np.sum(weights)
                # d(overlap)/dh: overlap[i] = h[i] * (1 - shifted[i])
                # shifted[i] = h[i+1] for i < n-1, shifted[n-1] = 0
                # d(overlap[i])/dh[i] = (1 - shifted[i])
                # d(overlap[i])/dh[i+1] = -h[i] for i < n-1
                d_oh_dh = np.zeros(n)
                d_oh_dh[:-1] = 1.0 - h[1:]  # d(overlap[i])/dh[i]
                d_oh_dh[1:] -= h[:-1]        # d(overlap[i])/dh[i+1]
                grad_smooth = weights * d_oh_dh
            return grad_penalty + grad_smooth
        
        try:
            res = minimize(obj_full, h_init, jac=grad_full,
                          method='L-BFGS-B',
                          bounds=[(0, 1)] * n,
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.fun < 1.0:
                h = _fix_sum(res.x)
                actual = _score(h)
                if actual < best_score:
                    best_score = actual
                    best_h = h.copy()
        except Exception:
            pass
    
    return best_h, best_score


def _local_search(h, deadline, steps=2000):
    """
    Binary local search: try flipping individual elements or small neighborhoods
    to reduce the max overlap.
    """
    n = len(h)
    best_h = h.copy()
    best_score = _score(best_h)
    improved = True
    step = 0
    
    while improved and step < steps and time.time() < deadline:
        improved = False
        step += 1
        
        # Try perturbing each element
        for i in range(n):
            if time.time() > deadline:
                break
            
            orig_val = best_h[i]
            
            # Try a few perturbations
            for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                new_val = orig_val + delta
                if new_val < 0 or new_val > 1:
                    continue
                
                candidate = best_h.copy()
                candidate[i] = new_val
                candidate = _fix_sum(candidate)
                
                score = _score(candidate)
                if score < best_score - 1e-12:
                    best_h = candidate
                    best_score = score
                    improved = True
                    break  # Move to next i
    
    return best_h, best_score


def _optimize_4block(deadline):
    """
    Optimize 4-block construction: h = [a]*k1 + [b]*k2 + [c]*k3 + [d]*k4
    where k1+k2+k3+k4 = N and k1*a + k2*b + k3*c + k4*d = N/2.
    """
    n = N
    best_score = float('inf')
    best_h = None
    
    # We'll use differential evolution to find good block lengths and heights
    # Parameters: [k1, k2, k3, a, b, c, d] where k4 = n - k1 - k2 - k3
    # and d is determined by the sum constraint.
    
    def de_obj(params):
        k1 = int(round(params[0]))
        k2 = int(round(params[1]))
        k3 = int(round(params[2]))
        a = params[3]
        b = params[4]
        c = params[5]
        
        k4 = n - k1 - k2 - k3
        if k4 <= 0 or k1 <= 0 or k2 <= 0 or k3 <= 0:
            return 1.0
        
        # d from sum constraint: k1*a + k2*b + k3*c + k4*d = n/2
        d = (n/2.0 - k1*a - k2*b - k3*c) / k4
        if d < 0 or d > 1:
            return 1.0 + abs(max(0, -d) + max(0, d-1))
        
        h = np.zeros(n)
        h[:k1] = a
        h[k1:k1+k2] = b
        h[k1+k2:k1+k2+k3] = c
        h[k1+k2+k3:] = d
        h = _fix_sum(h)
        return _overlap_smooth(h, beta=150.0)
    
    de_bounds = [(1, n//4), (1, n//4), (1, n//4), (0, 1), (0, 1), (0, 1)]
    
    try:
        result = differential_evolution(
            de_obj, de_bounds,
            seed=123, maxiter=200, tol=1e-10,
            popsize=20, mutation=(0.3, 1.0), recombination=0.8,
            polish=True, init='sobol'
        )
        
        if result.fun < 1.0:
            k1 = int(round(result.x[0]))
            k2 = int(round(result.x[1]))
            k3 = int(round(result.x[2]))
            a = result.x[3]
            b = result.x[4]
            c = result.x[5]
            k4 = n - k1 - k2 - k3
            d = (n/2.0 - k1*a - k2*b - k3*c) / k4
            
            if 0 <= d <= 1:
                h = np.zeros(n)
                h[:k1] = a
                h[k1:k1+k2] = b
                h[k1+k2:k1+k2+k3] = c
                h[k1+k2+k3:] = d
                h = _fix_sum(h)
                actual = _score(h)
                if actual < best_score:
                    best_score = actual
                    best_h = h.copy()
    except Exception:
        pass
    
    return best_h, best_score


def construct():
    deadline = time.time() + 135.0  # safety margin before 150s kill
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Sinusoidal optimization (fast, different family)
    if time.time() < deadline - 60:
        hs, ss = _optimize_sinusoidal(deadline - 60)
        if hs is not None and ss < best_score:
            best_score = ss
            best_h = hs
    
    # Phase 2: 4-block optimization
    if time.time() < deadline - 40:
        h4, s4 = _optimize_4block(deadline - 40)
        if h4 is not None and s4 < best_score:
            best_score = s4
            best_h = h4
    
    # Phase 3: Full vector optimization (the key structural change)
    if time.time() < deadline - 25:
        hf, sf = _optimize_full_vector(deadline - 25)
        if hf is not None and sf < best_score:
            best_score = sf
            best_h = hf
    
    # Phase 4: Local search polish on best found
    if time.time() < deadline - 5 and best_h is not None:
        hl, sl = _local_search(best_h, deadline - 5, steps=3000)
        if sl < best_score:
            best_score = sl
            best_h = hl
    
    # Fallback: golden ratio 2-block
    if best_h is None:
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = _fix_sum(h)
    
    return best_h
