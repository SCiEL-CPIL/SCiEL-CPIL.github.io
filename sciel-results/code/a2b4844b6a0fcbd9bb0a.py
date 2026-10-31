"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _normalize(h):
    """Enforce exact sum N/2 and bounds [0,1]."""
    h = h.copy()
    for _ in range(20):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _make_periodic(period_pattern):
    """Create a periodic h from a pattern, then normalize."""
    n_rep = N // len(period_pattern) + 1
    h = np.tile(period_pattern, n_rep)[:N]
    return _normalize(h)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    # Phase 0: Try structured periodic patterns
    best_h = None
    best_score = float('inf')
    
    # 2-block baseline
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    half = N // 2
    h0 = np.zeros(N)
    h0[:half] = A
    h0[half:] = B
    s0 = _score(h0)
    if s0 < best_score:
        best_score = s0
        best_h = h0.copy()
    
    # Try various periodic patterns
    np.random.seed(42)
    for period_len in range(2, min(10, N//2 + 1)):
        if time.time() > deadline - 60:
            break
        # Random periodic pattern
        for trial in range(20):
            pattern = np.random.uniform(0.1, 0.9, period_len)
            h_pat = _make_periodic(pattern)
            s_pat = _score(h_pat)
            if s_pat < best_score:
                best_score = s_pat
                best_h = h_pat.copy()
    
    # Phase 1: Smooth optimization from best structured start
    h_final = best_h.copy()
    current_score = best_score
    
    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    
    def _run_smooth_optimization(h_init, t_init, alpha, maxiter, ftol):
        """Run SLSQP on smooth epigraph formulation."""
        def obj(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            smax = _smooth_max(overlap, alpha=alpha)
            penalty = 1000.0 * np.maximum(0.0, smax - t)**2
            return t + penalty
        
        def con_sum(params):
            return np.sum(params[:N]) - N/2.0
        
        def con_epi(params):
            h = params[:N]
            t = params[N]
            shifted = np.zeros(N)
            shifted[:-1] = h[1:]
            overlap = h * (1.0 - shifted)
            return t - np.max(overlap)

        constraints = [
            {'type': 'eq', 'fun': con_sum},
            {'type': 'ineq', 'fun': con_epi}
        ]

        x0 = np.concatenate([h_init, [t_init]])
        result = minimize(
            obj,
            x0,
            method='SLSQP',
            bounds=bounds_ext,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol, 'disp': False}
        )
        return result

    # Phase 2: Multi-stage smooth optimization
    alphas = [50.0, 100.0, 200.0, 500.0, 1000.0]
    for alpha in alphas:
        try:
            if time.time() > deadline - 20:
                break
            t_init = current_score + 0.01
            result = _run_smooth_optimization(h_final, t_init, alpha=alpha, maxiter=300, ftol=1e-12)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
        except Exception:
            pass
    
    # Phase 3: Try multiple random restarts with structured perturbations
    np.random.seed(123)
    for trial in range(15):
        if time.time() > deadline - 10:
            break
        
        # Perturb the current best
        h_pert = h_final.copy()
        noise_scale = 0.05 * (1 - trial / 15.0)
        h_pert += np.random.uniform(-noise_scale, noise_scale, N)
        h_pert = _normalize(h_pert)
        
        pert_score = _score(h_pert)
        if pert_score < current_score + 0.02:
            try:
                t_init = max(pert_score, 0.01) + 0.01
                result = _run_smooth_optimization(h_pert, t_init, alpha=300.0, maxiter=200, ftol=1e-12)
                if result.fun < 100.0:
                    h_opt = _normalize(result.x[:N])
                    new_score = _score(h_opt)
                    if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                        h_final = h_opt
                        current_score = new_score
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    try:
        if time.time() < deadline - 3.0:
            t_init = current_score + 0.001
            result = _run_smooth_optimization(h_final, t_init, alpha=2000.0, maxiter=500, ftol=1e-14)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
    except Exception:
        pass
    
    return h_final
