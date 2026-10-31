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
    for _ in range(10):
        s = np.sum(h)
        diff = N/2.0 - s
        if abs(diff) < 1e-15:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_objective(params, alpha):
    """Smooth epigraph objective: minimize t + penalty for violating t >= smooth_max(overlap)."""
    h = params[:N]
    t = params[N]
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha=alpha)
    penalty = 1000.0 * np.maximum(0.0, smax - t)**2
    return t + penalty

def _smooth_constraints(params):
    """Constraints: sum(h) = N/2, t >= max(overlap)."""
    h = params[:N]
    t = params[N]
    sum_c = np.sum(h) - N/2.0
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    epi_c = t - np.max(overlap)
    return sum_c, epi_c

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Initial guess: 2-block solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()

    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

    def _run_smooth_optimization(h_init, t_init, alpha, maxiter, ftol):
        """Run SLSQP on smooth epigraph formulation."""
        def obj(params):
            return _smooth_objective(params, alpha)
        
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

    # Phase 1: Smooth optimization with moderate alpha
    try:
        if time.time() < deadline - 30.0:
            result = _run_smooth_optimization(h_final, current_score + 0.05, alpha=100.0, maxiter=500, ftol=1e-12)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
    except Exception:
        pass

    # Phase 2: Smooth optimization with higher alpha (tighter approximation)
    try:
        if time.time() < deadline - 20.0:
            result = _run_smooth_optimization(h_final, current_score + 0.01, alpha=300.0, maxiter=500, ftol=1e-13)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
    except Exception:
        pass

    # Phase 3: High-precision polish with very high alpha
    try:
        if time.time() < deadline - 10.0:
            result = _run_smooth_optimization(h_final, current_score + 0.001, alpha=1000.0, maxiter=1000, ftol=1e-15)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
    except Exception:
        pass

    # Phase 4: Structured random restarts with smooth optimization
    try:
        np.random.seed(42)
        n_restarts = 8
        for trial in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            
            # Create diverse structured initial points
            if trial % 3 == 0:
                # Periodic pattern with random period
                period = np.random.randint(2, N//4 + 1)
                base = np.random.uniform(0.2, 0.8, period)
                h_rand = np.tile(base, N // period + 1)[:N]
            elif trial % 3 == 1:
                # Two-block with random split
                split = np.random.randint(N//4, 3*N//4)
                a_val = np.random.uniform(0.2, 0.8)
                b_val = np.random.uniform(0.2, 0.8)
                h_rand = np.zeros(N)
                h_rand[:split] = a_val
                h_rand[split:] = b_val
            else:
                # Smooth random with low-frequency structure
                freqs = np.random.uniform(0.5, 3.0, 3)
                phases = np.random.uniform(0, 2*np.pi, 3)
                amplitudes = np.random.uniform(0.1, 0.3, 3)
                x = np.arange(N) / N
                h_rand = 0.5
                for f, p, a in zip(freqs, phases, amplitudes):
                    h_rand += a * np.sin(2*np.pi*f*x + p)
                h_rand = np.clip(h_rand, 0.0, 1.0)
            
            h_rand = _normalize(h_rand)
            rand_score = _score(h_rand)
            
            # Only optimize if this starting point is promising
            if rand_score < current_score + 0.05:
                t_init = max(rand_score, 0.01) + 0.02
                result = _run_smooth_optimization(h_rand, t_init, alpha=200.0, maxiter=300, ftol=1e-12)
                if result.fun < 100.0:
                    h_opt = _normalize(result.x[:N])
                    new_score = _score(h_opt)
                    if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                        h_final = h_opt
                        current_score = new_score
    except Exception:
        pass

    # Final: one more polish on the best found
    try:
        if time.time() < deadline - 2.0:
            result = _run_smooth_optimization(h_final, current_score + 0.0001, alpha=500.0, maxiter=500, ftol=1e-14)
            if result.fun < 100.0:
                h_opt = _normalize(result.x[:N])
                new_score = _score(h_opt)
                if new_score < current_score and np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                    h_final = h_opt
                    current_score = new_score
    except Exception:
        pass

    return h_final
