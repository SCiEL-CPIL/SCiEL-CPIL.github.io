"""Combined approach: parameterized chirp+harmonics with DE global search + L-BFGS-B local polish."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 400


def compute_value(h):
    """Compute the task value: 2*N*max(conv(h,h))/sum(h)^2"""
    h = np.asarray(h, dtype=float)
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s ** 2))


def make_h_from_params(params):
    """
    Convert parameter vector to step heights h of length N.
    params layout:
      [phase0, chirp_rate, amp1, ph1, freq1, amp2, ph2, freq2,
       amp3, ph3, freq3, amp4, ph4, freq4, dc_offset]
    Total 15 parameters.
    """
    t = np.arange(N, dtype=float) / N  # normalized time in [0,1)
    
    # Chirp component: cos(phase0 + pi * chirp_rate * t^2)
    phase = params[0] + np.pi * params[1] * t ** 2
    h = np.cos(phase)
    
    # Add 4 DFT harmonics
    for j in range(4):
        base = 2 + 3 * j
        amp = params[base]
        ph = params[base + 1]
        freq = params[base + 2]
        h += amp * np.cos(ph + 2.0 * np.pi * freq * t)
    
    # DC offset
    h += params[14]
    
    return h


def objective_from_params(params):
    """Objective function operating on parameter space."""
    h = make_h_from_params(params)
    return compute_value(h)


def construct():
    t_start = time.time()
    DEADLINE = t_start + 135.0  # safety margin before 150s kill
    
    def remaining():
        return DEADLINE - time.time()
    
    # Initial best: seed
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # 2.0
    
    # --- Phase 1: Global search with differential evolution on parameter space ---
    bounds = [
        (0, 2 * np.pi),       # phase0
        (-20, 20),            # chirp_rate
        (-1.5, 1.5),          # amp1
        (0, 2 * np.pi),       # ph1
        (0.01, 50.0),         # freq1
        (-1.5, 1.5),          # amp2
        (0, 2 * np.pi),       # ph2
        (0.01, 50.0),         # freq2
        (-1.5, 1.5),          # amp3
        (0, 2 * np.pi),       # ph3
        (0.01, 50.0),         # freq3
        (-1.5, 1.5),          # amp4
        (0, 2 * np.pi),       # ph4
        (0.01, 50.0),         # freq4
        (-3.0, 3.0),          # dc_offset
    ]
    
    if remaining() > 40.0:
        try:
            res_de = differential_evolution(
                objective_from_params,
                bounds,
                seed=42,
                maxiter=150,
                popsize=12,
                tol=1e-6,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1,
            )
            if res_de.fun < best_val:
                best_val = res_de.fun
                best_h = make_h_from_params(res_de.x)
        except Exception:
            pass
    
    # --- Phase 2: Multi-restart Nelder-Mead on parameter space ---
    rng = np.random.default_rng(123)
    n_restarts = 10
    for i in range(n_restarts):
        if remaining() < 15.0:
            break
        
        # Generate random parameters within bounds
        params_rand = np.array([
            rng.uniform(0, 2 * np.pi),
            rng.uniform(-20, 20),
            rng.uniform(-1.5, 1.5),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0.01, 50.0),
            rng.uniform(-1.5, 1.5),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0.01, 50.0),
            rng.uniform(-1.5, 1.5),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0.01, 50.0),
            rng.uniform(-1.5, 1.5),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0.01, 50.0),
            rng.uniform(-3.0, 3.0),
        ])
        
        try:
            res_nm = minimize(
                objective_from_params,
                params_rand,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True},
            )
            if res_nm.fun < best_val:
                best_val = res_nm.fun
                best_h = make_h_from_params(res_nm.x)
        except Exception:
            pass
    
    # --- Phase 3: L-BFGS-B local polish directly on h (400-dim) ---
    if remaining() > 20.0:
        # Use soft-max smoothed objective for differentiability
        def soft_max_obj(h, beta=500.0):
            s = np.sum(h)
            if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
                return 1e10
            conv = np.convolve(h, h)
            m = np.max(conv)
            if not np.isfinite(m):
                return 1e10
            peak = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
            return abs(2.0 * N * peak) / (s ** 2)
        
        h0 = best_h.copy()
        try:
            res_lbfgs = minimize(
                lambda x: soft_max_obj(x, 500.0),
                h0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            val_new = compute_value(res_lbfgs.x)
            if val_new < best_val:
                best_val = val_new
                best_h = res_lbfgs.x.copy()
        except Exception:
            pass
    
    # --- Phase 4: Final refinement with higher beta soft-max ---
    if remaining() > 10.0:
        def soft_max_obj_high(h, beta=2000.0):
            s = np.sum(h)
            if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
                return 1e10
            conv = np.convolve(h, h)
            m = np.max(conv)
            if not np.isfinite(m):
                return 1e10
            peak = m + np.log(np.sum(np.exp(beta * (conv - m)))) / beta
            return abs(2.0 * N * peak) / (s ** 2)
        
        h0 = best_h.copy()
        try:
            res_lbfgs2 = minimize(
                lambda x: soft_max_obj_high(x, 2000.0),
                h0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            val_new = compute_value(res_lbfgs2.x)
            if val_new < best_val:
                best_val = val_new
                best_h = res_lbfgs2.x.copy()
        except Exception:
            pass
    
    # --- Phase 5: Coordinate descent polish on exact objective ---
    if remaining() > 5.0:
        h = best_h.copy()
        for _ in range(30):
            if remaining() < 2.0:
                break
            improved = False
            # Sample a subset of coordinates for speed
            indices = np.random.default_rng(_).choice(N, size=min(100, N), replace=False)
            for i in indices:
                if remaining() < 2.0:
                    break
                base = h[i]
                for step in [0.001, -0.001, 0.005, -0.005, 0.0001, -0.0001, 0.01, -0.01]:
                    h[i] = base + step
                    v = compute_value(h)
                    if v < best_val:
                        best_val = v
                        improved = True
                        break
                else:
                    h[i] = base
            if not improved:
                break
        best_h = h
    
    # Ensure validity
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h


if __name__ == "__main__":
    h = construct()
    print(f"Value: {compute_value(h)}")
    print(f"Score: {2.0 / compute_value(h)}")
