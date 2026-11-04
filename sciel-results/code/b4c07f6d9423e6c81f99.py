import numpy as np
import scipy.optimize
import time

N = 100
TARGET = N / 2.0

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for the 120s limit

    def get_correlations(h):
        """
        Compute linear correlations of h and 1-h for all non-zero shifts.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude shift 0 (index N-1)
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def compute_all_correlations(h):
        """
        Compute all pairwise correlations for h. Returns (correlations, indices).
        """
        c = np.correlate(h, h, mode='full')
        # Exclude shift 0 (index N-1)
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        Uses bisection on a Lagrange multiplier for exact projection.
        """
        h = np.clip(h, 0.0, 1.0).copy()
        target = N / 2.0
        current = np.sum(h)
        
        if abs(current - target) < 1e-12:
            return h
        
        lo = -2.0
        hi = 2.0
        
        def sum_at_lambda(lam):
            return np.sum(np.clip(h - lam, 0.0, 1.0))
        
        for _ in range(60):
            mid = (lo + hi) / 2.0
            s_mid = sum_at_lambda(mid)
            if abs(s_mid - target) < 1e-12:
                break
            if s_mid > target:
                lo = mid
            else:
                hi = mid
        
        lam = (lo + hi) / 2.0
        h = np.clip(h - lam, 0.0, 1.0)
        
        # Final adjustment for any residual error due to floating point
        residual = target - np.sum(h)
        if abs(residual) > 1e-14:
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        
        return h

    def lse_objective(h, beta):
        """
        LogSumExp of correlations for smooth approximation of max.
        """
        c_all = get_correlations(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse

    def epigraph_objective(h, t, beta):
        """
        Epigraph formulation: minimize t subject to c_i <= t for all i.
        Uses smooth penalty for constraint violations.
        """
        c_all = get_correlations(h)
        violations = np.maximum(0, c_all - t)
        penalty = np.sum(violations ** 2) * beta
        return t + penalty

    def optimize_instance(h0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using L-BFGS-B.
        """
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            h_proj = project_to_feasible(h)
            return lse_objective(h_proj, beta)

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    def optimize_epigraph(h0, t0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using epigraph formulation with L-BFGS-B.
        """
        bounds = [(0, 1) for _ in range(N)] + [(0, N)]
        
        def obj(x):
            h = x[:N]
            t = x[N]
            h_proj = project_to_feasible(h)
            return epigraph_objective(h_proj, t, beta)

        try:
            res = scipy.optimize.minimize(
                obj, 
                np.concatenate([h0, [t0]]), 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x[:N]
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations(h)
        return np.max(c_all)

    def coordinate_descent_refine(h, max_rounds=20, time_budget=10.0):
        """
        Coordinate descent: try small adjustments to each coordinate.
        """
        h = h.copy()
        best_score = score_h(h)
        start_cd = time.time()
        
        for round_num in range(max_rounds):
            if time.time() - start_cd > time_budget:
                break
            improved = False
            for i in range(N):
                if time.time() - start_cd > time_budget:
                    break
                orig_val = h[i]
                # Try small perturbations
                for delta in [-0.02, -0.01, 0.01, 0.02]:
                    h[i] = orig_val + delta
                    h = project_to_feasible(h)
                    new_score = score_h(h)
                    if new_score < best_score - 1e-12:
                        best_score = new_score
                        improved = True
                    else:
                        h[i] = orig_val
                        h = project_to_feasible(h)
            if not improved:
                break
        
        return h, best_score

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)  # N/4 = 25.0

    rng = np.random.default_rng(42)
    
    # Multi-restart strategy with better initialization
    restarts = []
    
    # 1. Uniform start with low beta
    restarts.append((np.full(N, 0.5), 5.0, 400))
    
    # 2-4. Random perturbations with moderate beta
    for i in range(3):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = project_to_feasible(h_rand)
        beta = 10.0 + i * 10.0
        restarts.append((h_rand, beta, 400))
        
    # 5-8. Bimodal initializations with higher beta
    for i in range(4):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.8, 0.2)
        h_rand = project_to_feasible(h_rand)
        beta = 20.0 + i * 10.0
        restarts.append((h_rand, beta, 300))

    # 9-11. Sturmian word candidates
    for i, alpha in enumerate([0.6180339887, 0.7071067812, 0.75]):
        n_vals = np.arange(N)
        h_sturm = np.floor((n_vals + 1) * alpha) - np.floor(n_vals * alpha)
        h_sturm = h_sturm.astype(float)
        h_sturm = project_to_feasible(h_sturm)
        beta = 15.0 + i * 10.0
        restarts.append((h_sturm, beta, 400))

    # 12-14. Sinusoidal initializations
    for i, freq in enumerate([0.05, 0.1, 0.2]):
        n_vals = np.arange(N)
        h_sin = 0.5 + 0.3 * np.sin(2 * np.pi * freq * n_vals)
        h_sin = project_to_feasible(h_sin)
        beta = 10.0 + i * 10.0
        restarts.append((h_sin, beta, 400))

    # Run all restarts
    for idx, (h0, beta, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit - 40:
            break
            
        h_opt = optimize_instance(h0, beta, max_iter)
        current_score = score_h(h_opt)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
            
    # Epigraph optimization on best solution
    if time.time() - start_time < time_limit - 30:
        t0 = best_score
        h_epigraph = optimize_epigraph(best_h, t0, 50.0, 800, ftol=1e-12, gtol=1e-12)
        epigraph_score = score_h(h_epigraph)
        
        if epigraph_score < best_score:
            best_score = epigraph_score
            best_h = h_epigraph.copy()
            
    # Coordinate descent for local refinement
    if time.time() - start_time < time_limit - 20:
        h_cd, cd_score = coordinate_descent_refine(best_h, max_rounds=15, time_budget=15.0)
        if cd_score < best_score:
            best_score = cd_score
            best_h = h_cd.copy()
            
    # High-precision polish pass on the best solution
    if time.time() - start_time < time_limit - 10:
        polish_beta = 80.0
        h_polished = optimize_instance(best_h, polish_beta, 1000, ftol=1e-12, gtol=1e-12)
        polished_score = score_h(h_polished)
        
        if polished_score < best_score:
            best_score = polished_score
            best_h = h_polished.copy()
            
    # Final coordinate descent
    if time.time() - start_time < time_limit - 5:
        h_cd2, cd2_score = coordinate_descent_refine(best_h, max_rounds=10, time_budget=10.0)
        if cd2_score < best_score:
            best_score = cd2_score
            best_h = h_cd2.copy()
            
    # Final constraint verification
    best_h = project_to_feasible(best_h)
    
    return best_h
