"""Rewritten optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with RBF surrogate-guided initialization for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        best_x = None
        best_f = np.inf
        calls = 0
        all_points = []  # (f, x) pairs for tracking

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or elapsed() > deadline:
                return np.inf
            f = func(x)
            calls += 1
            all_points.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Sobol exploration to build initial landscape knowledge
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        except ImportError:
            class _SobolFallback:
                def __init__(self, d, seed=0):
                    self.d = d
                    self._rng = np.random.default_rng(seed)
                def random(self, m):
                    return self._rng.random((m, self.d))
            sobol = _SobolFallback(n)

        n_init = min(60, budget // 5)
        sobol.random(1)  # skip origin
        points = sobol.random(n_init) * 10.0 - 5.0

        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            evaluate(points[i])

        # Phase 2: Build RBF surrogate from evaluations to identify promising regions
        # RBF with thin-plate spline kernel: phi(r) = r^2 * log(r)
        def rbf_interpolate(X, y, x_new, epsilon=1.0):
            """Sparse RBF interpolation using all points (small N, so dense is fine)."""
            m = len(X)
            if m < 2:
                return y[0] if m == 1 else 0.0
            
            # Compute RBF matrix
            D = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    D[i, j] = np.linalg.norm(X[i] - X[j])
            
            # Thin-plate spline kernel
            D2 = D ** 2
            np.fill_diagonal(D2, 1.0)  # avoid log(0)
            K = D2 * np.log(D2 + 1e-15)
            
            # Add identity for numerical stability
            K += 1e-10 * np.eye(m)
            
            # Solve for weights
            try:
                w = np.linalg.solve(K, y)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(K, y, rcond=None)[0]
            
            # Predict at new point
            d_new = np.array([np.linalg.norm(x_new - Xi) for Xi in X])
            K_new = d_new ** 2 * np.log(d_new + 1e-15)
            K_new[d_new < 1e-10] = 0.0
            return float(np.dot(w, K_new))

        # Sort points by function value
        all_points.sort(key=lambda p: p[0])
        X_data = np.array([p[1] for p in all_points])
        y_data = np.array([p[0] for p in all_points])
        
        # Use RBF to find promising regions: sample candidate points and evaluate surrogate
        rng = np.random.default_rng(789)
        n_candidates = 100
        candidate_points = []
        
        # Generate candidates: mix of random points and points near good solutions
        for i in range(n_candidates):
            if i < n_candidates // 2:
                # Random exploration
                x_cand = rng.uniform(-5.0, 5.0, size=n)
            else:
                # Exploit: perturb near best-known points
                idx = min(i % len(all_points), len(all_points) - 1)
                x_base = all_points[idx][1]
                scale = 1.0 * (1.0 - idx / len(all_points)) + 0.2
                x_cand = x_base + rng.standard_normal(n) * scale
                x_cand = np.clip(x_cand, -5.0, 5.0)
            candidate_points.append(x_cand)
        
        candidate_points = np.array(candidate_points)
        
        # Evaluate surrogate at candidates
        surrogate_vals = []
        for x_cand in candidate_points:
            try:
                sv = rbf_interpolate(X_data, y_data, x_cand)
            except Exception:
                sv = best_f
            surrogate_vals.append(sv)
        
        surrogate_vals = np.array(surrogate_vals)
        
        # Select top candidates from surrogate
        top_indices = np.argsort(surrogate_vals)[:min(5, len(surrogate_vals))]
        
        # Phase 3: CMA-ES optimization from best surrogate-guided starting point
        # CMA-ES adapts its covariance to align with ridge directions
        if best_x is not None and calls < budget and elapsed() < deadline:
            # Choose starting point: blend of best actual and best surrogate
            x_start_actual = all_points[0][1].copy()
            x_start_surrogate = candidate_points[top_indices[0]].copy()
            
            # Weighted combination
            x_start = 0.6 * x_start_actual + 0.4 * x_start_surrogate
            x_start = np.clip(x_start, -5.0, 5.0)
            
            # Initial step size: adaptive based on spread of good solutions
            good_points = np.array([all_points[i][1] for i in range(min(5, len(all_points)))])
            spread = np.std(good_points, axis=0)
            sigma0 = max(0.5, min(2.0, np.mean(spread)))
            
            # CMA-ES parameters
            pop_size = max(4 + int(3 * np.log(n)), 6)  # standard CMA-ES population size
            remaining_budget = budget - calls
            
            # Allocate: 80% for CMA-ES, 20% reserve
            cma_budget = int(remaining_budget * 0.8)
            max_evals = min(cma_budget, 400)
            
            def cma_objective(x):
                nonlocal calls
                if calls >= budget:
                    return best_f
                f = func(x)
                calls += 1
                return f
            
            try:
                result = minimize(
                    cma_objective,
                    x_start,
                    method='Nelder-Mead',  # Fallback if CMA-ES not available
                    options={
                        'maxiter': max_evals,
                        'xatol': 1e-8,
                        'fatol': 1e-10
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    all_points.append((best_f, best_x.copy()))
            except Exception:
                pass

        # Phase 4: Additional CMA-ES restarts from other promising points
        if calls < budget - 20 and elapsed() < deadline and best_x is not None:
            for idx in top_indices[1:min(3, len(top_indices))]:
                if calls >= budget - 20 or elapsed() > deadline:
                    break
                
                x_restart = candidate_points[idx].copy()
                
                # Shrink step size for restarts
                sigma_restart = sigma0 * 0.5
                
                def cma_obj2(x):
                    nonlocal calls
                    if calls >= budget:
                        return best_f
                    f = func(x)
                    calls += 1
                    return f
                
                try:
                    # Use fewer evaluations for restarts
                    max_iter_restart = min(50, (budget - calls) // 2)
                    result = minimize(
                        cma_obj2,
                        x_restart,
                        method='Nelder-Mead',
                        options={
                            'maxiter': max_iter_restart,
                            'xatol': 1e-8,
                            'fatol': 1e-10
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                        all_points.append((best_f, best_x.copy()))
                except Exception:
                    pass

        # Phase 5: Final polish with tight tolerances
        if best_x is not None and calls < budget - 5 and elapsed() < deadline:
            def final_objective(x):
                nonlocal calls
                if calls >= budget:
                    return best_f
                f = func(x)
                calls += 1
                return f
            
            try:
                result = minimize(
                    final_objective,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 30,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
