# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Tchebycheff decomposition with SLSQP local optimization for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random  # evaluator sets np.random.seed(seed) before calling

        # ------------------------------------------------------------------
        # Evaluation bookkeeping
        # ------------------------------------------------------------------
        evaluated = []          # list of (x, f)
        eval_set = set()
        eval_count = 0

        def evaluate(x):
            """Evaluate x (clipped to bounds). Returns f or None if duplicate/no budget."""
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
            key = tuple(np.round(x, 9))
            if key in eval_set:
                return None
            f = np.array(problem(x), dtype=np.float64)
            if not np.all(np.isfinite(f)):
                eval_set.add(key)
                return None
            eval_set.add(key)
            evaluated.append((x, f))
            eval_count += 1
            return f

        # ------------------------------------------------------------------
        # Phase 1: Boundary + stratified initialization
        # ------------------------------------------------------------------
        N = 40
        n_init = min(N, budget)

        init_X = []
        # (a) boundary points: each dimension at its low or high end (2^dim corners)
        for mask in range(1 << dim):
            x = lo.copy()
            for d in range(dim):
                if (mask >> d) & 1:
                    x[d] = hi[d]
            init_X.append(x)
        # (b) fill remaining with stratified (LHS-like) interior points
        n_corners = len(init_X)
        n_fill = n_init - n_corners
        for i in range(max(0, n_fill)):
            x = np.zeros(dim)
            for d in range(dim):
                frac = (i + rng.random()) / max(1, n_fill)
                x[d] = lo[d] + frac * (hi[d] - lo[d])
            init_X.append(x)

        init_F = []
        for x in init_X:
            f = evaluate(x)
            if f is not None:
                init_F.append(f)
            if eval_count >= budget:
                break

        if eval_count >= budget or not init_F:
            return evaluated

        X = np.array([ex for ex, _ in evaluated[:len(init_F)]], dtype=np.float64)
        F_vals = np.array(init_F, dtype=np.float64)

        # ------------------------------------------------------------------
        # Phase 2: Tchebycheff Decomposition with SLSQP
        # ------------------------------------------------------------------
        # Define Tchebycheff weight vectors
        # For 2 objectives, weights are (w1, w2) with w1 + w2 = 1
        # We use 10 weights to cover the trade-off space
        n_weights = 10
        weights = []
        for i in range(n_weights):
            w1 = 0.1 + 0.8 * i / (n_weights - 1)
            w2 = 1.0 - w1
            weights.append(np.array([w1, w2]))
        
        # Smooth Tchebycheff objective function
        # min max_i { w_i * (f_i(x) - rho) }
        # Approximate max with log-sum-exp for differentiability
        # max(z) ≈ (1/p) * log(sum(exp(p*z)))
        p_smooth = 100.0
        
        def tchebycheff_obj(x, w):
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            if not np.all(np.isfinite(f)):
                return 1e9
            # rho is the ideal point (minimum of each objective)
            # We can estimate rho from the current archive or use 0
            # Using 0 is safer if objectives are positive
            rho = np.array([0.0, 0.0]) 
            # Actually, let's use the current min of each objective in the archive as a better rho
            # But that changes the objective during optimization. 
            # Standard Tchebycheff uses a fixed rho, often the ideal point.
            # Let's use the min observed so far for each objective.
            min_f = np.min(F_vals, axis=0)
            z = w * (f - min_f)
            # Log-sum-exp for stability
            max_z = np.max(z)
            if max_z > 700: # prevent overflow
                return float(max_z / p_smooth)
            log_sum_exp = max_z + np.log(np.sum(np.exp(p_smooth * (z - max_z))))
            return float(log_sum_exp / p_smooth)

        # For each weight, find the best starting point and optimize
        for w in weights:
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Find the best starting point in the archive for this weight
            # We want to minimize the Tchebycheff value for this weight
            # Calculate Tchebycheff value for all current points
            min_f = np.min(F_vals, axis=0)
            z_vals = w * (F_vals - min_f)
            tcheb_vals = np.max(z_vals, axis=1)
            best_idx = np.argmin(tcheb_vals)
            x0 = X[best_idx].copy()
            
            # Run SLSQP
            try:
                res = minimize(tchebycheff_obj, x0, args=(w,), method='SLSQP',
                               bounds=list(zip(lo, hi)),
                               options={'maxiter': 20, 'ftol': 1e-8})
                if res.success and res.fun < 1e8:
                    evaluate(res.x)
            except Exception:
                pass

            # Optionally, try a second starting point (e.g., a random one or another corner)
            # if budget allows
            if eval_count < budget - 5 and time.time() < deadline:
                # Pick a random point from the archive that is not the best
                other_indices = [i for i in range(len(X)) if i != best_idx]
                if other_indices:
                    x0_2 = X[rng.choice(other_indices)].copy()
                    try:
                        res2 = minimize(tchebycheff_obj, x0_2, args=(w,), method='SLSQP',
                                       bounds=list(zip(lo, hi)),
                                       options={'maxiter': 10, 'ftol': 1e-8})
                        if res2.success and res2.fun < 1e8:
                            evaluate(res2.x)
                    except Exception:
                        pass

        return evaluated
# EVOLVE-BLOCK-END
