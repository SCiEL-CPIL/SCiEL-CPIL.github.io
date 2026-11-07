"""Combined strategy: Deterministic axis sweep for initialization, followed by CMA-ES for anisotropic optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for BBOB f11 (Discus) function.
    Strategy:
    1. Dense deterministic axis sweep to identify the anisotropic 'needle' direction and approximate minimum location.
    2. CMA-ES to exploit the anisotropy and converge to the precise minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: 300s limit, use 280s
        deadline = time.time() + 280
        
        # Fixed seed for consistent behavior across instances if needed, 
        # but random seed might explore better if instances vary. 
        # Given the "hidden seeded shift", the landscape shape is fixed (Discus), 
        # just shifted/rotated. A fixed internal RNG ensures we don't get "unlucky" 
        # random restarts, but CMA-ES is deterministic given the start.
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Deterministic Axis Sweep ---
        # Discus f11 is highly anisotropic. The gradient along the "needle" axis is much steeper.
        # Evaluating along the axes helps identify the needle and the sign of the shift.
        
        # Evaluate origin
        evaluate(np.zeros(dim))

        # Sweep along each axis
        # Radii chosen to cover the domain [-5, 5] efficiently.
        # Since the minimum is in [-4, 4], we focus there but check edges.
        radii = [1.0, 2.0, 3.0, 4.0]
        
        for axis in range(dim):
            for r in radii:
                # Positive direction
                x_pos = np.zeros(dim)
                x_pos[axis] = r
                evaluate(x_pos)
                
                # Negative direction
                if calls < budget and time.time() < deadline:
                    x_neg = np.zeros(dim)
                    x_neg[axis] = -r
                    evaluate(x_neg)
            
            # Check budget/time before next axis
            if calls >= budget or time.time() > deadline:
                break

        # If no valid evaluation happened (shouldn't happen), fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)
            if best_f == np.inf:
                return best_x, 0.0 # Invalid, but return something

        # --- Phase 2: CMA-ES ---
        # Use CMA-ES to refine the solution found by the sweep.
        # CMA-ES adapts the covariance matrix to the local geometry, which is ideal for Discus.
        
        if calls < budget and time.time() < deadline:
            x0 = best_x.copy()
            # Initial sigma: 
            # Too small -> slow adaptation to anisotropy.
            # Too large -> wastes evals exploring far away.
            # 1.5 is a reasonable guess for a domain of size 10.
            sigma0 = 1.5
            
            self._run_cma_es(func, x0, sigma0, budget, deadline, rng, evaluate, best_x, best_f)

        return best_x, best_f

    def _run_cma_es(self, func, x0, sigma0, budget, deadline, rng, evaluate, best_x, best_f):
        """
        Runs CMA-ES optimization.
        Note: We pass 'evaluate' to ensure consistent counting and tracking.
        """
        dim = self.dim
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of recombination points
        
        # Weights
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_c = min(c_c, 1.0)
        mueff_inv = 1.0 / mueff
        c_1 = (2 - mueff_inv + c_sigma) / ((dim + 1.3) ** 2 + mueff_inv + c_sigma)
        c_mu = (2 * (mueff_inv - 1) + c_sigma) / ((dim + 2) ** 2 + mueff_inv + c_sigma)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff_inv - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        x_mean = x0.copy()
        sigma = sigma0
        evals = 0
        
        # Initial population
        pop_x = []
        pop_f = []
        for i in range(lambda_):
            if evals >= budget - self.budget + 0: # Just a dummy check, real check below
                break
            z = rng.standard_normal(dim)
            xi = x_mean + sigma * (B @ (D * z))
            fi = evaluate(xi)
            if fi == np.inf:
                break
            pop_x.append(xi)
            pop_f.append(fi)
            evals += 1
            if evals >= budget - 0: # We are using global budget via 'evaluate'
                break

        # The 'evaluate' function tracks global calls. 
        # We need to know how many calls were made *in this CMA-ES phase* to stop correctly.
        # Actually, 'evaluate' returns np.inf if budget is exceeded. 
        # We can just loop while 'evaluate' doesn't return inf and time is ok.
        
        # Reset local evals counter for CMA-ES logic if needed, but we rely on global 'calls' 
        # which is not directly accessible here without nonlocal. 
        # Better approach: Pass a mutable counter or just rely on the fact that 
        # 'evaluate' will start returning inf when budget is hit.
        
        # However, CMA-ES logic needs to know if it should stop generating new populations.
        # If the first population generation hits the budget limit, we stop.
        
        if len(pop_x) < 2:
            return

        idx = np.argsort(pop_f)
        pop_x = [pop_x[i] for i in idx]
        pop_f = [pop_f[i] for i in idx]

        while time.time() < deadline:
            # Check if we have budget left. 
            # Since 'evaluate' increments a global counter, and we don't have access to it here 
            # (it's in the outer scope), we must assume that if 'evaluate' returns inf, we stop.
            # But we need to generate the *next* population only if we have budget.
            # How do we know if we have budget? 
            # We can capture the call count before and after.
            
            # Let's refactor slightly: We will check if the last evaluation returned inf.
            # If the population generation loop broke due to inf, we should stop.
            
            # A simpler check: If the last few evaluations were inf, stop.
            # Or, just try to generate. If the first point of the new population returns inf, we stop.
            
            # Update CMA-ES parameters based on current population
            w = weights[:min(mu, len(pop_f))]
            x_mean_new = np.sum(np.array(pop_x[:len(w)]) * w[:, None], axis=0)

            # Update translation (cm)
            # Note: sigma * sqrt(sum(D^2)) is the norm of the standard deviation vector
            sigma_norm = sigma * np.sqrt(np.sum(D ** 2))
            if sigma_norm > 1e-12:
                pc = (1 - c_c) * pc + c_c * np.sqrt(mueff) / sigma_norm * (x_mean_new - x_mean)
            else:
                pc = (1 - c_c) * pc
            
            hs = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (evals + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            
            # Update covariance
            # C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hs) * cc * (2 - cc) * C) + cmu * sum(w_i * (x_i - x_mean) @ (x_i - x_mean).T)
            C = (1 - c_1 - c_mu) * C 
            C += c_1 * (np.outer(pc, pc) + (1 - hs) * c_c * (2 - c_c) * C)
            
            # Add rank-mu update
            # pop_x is sorted, so first mu are the best
            if len(pop_x) >= mu:
                pop_best = np.array(pop_x[:mu])
                diff = pop_best - x_mean
                C += c_mu * np.sum(w[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)

            # Eigendecomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-10))
            B = B * np.sqrt(D[:, None])

            # Update sigma (cs)
            if sigma_norm > 1e-12:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma_norm * (x_mean_new - x_mean)
            else:
                ps = (1 - c_sigma) * ps
            
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1e6)
            sigma = max(sigma, 1e-12)

            x_mean = x_mean_new

            # Generate new population
            pop_x_new = []
            pop_f_new = []
            budget_exhausted = False
            
            for i in range(lambda_):
                if time.time() > deadline:
                    budget_exhausted = True
                    break
                z = rng.standard_normal(dim)
                xi = x_mean + sigma * (B @ z)
                fi = evaluate(xi)
                if fi == np.inf:
                    budget_exhausted = True
                    break
                pop_x_new.append(xi)
                pop_f_new.append(fi)
                evals += 1

            if budget_exhausted or len(pop_x_new) < 2:
                break

            idx_new = np.argsort(pop_f_new)
            pop_x = [pop_x_new[i] for i in idx_new]
            pop_f = [pop_f_new[i] for i in idx_new]

# EVOLVE-BLOCK-END
