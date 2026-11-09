"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-Start CMA-ES with L-BFGS-B polish for Sphere function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin well before 330s limit
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial Sampling
        # We allocate a small portion of the budget to find a good starting point.
        # For Sphere, the global minimum is unique. A shift means the center (0,0,...) 
        # might not be the minimum, but it's a good reference.
        # We sample a few points including the origin.
        
        n_init = 10
        # Ensure we don't use more than budget
        n_init = min(n_init, self.budget // 2) 
        
        # Points to test: Origin + random points
        init_points = [np.zeros(self.dim)]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4.5, 4.5, self.dim))
            
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for p in init_points:
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 10:
                break
            fp = func(p)
            calls_made += 1
            if fp < best_f:
                best_f = fp
                best_x = p.copy()
                
        # If we haven't used much budget, we might want to do a quick local search?
        # No, let's just jump to CMA-ES.
        
        # Phase 2: CMA-ES
        # Remaining budget
        remaining_budget = self.budget - calls_made
        
        # We need to reserve some budget for the final polish (L-BFGS-B)
        # L-BFGS-B on 5D is very fast, maybe 50-100 calls?
        # Let's reserve 100 calls for polish, or 20% of budget.
        polish_budget = min(100, remaining_budget // 5)
        cma_budget = remaining_budget - polish_budget
        
        if cma_budget < 10:
            # Not enough for CMA-ES, just return best
            return best_x, best_f
            
        # CMA-ES Parameters
        # Standard lambda for dim=5 is 4 + floor(3*log(5)) = 4 + 4 = 8
        lam = 4 + int(3 * np.log(self.dim))
        
        # Initialize CMA-ES at the best point found
        x = best_x.copy()
        sigma = 0.5  # Start with a moderate step size, since we are in a basin
        
        # Covariance matrix
        C = np.eye(self.dim)
        # Inverse square root of C
        Cs = np.eye(self.dim)
        # Evolution paths
        ps = np.zeros(self.dim)
        pc = np.zeros(self.dim)
        
        # Adaptation parameters
        cmu = 2 / ((self.dim + 4)**2 + lam)
        ccs = (4 + lam / self.dim) / (self.dim + 4 + 2 * lam / self.dim)
        cc = (2 + 2 * min(1.0, 1 - ccs) + cmu) / (self.dim + 2 + 2 * min(1.0, 1 - ccs))
        cs = (self.dim + 1.3) / (self.dim + 5.5)
        
        # Expected norm of standard normal
        e_norm = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(32*self.dim**2))
        
        # Weights
        mu = lam // 2
        w = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # CMA-ES Loop
        while calls_made < self.budget - polish_budget:
            if time.time() > deadline - 20: # Leave time for polish
                break
                
            # Check if we can afford a full population
            if self.budget - calls_made < lam + polish_budget:
                break
                
            # Generate population
            Z = rng.standard_normal((lam, self.dim))
            Y = Z @ Cs.T
            Pop = x + sigma * Y
            Pop = np.clip(Pop, -5, 5)
            
            # Evaluate population
            F = np.array([func(xi) for xi in Pop])
            calls_made += lam
            
            # Sort by fitness
            idx = np.argsort(F)
            idx_top = idx[:mu]
            
            # Update mean
            x_new = np.sum(w[:, None] * Pop[idx_top], axis=0)
            
            # Update evolution path ps
            diff = x_new - x
            try:
                y = np.linalg.solve(Cs.T, diff)
            except np.linalg.LinAlgError:
                y = diff / (np.linalg.norm(diff) + 1e-12)
            
            ps = (1 - cs) * ps + cs / sigma * np.sqrt(mu_eff) * y
            
            # Update pc
            thresh = (1.4 + 2 / (self.dim + 1)) * e_norm
            if np.linalg.norm(ps) < thresh:
                hsig = 1.0
            else:
                hsig = 0.0
                
            pc = (1 - cc) * pc + hsig * cc * (x_new - x) / sigma
            
            # Update C
            Y_top = (Pop[idx_top] - x[None, :]) / sigma
            C = (1 + cmu * (mu_eff + 2) - cc) * C + cmu * (Y_top.T * w) @ Y_top
            
            # Update sigma
            d_sigma = (self.dim + 2) / 2.0 + min(cmu, 0.5) * (mu_eff / 2.0 - 1.0)
            sigma = sigma * np.exp(cs / d_sigma * (np.linalg.norm(ps) / e_norm - 1.0))
            
            # Update Cs
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                Cs = eigvecs @ np.diag(np.sqrt(eigvals))
            except np.linalg.LinAlgError:
                Cs = np.eye(self.dim)
                C = np.eye(self.dim)
                
            # Update best
            if F[idx[0]] < best_f:
                best_f = F[idx[0]]
                best_x = Pop[idx[0]].copy()
                
            x = x_new
            
        # Phase 3: L-BFGS-B Polish
        # Use the best point found by CMA-ES as the starting point for L-BFGS-B.
        # L-BFGS-B is a gradient-free method in terms of our black-box interface?
        # No, it requires gradients. We don't have gradients.
        # So we cannot use L-BFGS-B directly on the black-box function unless we approximate gradients.
        # Approximating gradients in 5D costs 10 extra calls (forward difference).
        # Is it worth it?
        # The task says "func(x) ... returns a float". It does not provide gradients.
        # So we must use a gradient-free method for the polish as well, or a very fine-grained CMA-ES.
        # Let's use a very fine-grained CMA-ES or a (1+1)-ES with small steps for the polish.
        # Or simply continue CMA-ES with a smaller sigma?
        
        # Actually, let's just run a small (1+1)-ES or a few more CMA-ES steps if budget allows.
        # Given the "polish" phase, let's use a very small step size (1+1)-ES.
        
        # Let's use the remaining budget for a (1+1)-ES with adaptive step size.
        x_polish = best_x.copy()
        f_polish = best_f
        sigma_polish = 0.01 # Very small step
        
        # We have `polish_budget` calls left roughly.
        # Let's use a simple (1+1)-ES loop.
        while calls_made < self.budget:
            if time.time() > deadline - 5:
                break
                
            # Generate one neighbor
            z = rng.standard_normal(self.dim)
            x_trial = x_polish + sigma_polish * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = func(x_trial)
            calls_made += 1
            
            if f_trial < f_polish:
                # Accept
                x_polish = x_trial
                f_polish = f_trial
                # Increase step size slightly? Or keep constant?
                # For convergence, maybe keep constant or decrease slowly.
                # Let's keep constant for simplicity in the final phase.
            else:
                # Reject, decrease step size
                sigma_polish *= 0.9
                if sigma_polish < 1e-6:
                    break # Converged enough
            
            if f_polish < best_f:
                best_f = f_polish
                best_x = x_polish.copy()
                
        return best_x, best_f
# EVOLVE-BLOCK-END
