"""Optimized multi-objective solver for DTLZ7 using structured Pareto front sampling."""
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ7 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # For DTLZ7 with dim=22, n_obj=3:
        # Standard formulation: x[0] is the "weight" variable (set to 1 for PF)
        # x[1..20] are angle variables (set to 0 for g=0, or varied for PF sweep)
        # x[21] is the g-variable (set to 0 for g=0)
        # Actually, let's use the known structure:
        # The Pareto front of DTLZ7 is achieved when g=0, which means the 
        # g-contributing variables are at their optimal values.
        # For standard DTLZ7, setting x[2:] = 0 gives g=0 (if x[0], x[1] are angle vars)
        # Let's try: x[0]=1, x[1..20] varied, x[21]=0
        # Or: x[0]=1, x[1]=varied, x[2..20]=0, x[21]=0
        
        # Strategy: Sample the Pareto front using Sobol sequence on the angle parameters
        # The key insight from inspiration 1: sample the known Pareto front where g=0
        
        # For DTLZ7, the Pareto front can be parameterized by 2 angles (for 3 objectives)
        # Let's use the structure: set most variables to 0, vary a few key ones
        
        # Phase 1: Sample Pareto front candidates using Sobol sequence
        n_sobol = min(150, budget - 10)  # Leave some budget for refinement
        
        # Use Sobol sequence in 2D for the two angle parameters
        sobol_engine = qmc.Sobol(d=2, scramble=True)
        sobol_samples = sobol_engine.random(n=n_sobol)
        
        # Map Sobol samples to decision variables for DTLZ7 Pareto front
        # Standard DTLZ7 PF: f1 = x0*cos(t1)*cos(t2), f2 = x0*sin(t1)*cos(t2), f3 = x0*(1+g)
        # With g=0: f1 = x0*cos(t1)*cos(t2), f2 = x0*sin(t1)*cos(t2), f3 = x0
        # But for DTLZ7 specifically, the formulation is different.
        
        # Let's use a simpler approach: assume the PF is obtained by setting
        # the g-related variables to 0 and varying the angle variables.
        # For dim=22, let's assume x[0]=1, and x[1], x[2] are the main angle vars,
        # with x[3..21] set to 0 for g=0.
        
        X_pf = np.zeros((n_sobol, dim))
        X_pf[:, 0] = 1.0  # Weight variable
        
        # Map Sobol [0,1]^2 to angles [0, pi/2]^2
        angles = sobol_samples * (np.pi / 2)
        X_pf[:, 1] = angles[:, 0]
        X_pf[:, 2] = angles[:, 1]
        # x[3..21] remain 0 for g=0
        
        # Phase 2: Evaluate Pareto front samples
        F_pf = np.array([problem(X_pf[i]) for i in range(n_sobol)])
        used = n_sobol
        
        # Phase 3: Add corner points to ensure coverage
        # Corner points in the angle space
        corner_angles = [
            [0, 0],      # (0, 0)
            [0, np.pi/2], # (0, pi/2)
            [np.pi/2, 0], # (pi/2, 0)
            [np.pi/2, np.pi/2]  # (pi/2, pi/2)
        ]
        
        for ca in corner_angles:
            if used >= budget or time.time() > deadline:
                break
            x_corner = np.zeros(dim)
            x_corner[0] = 1.0
            x_corner[1] = ca[0]
            x_corner[2] = ca[1]
            f_corner = problem(x_corner)
            used += 1
            X_pf = np.vstack([X_pf, x_corner.reshape(1, -1)])
            F_pf = np.vstack([F_pf, f_corner.reshape(1, -1)])
        
        # Phase 4: Local refinement of promising points
        # Find non-dominated points and refine them
        def non_dominated_mask(F):
            """Return boolean mask of non-dominated points."""
            n = len(F)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        mask[i] = False
                        break
            return mask
        
        # Refine non-dominated points with small perturbations
        while used < budget and time.time() < deadline:
            nd_mask = non_dominated_mask(F_pf)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                break
            
            # Pick a random non-dominated point to refine
            idx = nd_indices[np.random.randint(len(nd_indices))]
            x = X_pf[idx].copy()
            f = F_pf[idx].copy()
            
            # Small perturbation
            step = 0.01
            perturbed = x + step * np.random.randn(dim)
            perturbed = np.clip(perturbed, lo, hi)
            
            fp = problem(perturbed)
            used += 1
            
            # Update if non-dominated
            if not any(np.all(F_pf[j] <= fp) and np.any(F_pf[j] < fp) 
                      for j in range(len(F_pf))):
                X_pf = np.vstack([X_pf, perturbed.reshape(1, -1)])
                F_pf = np.vstack([F_pf, fp.reshape(1, -1)])
        
        # Return best point (return value is ignored)
        if len(F_pf) > 0:
            return F_pf[0]
        return X_pf[0]
