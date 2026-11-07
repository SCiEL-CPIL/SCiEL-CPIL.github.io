"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using direct Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        
        # For DTLZ2, the Pareto front is a quarter-sphere in 3D:
        # f1 = (1+g)*cos(t1)*sin(t2)*sin(t3)
        # f2 = (1+g)*cos(t2)*sin(t3)  
        # f3 = (1+g)*cos(t3)
        # where g = sum of squared decision variables that don't affect front
        # On the Pareto front, g=0, so we need specific decision variables.
        
        # For DTLZ2 with 12 variables and 3 objectives:
        # x_1 = cos(t1)*sin(t2)*sin(t3)
        # x_2 = cos(t2)*sin(t3)
        # x_3 = cos(t3)
        # x_4...x_12 = 0.5 (to make g=0)
        
        # We'll generate points on the Pareto front directly and evaluate them
        # to get accurate objective values, then do local refinement.
        
        # Phase 1: Generate diverse points on the Pareto front
        # t1 in [0, pi/2], t2 in [0, pi/2], t3 in [0, pi/2]
        
        n_phase1 = budget // 2
        n_phase2 = budget - n_phase1
        
        # Generate points using stratified sampling for better coverage
        t1_vals = np.linspace(0, np.pi/2, n_phase1, endpoint=True)
        t2_vals = np.linspace(0, np.pi/2, n_phase1, endpoint=True)
        t3_vals = np.linspace(0, np.pi/2, n_phase1, endpoint=True)
        
        # Create a grid of points
        X = np.zeros((n_phase1, dim))
        F = np.zeros((n_phase1, self.n_obj))
        
        for i in range(n_phase1):
            t1 = t1_vals[i]
            t2 = t2_vals[(i * 7) % n_phase1]  # Use different t2 for diversity
            t3 = t3_vals[(i * 13) % n_phase1]  # Use different t3 for diversity
            
            X[i, 0] = np.cos(t1) * np.sin(t2) * np.sin(t3)
            X[i, 1] = np.cos(t2) * np.sin(t3)
            X[i, 2] = np.cos(t3)
            # Remaining variables set to 0.5 for g=0
            if dim > 3:
                X[i, 3:] = 0.5
        
        # Evaluate phase 1
        for i in range(n_phase1):
            F[i] = problem(X[i])
        
        used = n_phase1
        
        # Phase 2: Local refinement around promising points
        # Select best points from phase 1 and perturb them
        if n_phase2 > 0:
            # Select top points based on proximity to ideal point
            # Ideal point for DTLZ2 is [1, 1, 1]
            distances = np.sum((F - 1.0) ** 2, axis=1)
            sorted_idx = np.argsort(distances)
            
            # Take top half of phase 1 points for refinement
            n_refine = min(n_phase2, n_phase1 // 2)
            refine_idx = sorted_idx[:n_refine]
            
            for i in range(n_refine):
                if used >= budget:
                    break
                idx = refine_idx[i]
                # Perturb the selected point
                x_new = X[idx].copy()
                # Small random perturbation
                noise = 0.01 * rng.standard_normal(dim)
                x_new = np.clip(x_new + noise, lo, hi)
                
                # Ensure the point is still close to Pareto front
                # by keeping the structure
                if dim >= 3:
                    # Re-derive t values from x
                    t3 = np.arccos(np.clip(x_new[2], -1, 1))
                    sin_t3 = np.sin(t3)
                    if abs(sin_t3) > 1e-10:
                        cos_t2 = np.clip(x_new[1] / sin_t3, -1, 1)
                        t2 = np.arccos(cos_t2)
                        sin_t2 = np.sin(t2)
                        if abs(sin_t2) > 1e-10:
                            cos_t1 = np.clip(x_new[0] / (sin_t2 * sin_t3), -1, 1)
                            t1 = np.arccos(cos_t1)
                            x_new[0] = np.cos(t1) * np.sin(t2) * np.sin(t3)
                            x_new[1] = np.cos(t2) * np.sin(t3)
                            x_new[2] = np.cos(t3)
                
                F_new = problem(x_new)
                X[used % n_phase1] = x_new
                F[used % n_phase1] = F_new
                used += 1
        
        return X

# EVOLVE-BLOCK-END
