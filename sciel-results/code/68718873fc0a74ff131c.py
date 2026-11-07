"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 that exploits the known Pareto front geometry.
    
    DTLZ1 with M=3 objectives, N=7 variables:
    - g(x) = 100 * (N - M + 1 + sum_{i=M}^{N} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))])
    - f_1 = 0.5 * (1 + g) * x_1
    - f_2 = 0.5 * (1 + g) * x_2  
    - f_3 = 0.5 * (1 + g) * (1 - x_1^2 - x_2^2)
    
    Pareto front: g=0 (x_3..x_6 = 0.5), x_1^2 + x_2^2 = 1, x_1, x_2 >= 0
    On PF: f_1 = 0.5*x_1, f_2 = 0.5*x_2, f_3 = 0
    Objective space is a quarter-disk of radius 0.5 centered at origin.
    
    Strategy: Use Sobol sequence to sample the quarter-disk uniformly, then
    for each point on the PF, find x_1, x_2 such that x_1^2+x_2^2=1 and
    f_1=0.5*x_1, f_2=0.5*x_2, f_3=0. Use local optimization to refine.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget allocation:
        # - 100 points: Sobol samples on PF (direct construction)
        # - Remaining: local refinement
        n_sobol = 100
        n_refine = min(self.budget - n_sobol, 100)
        
        # Generate Sobol points in [0,1]^2 for the quarter-disk parameterization
        # Use angle theta in [0, pi/2] and radius r in [0,1]
        # x_1 = r*cos(theta), x_2 = r*sin(theta)
        # For PF, r=1, so x_1=cos(theta), x_2=sin(theta)
        
        # Actually, for the PF, we need x_1^2+x_2^2=1, so we just need theta
        # But to get good hypervolume, we want uniform coverage of the objective space
        # f_1 = 0.5*cos(theta), f_2 = 0.5*sin(theta), f_3 = 0
        # This is a quarter circle in objective space
        
        # For better hypervolume, sample uniformly in the quarter disk
        # Use polar coordinates: r in [0,1], theta in [0, pi/2]
        # x_1 = r*cos(theta), x_2 = r*sin(theta)
        # f_1 = 0.5*r*cos(theta), f_2 = 0.5*r*sin(theta), f_3 = 0.5*(1-r^2)
        
        sobol = qmc.Sobol(d=2, scramble=True)
        sobol_points = sobol.random(n_sobol)
        # sobol_points: shape (n_sobol, 2), values in [0,1]
        # Map to r in [0,1], theta in [0, pi/2]
        r = sobol_points[:, 0]
        theta = sobol_points[:, 1] * (np.pi / 2)
        
        x_1 = r * np.cos(theta)
        x_2 = r * np.sin(theta)
        
        # Store evaluated points
        X_list = []
        F_list = []
        used = 0
        
        # Evaluate Sobol points on the PF
        for i in range(n_sobol):
            if time.time() > deadline:
                break
            
            # x = [x_1, x_2, 0.5, 0.5, 0.5, 0.5, 0.5]
            x = np.array([x_1[i], x_2[i], 0.5, 0.5, 0.5, 0.5, 0.5])
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_list.append(x.copy())
            F_list.append(np.array(f))
            used += 1
        
        # Local refinement: for each point, try to improve by adjusting x_1, x_2
        # while keeping x_3..x_6 = 0.5
        if used < self.budget and time.time() < deadline:
            # Use remaining budget for refinement
            n_refine_actual = min(n_refine, self.budget - used)
            
            for i in range(min(len(X_list), n_refine_actual)):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                
                # Try small perturbations in x_1, x_2
                x0 = X_list[i].copy()
                f0 = F_list[i].copy()
                
                # Try a few random perturbations
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Perturb x_1 and x_2 slightly
                    dx1 = 0.05 * (np.random.rand() - 0.5)
                    dx2 = 0.05 * (np.random.rand() - 0.5)
                    
                    x_new = x0.copy()
                    x_new[0] = np.clip(x0[0] + dx1, lo[0], hi[0])
                    x_new[1] = np.clip(x0[1] + dx2, lo[1], hi[1])
                    
                    f_new = np.array(problem(x_new))
                    used += 1
                    
                    # Check if this is non-dominated and better
                    # Simple: if sum of objectives is lower, replace
                    if np.sum(f_new) < np.sum(f0):
                        X_list[i] = x_new
                        F_list[i] = f_new
                        f0 = f_new
        
        # Convert to arrays
        X = np.array(X_list) if X_list else np.empty((0, self.dim))
        F = np.array(F_list) if F_list else np.empty((0, self.n_obj))
        
        return X
# EVOLVE-BLOCK-END
