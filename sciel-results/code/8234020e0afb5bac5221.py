"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Nelder-Mead with restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 300s limit
        
        # We will use a multi-start strategy if budget allows, 
        # but primarily rely on a robust Nelder-Mead.
        # For dim=5, a single well-tuned NM run is often best.
        # However, to be safe against bad initialization, we can do 2 runs 
        # if the first one doesn't converge well.
        
        # Let's stick to one robust run to maximize depth, 
        # but initialize the simplex carefully.
        
        rng = np.random.default_rng(42)
        
        # Initial point: center of the box or random?
        # The minimizer is in [-4, 4]^5. 
        # Starting at 0 is a good bet.
        x0 = np.zeros(self.dim)
        
        # Initial simplex
        # Standard NM simplex: x0, and x0 + h * e_i
        # h should be large enough to see the slope, small enough to stay in bounds.
        # Box is [-5, 5].
        h = 1.0
        
        simplex = [x0.copy()]
        for i in range(self.dim):
            point = x0.copy()
            point[i] += h
            simplex.append(point)
        
        # Evaluate initial simplex
        fvals = []
        calls = 0
        for p in simplex:
            fvals.append(func(p))
            calls += 1
            
        best_x = simplex[np.argmin(fvals)]
        best_f = min(fvals)
        
        # Nelder-Mead Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Convergence tolerance
        tol = 1e-10
        
        while calls < self.budget:
            if time.time() - start_time > time_limit:
                break
                
            # Order simplex
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < tol:
                break
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            
            if fr < fvals[0]:
                # Expand
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contract
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= self.budget:
                            break
                    if calls >= self.budget:
                        break
            
            # Update best
            current_best_idx = np.argmin(fvals)
            if fvals[current_best_idx] < best_f:
                best_f = fvals[current_best_idx]
                best_x = simplex[current_best_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
