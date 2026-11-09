"""Deterministic Nelder-Mead Simplex Search for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin for 300s limit
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Initialize at the center of the box
        # Nelder-Mead is a local method, so starting at the center is a robust
        # deterministic choice that avoids random initialization risks.
        x0 = np.zeros(self.dim)
        
        # Initial simplex setup
        # Standard Nelder-Mead requires dim+1 points.
        # We will use a small initial step size for the simplex vertices.
        # A step size of 0.05 is small enough to stay local but large enough
        # to make progress.
        alpha = 0.05 
        
        # Create initial simplex
        # v[0] = x0
        # v[i] = x0 + alpha * e_i for i=1..dim
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x0
        for i in range(1, self.dim + 1):
            simplex[i] = x0
            simplex[i, i-1] += alpha
            
        # Evaluate initial simplex
        fvals = np.zeros(self.dim + 1)
        for i in range(self.dim + 1):
            if time.time() > deadline:
                break
            fx = func(simplex[i])
            fvals[i] = fx
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = simplex[i].copy()
                
        # Nelder-Mead parameters
        # Standard values: alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5
        alpha_nm = 1.0
        gamma_nm = 2.0
        rho_nm = 0.5
        sigma_nm = 0.5
        
        # Main loop
        # We track the number of function calls.
        # Each iteration of Nelder-Mead typically uses 1 function call (for reflection, expansion, contraction, or shrink).
        
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # The worst point is the last one
            x_worst = simplex[-1]
            f_worst = fvals[-1]
            
            # Centroid of the best dim points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha_nm * (centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma_nm * (centroid - x_worst)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept Reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                # Check if contraction is needed.
                # We try an outside contraction first if f_worst < f_r, else inside.
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - 0.5 * (x_worst - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    # Shrink all points except the best towards the best point
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        if time.time() > deadline:
                            break
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                        
                        if calls >= self.budget:
                            break
                    
                    if calls >= self.budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
