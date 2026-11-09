"""Nelder-Mead Simplex Optimizer for Schaffers F7."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Nelder-Mead Parameters
        # Initial simplex size
        # Box is [-5, 5]^d. Center is 0.
        # We want the simplex to be large enough to cover potential gradients but small enough for precision.
        # Initial step size alpha = 0.5 is standard.
        alpha = 0.5
        
        # Create initial simplex
        # Vertex 0 is the origin
        simplex = [np.zeros(d)]
        
        # Other vertices are displaced along the axes
        for i in range(d):
            v = np.zeros(d)
            v[i] = alpha
            simplex.append(v)
        
        # Evaluate initial simplex
        f_simplex = [func(x) for x in simplex]
        calls = len(simplex)
        
        # Track best
        best_idx = np.argmin(f_simplex)
        best_x = simplex[best_idx].copy()
        best_f = f_simplex[best_idx]
        
        # Nelder-Mead constants
        gamma = 2.0  # Reflection
        rho = 2.0    # Expansion
        sigma = 0.5  # Contraction
        delta = 0.5  # Shrinkage
        
        # Main Loop
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Sort simplex by fitness
            # We maintain a list of (f, x) pairs
            # Simple sort for d+1 points is O(d log d)
            indexed = sorted(range(len(simplex)), key=lambda i: f_simplex[i])
            f_simplex = [f_simplex[i] for i in indexed]
            simplex = [simplex[i] for i in indexed]
            
            # Update best
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                best_x = simplex[0].copy()
                
            # Check for convergence
            # If the spread of function values is small, we might be converged
            # Or if the simplex is too small
            # However, for Schaffers F7, we might need to keep going if budget allows.
            # A simple check: if max(f) - min(f) < 1e-10, stop?
            # But AOCC rewards early progress. Stopping early doesn't help AOCC if we can keep improving.
            # Actually, if we stop, the last n_i is repeated.
            # If we are stuck, repeating the last n_i is fine.
            # But if we can still improve, we should.
            
            # Centroid of all points except the worst
            worst_idx = d
            centroid = np.mean(simplex[:d], axis=0)
            
            # Reflection
            x_ref = centroid + gamma * (centroid - simplex[worst_idx])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < f_simplex[0]:
                # Expansion
                x_exp = centroid + rho * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                
                if f_exp < f_ref:
                    # Accept expansion
                    simplex[worst_idx] = x_exp
                    f_simplex[worst_idx] = f_exp
                else:
                    # Accept reflection
                    simplex[worst_idx] = x_ref
                    f_simplex[worst_idx] = f_ref
            elif f_ref < f_simplex[d-1]:
                # Accept reflection
                simplex[worst_idx] = x_ref
                f_simplex[worst_idx] = f_ref
            else:
                # Contraction
                # Outside contraction
                x_con = centroid + 0.5 * (x_ref - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                
                if f_con < f_ref:
                    # Accept outside contraction
                    simplex[worst_idx] = x_con
                    f_simplex[worst_idx] = f_con
                else:
                    # Inside contraction
                    x_con_in = centroid - 0.5 * (centroid - simplex[worst_idx])
                    x_con_in = np.clip(x_con_in, -5, 5)
                    f_con_in = func(x_con_in)
                    calls += 1
                    
                    if f_con_in < f_simplex[worst_idx]:
                        # Accept inside contraction
                        simplex[worst_idx] = x_con_in
                        f_simplex[worst_idx] = f_con_in
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + delta * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                        if calls >= budget:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
