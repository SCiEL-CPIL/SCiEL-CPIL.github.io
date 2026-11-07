"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead optimizer with random initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget

        # Counting wrapper
        def wrapped(x):
            if self._call_count >= budget:
                return 1e300
            self._call_count += 1
            try:
                return float(func(x))
            except Exception:
                return 1e300

        rng = np.random.default_rng(42)
        
        # Phase 1: Random search to find a good starting point
        # Use 50 calls to sample random points
        num_initial = min(50, budget)
        best_x = None
        best_f = float('inf')
        
        for _ in range(num_initial):
            if self._call_count >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            f = wrapped(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        # Phase 2: Nelder-Mead
        # Initialize simplex around best_x
        # Simplex has dim+1 points
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_x
        simplex_f = np.zeros(dim + 1)
        simplex_f[0] = best_f
        
        # Initialize other points with small perturbations
        for i in range(1, dim + 1):
            if self._call_count >= budget:
                break
            # Perturb along i-th dimension
            x_i = best_x.copy()
            x_i[i-1] += 0.5 # Step size
            x_i = np.clip(x_i, -5.0, 5.0)
            simplex[i] = x_i
            simplex_f[i] = wrapped(x_i)
            
            # Update best if this new point is better
            if simplex_f[i] < best_f:
                best_f = simplex_f[i]
                best_x = simplex[i].copy()

        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma_nm = 0.5 # Shrinkage
        
        while self._call_count < budget:
            # Sort simplex by fitness
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Worst point
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            
            # Reflection
            x_reflect = centroid + alpha * (centroid - x_worst)
            x_reflect = np.clip(x_reflect, -5.0, 5.0)
            f_reflect = wrapped(x_reflect)
            
            if self._call_count >= budget:
                break
                
            if f_reflect < simplex_f[0]:
                # Try Expansion
                x_expand = centroid + gamma * (x_reflect - centroid)
                x_expand = np.clip(x_expand, -5.0, 5.0)
                f_expand = wrapped(x_expand)
                
                if self._call_count >= budget:
                    break
                    
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    simplex_f[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
            elif f_reflect < simplex_f[-2]:
                # Accept Reflection
                simplex[-1] = x_reflect
                simplex_f[-1] = f_reflect
            else:
                # Contraction
                # Check if we have budget for contraction
                if self._call_count >= budget:
                    break
                    
                x_contract = centroid + rho * (x_worst - centroid)
                x_contract = np.clip(x_contract, -5.0, 5.0)
                f_contract = wrapped(x_contract)
                
                if f_contract < f_worst:
                    simplex[-1] = x_contract
                    simplex_f[-1] = f_contract
                else:
                    # Shrink
                    # We need to shrink all points except the best
                    # This requires dim evaluations
                    if self._call_count + dim > budget:
                        # Not enough budget for full shrink, just stop or do partial
                        break
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        simplex_f[i] = wrapped(simplex[i])
                        
                        if self._call_count >= budget:
                            break
                            
            # Update global best
            idx_min = np.argmin(simplex_f)
            if simplex_f[idx_min] < best_f:
                best_f = simplex_f[idx_min]
                best_x = simplex[idx_min].copy()

        return None
# EVOLVE-BLOCK-END
