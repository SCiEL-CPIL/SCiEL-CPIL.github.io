"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Sparse lattice global search + adaptive Nelder-Mead with momentum."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        # Phase 1: Sparse global lattice to identify basin of attraction
        # Use a stratified grid on [-4, 4]^dim (minimizer lies in [-4,4]^5)
        # For dim=5, use 3 levels per axis => 3^5 = 243 points, but that's too many
        # Use a sparse lattice: 2 levels per axis => 2^5 = 32 points
        # Plus some intermediate points for better coverage
        lattice_levels = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        
        # Generate all combinations from the lattice (5^5 = 3125 is too many)
        # Instead, use a random subset of lattice points
        n_lattice = min(40, budget // 10)
        
        # Create lattice points by sampling from the 5-level grid
        lattice_points = []
        for _ in range(n_lattice):
            pt = rng.choice(lattice_levels, size=dim)
            lattice_points.append(pt)
        lattice_points = np.array(lattice_points)
        
        # Add a few random points in [-4, 4]^dim
        n_random = min(20, budget // 20)
        random_points = rng.uniform(-4, 4, size=(n_random, dim))
        
        all_global = np.vstack([lattice_points, random_points])
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for i in range(len(all_global)):
            if calls_made >= budget:
                break
            x = np.clip(all_global[i], -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Adaptive Nelder-Mead with momentum
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Adaptive step size based on initial basin width
        current_step = 0.8
        max_restarts = 8
        restart_count = 0
        
        # Momentum for exploiting smoothness
        momentum = np.zeros(dim)
        momentum_alpha = 0.3
        
        while calls_made < budget and restart_count <= max_restarts:
            # Initialize simplex with adaptive step size
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x
            for i in range(1, dim + 1):
                simplex[i] = best_x.copy()
                simplex[i, i - 1] += current_step
                simplex[i] = np.clip(simplex[i], -5, 5)
                
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    simplex_f[i] = np.inf
                else:
                    simplex_f[i] = func(simplex[i])
                    calls_made += 1
                    if simplex_f[i] < best_f:
                        best_f = simplex_f[i]
                        best_x = simplex[i].copy()
                        
            no_improve = 0
            max_no_improve = 12
            consecutive_shrinks = 0
            
            while calls_made < budget:
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                spread = np.max(simplex_f) - np.min(simplex_f)
                if spread < 1e-10:
                    break
                    
                centroid = simplex[:-1].mean(axis=0)
                
                # Apply momentum to centroid for smooth exploitation
                momentum = momentum_alpha * (centroid - simplex[-1]) + (1 - momentum_alpha) * momentum
                centroid_momentum = centroid + 0.5 * momentum
                
                if calls_made >= budget: break
                x_r = centroid_momentum + alpha * (centroid_momentum - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                if f_r < simplex_f[0]:
                    # Strong improvement: try expansion
                    if calls_made >= budget: break
                    x_e = centroid_momentum + gamma * (centroid_momentum - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    no_improve = 0
                    consecutive_shrinks = 0
                    # Increase step size slightly on success
                    current_step = min(1.5, current_step * 1.1)
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                    no_improve = 0
                    consecutive_shrinks = 0
                else:
                    # Contraction
                    if calls_made >= budget: break
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                        no_improve = 0
                    else:
                        no_improve += 1
                        consecutive_shrinks += 1
                        # Shrink simplex toward best point
                        for i in range(1, dim + 1):
                            if calls_made >= budget: break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            calls_made += 1
                            if simplex_f[i] < best_f:
                                best_f = simplex_f[i]
                                best_x = simplex[i].copy()
                        
                        # Reduce step size on failure
                        current_step = max(0.05, current_step * 0.7)
                        
                if no_improve > max_no_improve or consecutive_shrinks > 3:
                    # Restart with perturbed center
                    restart_count += 1
                    current_step = max(0.1, current_step * 0.5)
                    best_x = best_x + rng.normal(0, 0.15, size=dim)
                    best_x = np.clip(best_x, -5, 5)
                    momentum = np.zeros(dim)
                    break
            
        return best_x, best_f
# EVOLVE-BLOCK-END
