"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Adaptive Simulated Annealing (ASA) for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Broad exploration with multiple restarts
        # Use 8 restarts with better spread for rotated landscapes
        n_restarts = 8
        # Budget allocation: decreasing for exploration, increasing for refinement
        # First few restarts explore widely, later ones refine
        fractions = [0.12, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.16]
        budget_alloc = [int(budget * f) for f in fractions]
        diff = budget - sum(budget_alloc)
        budget_alloc[-1] += diff

        for restart in range(n_restarts):
            b = budget_alloc[restart]
            if b < 5 or calls_used >= budget:
                break

            # Initialize starting point with diverse strategies
            if restart == 0:
                x0 = np.zeros(dim)
            elif restart == 1:
                x0 = np.array([2.0, -2.0, 1.5, -1.5, 0.5])
            elif restart == 2:
                x0 = np.array([-2.5, 2.5, -1.0, 1.0, -2.0])
            elif restart == 3:
                x0 = np.array([3.0, -3.0, 2.0, -2.0, 1.0])
            elif restart == 4:
                x0 = rng.uniform(-3.5, 3.5, dim)
            elif restart == 5:
                x0 = rng.uniform(-4.0, 4.0, dim)
            elif restart == 6:
                x0 = rng.uniform(-4.5, 4.5, dim)
            else:
                x0 = rng.uniform(-5.0, 5.0, dim)

            x = np.clip(x0, -5, 5)
            fx = func(x)
            calls_used += 1
            best_local_x = x.copy()
            best_local_f = fx

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # ASA with adaptive step scaling
            # Start with larger steps for exploration, cool down
            T0 = 3.0 if restart < 4 else 1.5
            T_end = 0.003
            alpha = (T_end / T0) ** (1.0 / max(b - 1, 1))
            T = T0

            # Track recent improvements to adapt temperature
            recent_improvements = []

            for t in range(b - 1):
                if calls_used >= budget:
                    break

                # Classic ASA direction: u * |v| where u is random unit vector
                u = rng.standard_normal(dim)
                norm_u = np.linalg.norm(u)
                if norm_u > 1e-10:
                    u = u / norm_u
                v = abs(rng.standard_normal(dim))
                perturbation = u * v
                x_new = np.clip(x + T * perturbation, -5, 5)

                fx_new = func(x_new)
                calls_used += 1

                # Metropolis acceptance with adaptive temperature
                if fx_new < fx:
                    improvement = fx - fx_new
                    x = x_new
                    fx = fx_new
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local_x = x.copy()
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    recent_improvements.append(improvement)
                    # If improving, keep temperature or increase slightly
                    if len(recent_improvements) >= 5:
                        avg_imp = np.mean(recent_improvements[-5:])
                        if avg_imp > T * 0.1:
                            T = min(T * 1.05, T0)
                else:
                    delta_f = fx_new - fx
                    if delta_f / T < 50:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                            if fx < best_local_f:
                                best_local_f = fx
                                best_local_x = x.copy()
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                            recent_improvements.append(0.0)
                    else:
                        recent_improvements.append(0.0)

                # Cool temperature
                T *= alpha
                T = max(T, T_end * 0.5)

                # If no improvement in last 20 steps, boost temperature
                if len(recent_improvements) >= 20:
                    recent_non_imp = recent_improvements[-20:]
                    if np.sum(recent_non_imp) < 1e-6:
                        T = min(T * 2.0, T0)

        # Phase 2: Local refinement around best solution
        remaining = budget - calls_used
        if remaining > 3 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            # Start refinement with moderate step
            T = 0.08
            T_end = 0.001
            alpha = (T_end / T) ** (1.0 / max(remaining - 1, 1))
            
            # Use coordinate-wise perturbations mixed with random directions
            for t in range(remaining - 1):
                if calls_used >= budget:
                    break
                
                # Mix of random direction and coordinate perturbations
                if rng.random() < 0.3:
                    # Coordinate-wise perturbation
                    coord = rng.integers(0, dim)
                    perturbation = np.zeros(dim)
                    perturbation[coord] = rng.standard_normal()
                    step_size = T
                else:
                    # Full random direction
                    u = rng.standard_normal(dim)
                    norm_u = np.linalg.norm(u)
                    if norm_u > 1e-10:
                        u = u / norm_u
                    v = abs(rng.standard_normal(dim))
                    perturbation = u * v
                    step_size = T

                x_new = np.clip(x + step_size * perturbation, -5, 5)
                fx_new = func(x_new)
                calls_used += 1
                
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Keep temperature when improving
                else:
                    delta_f = fx_new - fx
                    if delta_f / T < 50:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

                T *= alpha
                T = max(T, T_end * 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
