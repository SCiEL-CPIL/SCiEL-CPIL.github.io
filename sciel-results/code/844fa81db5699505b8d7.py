"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start coordinate descent with adaptive step-size and random dim reordering."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def remaining():
            return self.budget - (self._calls if hasattr(self, '_calls') else 0)

        self._calls = 0

        def count_call():
            self._calls += 1

        def safe_func(x):
            count_call()
            return func(np.asarray(x, dtype=float))

        # Phase 1: Multi-start coordinate descent
        # Use diverse starting points spread across the box
        n_starts = 12  # number of diverse initial points
        starts = []
        
        # Include origin, corners, and random points
        starts.append(np.zeros(self.dim))
        starts.append(np.full(self.dim, -5.0))
        starts.append(np.full(self.dim, 5.0))
        # Random points
        for _ in range(n_starts - 3):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Also add points based on f14 structure hints (different powers suggest 
        # the optimum might have components of varying magnitudes)
        starts.append(np.array([1.0, 2.0, 0.5, -1.5, 3.0]))
        starts.append(np.array([-2.0, 1.0, -3.0, 0.5, 1.5]))
        starts.append(rng.uniform(-4, 4, self.dim))
        starts.append(rng.uniform(-4, 4, self.dim))
        
        # Deduplicate-ish: just take first n_starts
        starts = starts[:n_starts]

        # For each start, do coordinate descent with adaptive step
        for si, x0 in enumerate(starts):
            if self._calls >= self.budget - 5 or time.time() >= deadline:
                break
            
            x = np.clip(np.array(x0, dtype=float), -5, 5)
            fx = safe_func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Adaptive step sizes per dimension
            steps = np.ones(self.dim) * 1.0
            
            # Coordinate descent iterations
            max_cd_iters = 80
            for it in range(max_cd_iters):
                if self._calls >= self.budget - 5 or time.time() >= deadline:
                    break
                
                # Random dimension reordering
                order = rng.permutation(self.dim)
                
                improved = False
                for d in order:
                    if self._calls >= self.budget - 5 or time.time() >= deadline:
                        break
                    
                    # Try moving in both directions along dimension d
                    x_plus = x.copy()
                    x_plus[d] = np.clip(x[d] + steps[d], -5, 5)
                    x_minus = x.copy()
                    x_minus[d] = np.clip(x[d] - steps[d], -5, 5)
                    
                    # Choose direction with better potential (evaluate both if possible)
                    # To save budget, try plus first, if it improves, keep going; else try minus
                    fp = safe_func(x_plus)
                    if fp < fx:
                        x = x_plus
                        fx = fp
                        steps[d] *= 1.3  # increase step
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        fm = safe_func(x_minus)
                        if fm < fx:
                            x = x_minus
                            fx = fm
                            steps[d] *= 1.3  # increase step
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        else:
                            steps[d] *= 0.6  # decrease step
                            # If step gets too small, try a random jump in this dim
                            if steps[d] < 0.01:
                                steps[d] = rng.uniform(0.5, 2.0)
                                x_new = x.copy()
                                x_new[d] = np.clip(x[d] + rng.uniform(-3, 3), -5, 5)
                                fn = safe_func(x_new)
                                if fn < fx:
                                    x = x_new
                                    fx = fn
                                    improved = True
                                    if fx < best_f:
                                        best_f = fx
                                        best_x = x.copy()
                                else:
                                    steps[d] = 0.5
                
                if not improved:
                    # No improvement in full cycle; try a random perturbation
                    if self._calls < self.budget - 10 and time.time() < deadline:
                        x_pert = x + rng.uniform(-1, 1, self.dim)
                        x_pert = np.clip(x_pert, -5, 5)
                        fp = safe_func(x_pert)
                        if fp < fx:
                            x = x_pert
                            fx = fp
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        # Reset some steps to explore
                        steps[rng.choice(self.dim, size=max(1, self.dim//2), replace=False)] = rng.uniform(0.5, 2.0, size=max(1, self.dim//2))
                    else:
                        break

        # Phase 2: Local refinement around best point using smaller steps
        if best_x is not None and self._calls < self.budget - 5 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            steps = np.ones(self.dim) * 0.3
            
            for it in range(50):
                if self._calls >= self.budget - 2 or time.time() >= deadline:
                    break
                
                order = rng.permutation(self.dim)
                improved = False
                for d in order:
                    if self._calls >= self.budget - 2 or time.time() >= deadline:
                        break
                    
                    x_plus = x.copy()
                    x_plus[d] = np.clip(x[d] + steps[d], -5, 5)
                    x_minus = x.copy()
                    x_minus[d] = np.clip(x[d] - steps[d], -5, 5)
                    
                    fp = safe_func(x_plus)
                    if fp < fx:
                        x = x_plus
                        fx = fp
                        steps[d] *= 1.1
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        fm = safe_func(x_minus)
                        if fm < fx:
                            x = x_minus
                            fx = fm
                            steps[d] *= 1.1
                            improved = True
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        else:
                            steps[d] *= 0.7
                
                if not improved:
                    break

        # Final: use remaining budget for fine-tuning if available
        if best_x is not None and self._calls < self.budget - 1 and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            for it in range(min(20, self.budget - self._calls - 1)):
                if self._calls >= self.budget - 1 or time.time() >= deadline:
                    break
                d = rng.integers(0, self.dim)
                step = 0.05 * (0.5 ** it)
                x_plus = x.copy()
                x_plus[d] = np.clip(x[d] + step, -5, 5)
                x_minus = x.copy()
                x_minus[d] = np.clip(x[d] - step, -5, 5)
                fp = safe_func(x_plus)
                if fp < fx:
                    x = x_plus
                    fx = fp
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    fm = safe_func(x_minus)
                    if fm < fx:
                        x = x_minus
                        fx = fm
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
