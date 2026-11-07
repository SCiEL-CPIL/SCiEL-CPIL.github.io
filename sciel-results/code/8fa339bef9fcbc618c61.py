"""Multi-Start Nelder-Mead Optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 330.0
_SAFETY_MARGIN = 15.0  # seconds before hard kill


class Optimizer:
    """Multi-start Nelder-Mead optimizer with explicit deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _elapsed(self):
        return time.time() - self._start_time

    def _within_budget(self):
        return self._elapsed() < (_DEADLINE - _SAFETY_MARGIN)

    def _call_and_record(self, func, x):
        """Call func, record in self._calls list, return value."""
        self._calls.append((x.copy(), func(x)))
        return self._calls[-1][1]

    def _best_so_far(self):
        if not self._calls:
            return None, None
        best_x = None
        best_f = float('inf')
        for x, f in self._calls:
            if f < best_f:
                best_f = f
                best_x = x
        return best_x, best_f

    def _nelder_mead(self, func, rng, budget, start_x):
        """
        Perform Nelder-Mead optimization from a given start point.
        Returns best_x, best_f found during this run.
        """
        if not self._within_budget():
            return start_x, self._call_and_record(func, start_x)

        # Initial simplex construction
        # Start with start_x and dim perturbed points
        simplex = [start_x.copy()]
        base_step = 0.5
        for i in range(self.dim):
            x_pert = start_x.copy()
            x_pert[i] += base_step
            x_pert = np.clip(x_pert, -5, 5)
            simplex.append(x_pert)
        
        # Evaluate initial simplex
        simplex_f = []
        for x in simplex:
            if not self._within_budget():
                break
            simplex_f.append(self._call_and_record(func, x))
        
        if len(simplex_f) < self.dim + 1:
            # Not enough budget for initial simplex
            best_idx = np.argmin(simplex_f) if simplex_f else 0
            if simplex_f:
                return simplex[best_idx], simplex_f[best_idx]
            return start_x, self._call_and_record(func, start_x)

        best_x = simplex[0].copy()
        best_f = min(simplex_f)
        
        # Sort simplex by function value
        order = np.argsort(simplex_f)
        simplex = [simplex[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]

        max_iter = budget
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        for _ in range(max_iter):
            if not self._within_budget():
                break
            
            # Sort again to ensure ordering
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst_point = simplex[-1]
            worst_f = simplex_f[-1]

            # Reflection
            x_r = centroid + alpha * (centroid - worst_point)
            x_r = np.clip(x_r, -5, 5)
            f_r = self._call_and_record(func, x_r)

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = self._call_and_record(func, x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Simple Reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < worst_f:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = self._call_and_record(func, x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = self._call_and_record(func, simplex[i])
                else:
                    # Inside contraction
                    x_c = centroid - rho * (worst_point - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = self._call_and_record(func, x_c)
                    if f_c < worst_f:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = self._call_and_record(func, simplex[i])
            
            # Check for convergence (optional, but helps stop early if stuck)
            if len(set([f for f in simplex_f])) == 1:
                break

        return best_x, best_f

    def __call__(self, func):
        self._start_time = time.time()
        self._calls = []
        rng = np.random.default_rng(42)

        # Determine number of restarts based on budget
        # Each NM run consumes at least dim+1 calls for initialization + iterations
        # Let's allocate budget per restart
        n_restarts = 5
        budget_per_restart = self.budget // n_restarts
        
        global_best_x = None
        global_best_f = float('inf')

        for i in range(n_restarts):
            if not self._within_budget():
                break
            
            # Choose start point
            if i == 0:
                # First start: center or random
                start_x = np.zeros(self.dim)
            else:
                # Subsequent starts: random in box
                start_x = rng.uniform(-5, 5, self.dim)
            
            # Run Nelder-Mead
            best_x, best_f = self._nelder_mead(func, rng, budget_per_restart, start_x)
            
            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x

        # If we have budget left, do a final local refinement from the global best
        remaining_calls = self.budget - len(self._calls)
        if remaining_calls > 10 and global_best_x is not None and self._within_budget():
            # Use a small NM or simple gradient-free local search
            # Let's use a very fine-grained NM or just a few random perturbations
            # Given the directive to use NM, let's do a short NM run with small initial step
            # But NM needs dim+1 points. If remaining is small, skip.
            if remaining_calls >= self.dim + 1:
                # Create a tiny simplex around the best point
                start_x = global_best_x.copy()
                base_step = 0.1
                simplex = [start_x.copy()]
                for i in range(self.dim):
                    x_pert = start_x.copy()
                    x_pert[i] += base_step
                    x_pert = np.clip(x_pert, -5, 5)
                    simplex.append(x_pert)
                
                # Evaluate
                simplex_f = []
                for x in simplex:
                    if not self._within_budget():
                        break
                    simplex_f.append(self._call_and_record(func, x))
                
                if len(simplex_f) == self.dim + 1:
                    # Run a few NM steps
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    for _ in range(remaining_calls - (self.dim + 1)):
                        if not self._within_budget():
                            break
                        centroid = np.mean(simplex[:-1], axis=0)
                        worst_point = simplex[-1]
                        x_r = centroid + 1.0 * (centroid - worst_point)
                        x_r = np.clip(x_r, -5, 5)
                        f_r = self._call_and_record(func, x_r)
                        
                        if f_r < simplex_f[0]:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                        elif f_r < simplex_f[-2]:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                        else:
                            # Shrink towards best
                            for j in range(1, len(simplex)):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                simplex[j] = np.clip(simplex[j], -5, 5)
                                simplex_f[j] = self._call_and_record(func, simplex[j])
                            
                            order = np.argsort(simplex_f)
                            simplex = [simplex[i] for i in order]
                            simplex_f = [simplex_f[i] for i in order]
                            
                    # Update global best
                    if simplex_f[0] < global_best_f:
                        global_best_f = simplex_f[0]
                        global_best_x = simplex[0].copy()

        # Fill remaining budget with random calls if needed to avoid "too few calls" issues?
        # The task says "repeat the last n_i if fewer calls are made", so we don't strictly need to hit 500.
        # However, hitting the budget might be safer for AOCC if early calls are bad.
        # But random calls usually hurt AOCC if they are worse than current best.
        # So we stop here.

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
