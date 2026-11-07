"""Derivative-free adaptive lattice search for f22 (Gallagher 21 peaks)."""
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Adaptive lattice search with Nelder-Mead polishing for 21-peak f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def _call(self, func, x):
        """Count a function call."""
        self.calls += 1
        return float(func(x))

    def _nelder_mead(self, func, x0, budget, rng):
        """Nelder-Mead local optimization with budget constraint."""
        n = self.dim
        # Initial simplex
        simplex = [x0.copy()]
        f0 = self._call(func, x0)
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.5
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
        
        fvals = [f0]
        for i in range(1, n + 1):
            fi = self._call(func, simplex[i])
            fvals.append(fi)
        
        best_idx = np.argmin(fvals)
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]
        
        # Nelder-Mead iterations
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage
        
        while self.calls < budget:
            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            best_idx = 0
            best_x = simplex[best_idx].copy()
            best_f = fvals[best_idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = self._call(func, xr)
            
            if fr < fvals[-2]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = self._call(func, xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-1]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = self._call(func, xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrinkage
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = self._call(func, simplex[i])
                    
                    # Check for convergence
                    if np.max(fvals) - np.min(fvals) < 1e-10:
                        break
        
        return best_x, best_f

    def __call__(self, func):
        deadline = time.time() + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        
        # Phase 1: Adaptive lattice search
        # Start with a coarse grid and refine promising regions
        n_grid = 4  # points per dimension initially
        grid_spacing = 8.0 / (n_grid - 1)  # range is [-4, 4], so 8 units
        grid_min = -4.0
        grid_max = 4.0
        
        # Generate initial grid
        grid_points = []
        for i in range(n_grid):
            for j in range(n_grid):
                for k in range(n_grid):
                    for l in range(n_grid):
                        for m in range(n_grid):
                            x = np.array([
                                grid_min + i * grid_spacing,
                                grid_min + j * grid_spacing,
                                grid_min + k * grid_spacing,
                                grid_min + l * grid_spacing,
                                grid_min + m * grid_spacing
                            ])
                            grid_points.append(x)
        
        # Evaluate initial grid
        grid_vals = []
        for x in grid_points:
            if self.calls >= self.budget or time.time() > deadline:
                break
            fx = self._call(func, x)
            grid_vals.append((fx, x))
        
        if not grid_vals:
            # Fallback: random evaluation
            x = rng.uniform(-4, 4, dim)
            fx = self._call(func, x)
            return x, fx
        
        grid_vals.sort()
        best_x = grid_vals[0][1].copy()
        best_f = grid_vals[0][0]
        
        # Phase 2: Iterative refinement of best region
        # Each iteration: pick best point, create finer local grid, evaluate
        n_iter = 5  # number of refinement iterations
        for it in range(n_iter):
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            # Focus on neighborhood of best point
            center = best_x
            radius = 1.0 / (it + 1)  # shrinking radius
            
            # Generate local grid around center
            local_n = 3  # points per dimension
            local_spacing = 2 * radius / (local_n - 1) if local_n > 1 else radius
            local_points = []
            for i in range(local_n):
                for j in range(local_n):
                    for k in range(local_n):
                        for l in range(local_n):
                            for m in range(local_n):
                                x = np.array([
                                    center[0] + (i - (local_n - 1) / 2) * local_spacing,
                                    center[1] + (j - (local_n - 1) / 2) * local_spacing,
                                    center[2] + (k - (local_n - 1) / 2) * local_spacing,
                                    center[3] + (l - (local_n - 1) / 2) * local_spacing,
                                    center[4] + (m - (local_n - 1) / 2) * local_spacing
                                ])
                                x = np.clip(x, -5, 5)
                                local_points.append(x)
            
            # Evaluate local grid
            local_found = False
            for x in local_points:
                if self.calls >= self.budget or time.time() > deadline:
                    break
                fx = self._call(func, x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    local_found = True
            
            # Phase 3: Nelder-Mead polishing every 50 evaluations
            if self.calls % 50 < 10 and self.budget - self.calls > 30:
                nm_budget = min(30, self.budget - self.calls)
                nm_x, nm_f = self._nelder_mead(func, best_x, self.calls + nm_budget, rng)
                if nm_f < best_f:
                    best_f = nm_f
                    best_x = nm_x.copy()
        
        self.best_x = best_x
        self.best_f = best_f
        return best_x, best_f
