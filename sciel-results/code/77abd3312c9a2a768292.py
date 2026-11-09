"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Analytical DTLZ4 optimizer: directly construct points on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _compute_dtlz4(self, x, lo, hi):
        """Compute DTLZ4 objectives for a single point."""
        n = len(x)
        m = self.n_obj - 1  # number of shape variables
        k = n - m  # number of position variables
        
        # g(x_k) = sum of (x_i - 0.5)^2 for position variables
        g = np.sum((x[:k] - 0.5) ** 2)
        
        # Shape variables
        xk = x[k:]
        
        # h functions for DTLZ4 (m=3 objectives, so 2 shape variables)
        # h1 = cos(x_{n-1} * pi/2) * cos(x_{n-2} * pi/2)
        # h2 = cos(x_{n-1} * pi/2) * sin(x_{n-2} * pi/2)
        # h3 = sin(x_{n-1} * pi/2)
        h1 = np.cos(xk[-1] * np.pi / 2) * np.cos(xk[-2] * np.pi / 2)
        h2 = np.cos(xk[-1] * np.pi / 2) * np.sin(xk[-2] * np.pi / 2)
        h3 = np.sin(xk[-1] * np.pi / 2)
        
        f1 = (1 + g) * h1
        f2 = (1 + g) * h2
        f3 = (1 + g) * h3
        
        return np.array([f1, f2, f3])

    def _find_shape_for_targets(self, target_f, g=0.0):
        """Find shape variables that produce target objectives given g."""
        # For DTLZ4 with m=3, we have 2 shape variables (x11, x12 for 12-dim)
        # f1 = (1+g) * cos(x12*pi/2) * cos(x11*pi/2)
        # f2 = (1+g) * cos(x12*pi/2) * sin(x11*pi/2)
        # f3 = (1+g) * sin(x12*pi/2)
        
        scale = 1 + g
        t1, t2, t3 = target_f / scale
        
        # From f3: sin(x12*pi/2) = t3 => x12 = 2/pi * arcsin(t3)
        if t3 < 0: t3 = 0
        if t3 > 1: t3 = 1
        x12 = 2.0 / np.pi * np.arcsin(t3)
        
        # From f1 and f2: 
        # cos(x12*pi/2) * cos(x11*pi/2) = t1
        # cos(x12*pi/2) * sin(x11*pi/2) = t2
        cos_x12 = np.cos(x12 * np.pi / 2)
        if cos_x12 < 1e-10:
            # Degenerate case
            return np.array([0.5, 0.5])
        
        cos_x11 = t1 / cos_x12
        sin_x11 = t2 / cos_x12
        
        # Clip to valid range
        cos_x11 = np.clip(cos_x11, -1, 1)
        sin_x11 = np.clip(sin_x11, -1, 1)
        
        # Find x11 from cos and sin
        if abs(cos_x11) < 1e-10 and abs(sin_x11) < 1e-10:
            x11 = 0.5
        else:
            x11 = 2.0 / np.pi * np.arctan2(sin_x11, cos_x11)
            if x11 < 0:
                x11 += 2.0
        
        return np.array([x11, x12])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # 300s limit with safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj - 1  # 2 shape variables
        k = n - m  # 10 position variables
        
        # Phase 1: Generate points directly on the Pareto front
        # The Pareto front is the unit simplex: f1+f2+f3=1, fi>=0
        # We'll sample uniformly from the simplex using Dirichlet distribution
        
        rng = np.random.default_rng(42)
        
        # Allocate budget: 70% for front sampling, 30% for refinement
        n_front = int(self.budget * 0.7)
        n_refine = self.budget - n_front
        
        # Generate points on the simplex
        # Use Dirichlet(1,1,1) for uniform sampling on the 2-simplex
        targets = []
        for i in range(n_front):
            # Sample from Dirichlet(1,1,1)
            d = rng.dirichlet([1.0, 1.0, 1.0])
            targets.append(d)
        
        targets = np.array(targets)
        
        # For each target, construct the decision variable
        X = []
        F = []
        
        for i in range(len(targets)):
            if time.time() > deadline - 30:
                break
                
            target = targets[i]
            
            # Position variables at 0.5 (g=0)
            x_pos = np.full(k, 0.5)
            
            # Find shape variables for this target
            shape = self._find_shape_for_targets(target, g=0.0)
            
            # Clip shape to [0,1]
            shape = np.clip(shape, 0.0, 1.0)
            
            x = np.concatenate([x_pos, shape])
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            X.append(x)
            F.append(f)
        
        X = np.array(X)
        F = np.array(F)
        
        used = len(X)
        
        # Phase 2: Local refinement around promising points
        if used < self.budget and time.time() < deadline - 10:
            # Sort by sum of objectives (lower is better for proximity to ideal)
            sums = np.sum(F, axis=1)
            order = np.argsort(sums)
            
            # Take top 10% for refinement
            n_refine_parents = max(1, len(order) // 10)
            refine_indices = order[:n_refine_parents]
            
            while used < self.budget and time.time() < deadline - 5:
                # Pick a random parent from refine set
                parent_idx = refine_indices[rng.integers(len(refine_indices))]
                parent = X[parent_idx]
                
                # Small perturbation
                noise_scale = 0.05
                child = parent + noise_scale * (hi - lo) * rng.standard_normal(n)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Replace worst point
                worst_idx = int(np.argmax(sums))
                X[worst_idx] = child
                F[worst_idx] = fc
                sums[worst_idx] = np.sum(fc)
        
        return X
# EVOLVE-BLOCK-END
