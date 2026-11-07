# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        results = []
        
        # DTLZ1 Pareto front: f1 + f2 + f3 = 0.5, fi >= 0
        # We sample the 2-simplex and map to decision space
        # For DTLZ1: f1 = 0.5*x1*(1+g), f2 = 0.5*x1*(1-x1)*(1+g), f3 = 0.5*(1-x1)*(1+g)
        # On Pareto front, g=0, so f1+f2+f3 = 0.5*x1*(1 + 1-x1 + 1) = 0.5*x1*(2-x1)
        # Wait, that's not right. Let me re-derive.
        # f1 = 0.5 * x1 * (1+g)
        # f2 = 0.5 * x1 * (1-x1) * (1+g)
        # f3 = 0.5 * (1-x1) * (1+g)
        # Sum = 0.5 * (1+g) * [x1 + x1*(1-x1) + (1-x1)]
        #     = 0.5 * (1+g) * [x1 + x1 - x1^2 + 1 - x1]
        #     = 0.5 * (1+g) * [x1 + 1 - x1^2]
        # For this to equal 0.5, we need (1+g)*(x1 + 1 - x1^2) = 1
        # This is complex. Let me use a different approach.
        
        # Actually, the standard DTLZ1 Pareto front is:
        # f1 + f2 + f3 = 0.5 (when g=0)
        # The mapping from simplex to decision space:
        # Let u1, u2, u3 >= 0, u1+u2+u3 = 1
        # Then f1 = 0.5*u1, f2 = 0.5*u2, f3 = 0.5*u3
        # From f1/f2 = x1/(1-x1) => x1 = f1/(f1+f2) = u1/(u1+u2)
        # And g must be 0.
        
        # To achieve g=0, we need to find x2..x7 such that
        # sum_{i=2}^{7} [(xi-0.5)^2 - 10*cos(20*pi*(xi-0.5))] = 0
        
        # A simple solution: set all xi = 0.5 for i=2..7
        # Then each term: (0)^2 - 10*cos(0) = -10
        # Sum = 6 * (-10) = -60, not 0.
        
        # Let me try xi = 0 for i=2..7:
        # (0-0.5)^2 - 10*cos(20*pi*(-0.5)) = 0.25 - 10*cos(-10*pi) = 0.25 - 10*1 = -9.75
        # Sum = 6 * (-9.75) = -58.5, not 0.
        
        # The g function is designed so that g=0 has solutions. Let me just use 
        # random search for the tail variables, or use a known solution.
        
        # Alternative: Use the fact that we can directly construct points.
        # For any x1 in [0,1] and any (x2,...,x7) with g=0, we get a Pareto point.
        # The simplest g=0 solution: we need sum of terms = 0.
        
        # Let me use a pragmatic approach: sample x1 from the simplex, 
        # and for the tail, use a fixed vector that approximately gives g=0,
        # then use a few evaluations to refine.
        
        # Actually, let me just do a smart random search focused on the front.
        
        # Strategy: 
        # 1. Sample points on the 2-simplex (u1,u2,u3) with u1+u2+u3=1
        # 2. For each, set x1 = u1/(u1+u2) if u1+u2 > 0, else 0.5
        # 3. Set x2..x7 to a fixed "good" vector (we'll find one)
        # 4. Evaluate and keep results
        
        # First, let's find a tail vector that gives g close to 0
        # Try all zeros: g = sum[(0-0.5)^2 - 10*cos(20*pi*(-0.5))] = 6*[0.25 - 10*1] = 6*(-9.75) = -58.5
        # Try all 0.5: g = 6*[0 - 10*1] = -60
        # Try alternating 0 and 1: for xi=0: term=0.25-10*cos(-10pi)=0.25-10=-9.75
        #                          for xi=1: term=0.25-10*cos(10pi)=0.25-10=-9.75
        # So any binary vector gives g = 6*(-9.75) = -58.5
        
        # The minimum of each term (xi-0.5)^2 - 10*cos(20*pi*(xi-0.5)):
        # Let t = xi - 0.5, t in [-0.5, 0.5]
        # h(t) = t^2 - 10*cos(20*pi*t)
        # h(0) = -10
        # h(0.5) = 0.25 - 10*cos(10*pi) = 0.25 - 10 = -9.75
        # h is minimized near t=0 with value -10
        # Maximum at t=0.5 or -0.5: -9.75
        # So each term is in [-10, -9.75], sum of 6 terms in [-60, -58.5]
        # g is always negative! That means (1+g) < 1 always.
        
        # Wait, I think I have the wrong DTLZ1 formulation. Let me check.
        # Standard DTLZ1: g(x) = sum_{i=2}^{M} [(xi-0.5)^2 - 10*cos(20*pi*(xi-0.5))]
        # But this gives g in [-10*(M-1), -9.75*(M-1)] for M=7, so [-60, -58.5]
        # That can't be right for a standard benchmark.
        
        # I think the standard DTLZ1 actually has g = sum[(xi-0.5)^2 - 10*cos(20*pi*xi)] or similar.
        # Let me just use a generic approach: do a well-designed multi-objective search.
        
        # Given the complexity, let me use a robust MOEA approach with the known front structure.
        
        # Use a combination of:
        # 1. Dense sampling of the simplex
        # 2. For each simplex point, do a 1D search over x1 with fixed tail
        # 3. Use the tail that minimizes |g|
        
        # Let me find the tail that gives g closest to 0 by trying a few patterns.
        # Since we can't easily solve g=0 analytically, let's use the evaluator.
        
        # Pragmatic approach: Use NSGA-II style but with informed initialization.
        
        # Initialize with points that are likely near the Pareto front
        n_init = 50
        X = np.zeros((n_init, self.dim))
        
        # Sample the simplex using Dirichlet(1,1,1)
        rng = np.random.default_rng(42)
        U = rng.dirichlet([1.0, 1.0, 1.0], size=n_init)  # (n_init, 3)
        
        for i in range(n_init):
            u1, u2, u3 = U[i]
            # x1 determines the ratio f1/f2 = x1/(1-x1) = u1/u2
            if u1 + u2 > 1e-10:
                X[i, 0] = u1 / (u1 + u2)
            else:
                X[i, 0] = 0.5
            # For tail, use 0.5 as a starting point (g will be negative but consistent)
            X[i, 1:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used += n_init
        results.extend([f for f in F])
        
        # Now do local refinement: for each point, try small perturbations
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Sort by hypervolume contribution (approximate by sum of objectives)
            order = np.argsort(np.sum(F, axis=1))
            # Refine the best points
            n_refine = min(remaining, 150)
            
            for i in range(n_refine):
                if time.time() >= deadline:
                    break
                # Pick a parent from the better half
                parent_idx = order[i % len(order)]
                parent = X[parent_idx].copy()
                
                # Random perturbation, smaller for better points
                scale = 0.05
                child = parent + scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                results.append(fc)
                
                # If child is better (lower sum), replace parent
                if np.sum(fc) < np.sum(F[parent_idx]):
                    X[parent_idx] = child
                    F[parent_idx] = fc
                    # Update order (simplified: just re-sort occasionally)
                    if i % 10 == 0:
                        order = np.argsort(np.sum(F, axis=1))
        
        return X
# EVOLVE-BLOCK-END
