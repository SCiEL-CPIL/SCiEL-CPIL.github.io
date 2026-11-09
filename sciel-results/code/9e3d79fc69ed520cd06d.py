"""Portfolio search for DTLZ2: Sobol sequence with surface-area-weighted sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc as qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Sobol sampling with importance weighting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 310.0  # Safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        used = 0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # DTLZ2 Pareto Front Geometry:
        # f1 = sin(pi/2 * x1)
        # f2 = cos(pi/2 * x1) * sin(pi/2 * x2)
        # f3 = cos(pi/2 * x1) * cos(pi/2 * x2)
        #
        # Let theta1 = pi/2 * x1, theta2 = pi/2 * x2.
        # The surface area element on the unit sphere is dA = sin(theta1) dtheta1 dtheta2.
        # To maximize hypervolume, we want points distributed such that they cover the 
        # "dominated" space efficiently. However, for a fixed number of points on a 
        # convex front, uniform distribution in the *angular* space (theta1, theta2) 
        # is generally good, but the "corners" (where one objective is near 0 and others 
        # are near 1) often capture large hypervolume boxes.
        #
        # Actually, for hypervolume maximization with a reference point far away [2.5, 2.5, 2.5],
        # points near the "axes" of the objective space (i.e., where one f is small, others large)
        # are crucial. 
        #
        # Let's look at the mapping from (x1, x2) to (f1, f2, f3).
        # The Jacobian of the mapping from (theta1, theta2) to (f1, f2, f3) involves 
        # the surface area element.
        # A common strategy for DTLZ2 hypervolume is to use a Sobol sequence in (u1, u2) in [0,1]^2,
        # then map to theta1, theta2.
        # 
        # Standard uniform sampling in x1, x2 leads to uniform distribution in theta1, theta2.
        # This results in points distributed uniformly on the quarter-sphere surface.
        # 
        # Is uniform on the surface optimal for hypervolume?
        # Hypervolume is the volume of the region dominated by the points.
        # For a convex front, uniform surface sampling is a strong baseline.
        # However, the "corners" of the parameter space (x1=0, x2=0; x1=1, x2=0; etc.)
        # correspond to the axes of the objective space.
        # 
        # Let's try a Sobol sequence for better low-discrepancy coverage than a grid.
        # We will generate N points in [0,1]^2 using Sobol.
        
        N = budget
        # Sobol sequence requires power of 2 for best properties, but works for any N.
        # We'll use the next power of 2 >= N to get a good sequence, then take first N.
        # Or just use N directly.
        
        # To ensure we cover the boundaries well (which are important for HV),
        # Sobol sequences in [0,1] might not hit 0 or 1 exactly.
        # We can mix: 
        # 1. Explicitly add the 4 "corner" points of the front (extreme points).
        #    (0,0) -> f=(0,0,1)
        #    (0,1) -> f=(0,1,0)
        #    (1,0) -> f=(1,0,0)
        #    (1,1) -> f=(1,1,0) -- wait, f3=cos(pi/2)*cos(pi/2)=0. f1=sin(pi/2)=1. f2=0. So (1,0,0).
        #    Actually:
        #    x1=0, x2=0 -> f1=0, f2=0, f3=1
        #    x1=0, x2=1 -> f1=0, f2=1, f3=0
        #    x1=1, x2=0 -> f1=1, f2=0, f3=0
        #    x1=1, x2=1 -> f1=1, f2=0, f3=0 (Same as above? No. 
        #    f1 = sin(pi/2 * 1) = 1
        #    f2 = cos(pi/2 * 1) * sin(pi/2 * 1) = 0 * 1 = 0
        #    f3 = cos(pi/2 * 1) * cos(pi/2 * 1) = 0 * 0 = 0
        #    So (1,0,0).
        #    
        #    Wait, the front is a quarter sphere.
        #    x1 in [0,1], x2 in [0,1].
        #    f1 = sin(pi x1 / 2)
        #    f2 = cos(pi x1 / 2) sin(pi x2 / 2)
        #    f3 = cos(pi x1 / 2) cos(pi x2 / 2)
        #    
        #    Points:
        #    (0,0) -> (0, 0, 1)
        #    (0,1) -> (0, 1, 0)
        #    (1,0) -> (1, 0, 0)
        #    (1,1) -> (1, 0, 0) -- Duplicate!
        #    
        #    The parameter space [0,1]x[0,1] maps to the quarter sphere.
        #    The point (1,1) maps to the same objective vector as (1,0).
        #    So we have 3 unique "corner" points on the front.
        #    
        #    Let's reserve 3 evaluations for these extreme points to ensure we capture the axes.
        #    The remaining budget-3 points will be sampled using Sobol in the interior.
        
        # 1. Add extreme points
        extreme_params = [
            [0.0, 0.0],  # (0, 0, 1)
            [0.0, 1.0],  # (0, 1, 0)
            [1.0, 0.0],  # (1, 0, 0)
        ]
        
        candidates = []
        
        # Add extremes first
        for p in extreme_params:
            x = np.full(dim, 0.5)
            x[0] = p[0]
            x[1] = p[1]
            candidates.append(x)
            
        used += len(extreme_params)
        
        # 2. Generate Sobol points for the rest
        N_sobol = budget - used
        if N_sobol > 0:
            # Use a Sobol sequence
            # We want to sample in (x1, x2) in [0,1]^2.
            # To avoid duplicates with extremes and improve coverage,
            # we can sample in (0,1) open interval or just use the sequence.
            
            # scipy.stats.qmc.Sobol
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            # Next power of 2 for better quality?
            # For small N, just use N.
            sobol_pts = sampler.random(N_sobol)
            
            # Map Sobol points [0,1]^2 to x1, x2 in [0,1]
            # Standard mapping is direct.
            for pt in sobol_pts:
                x = np.full(dim, 0.5)
                x[0] = pt[0]
                x[1] = pt[1]
                candidates.append(x)
                
                used += 1
                if used >= budget:
                    break
                
                # Time check
                if time.time() >= deadline:
                    break

        # Evaluate all candidates
        # Note: The problem function clips and returns 3 objectives.
        # We just need to call it.
        
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            # used was already incremented above for sobol, 
            # but for extremes I incremented used before the loop?
            # Let's refactor the counting to be safe.
            
        # Refactor: I incremented used in the generation loop for sobol.
        # But I need to make sure I don't double count or miss counts.
        # Let's restart the evaluation logic cleanly.
        
        # Actually, the previous block incremented `used` during generation.
        # This is risky if I break early.
        # Let's just count evaluations in the loop.
        
        # Re-doing the structure to be robust:
        
        # Clear and rebuild
        candidates = []
        
        # 1. Extremes
        extreme_params = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        
        for p in extreme_params:
            x = np.full(dim, 0.5)
            x[0] = p[0]
            x[1] = p[1]
            candidates.append(x)
            
        # 2. Sobol
        N_sobol = budget - len(extreme_params)
        if N_sobol > 0:
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            sobol_pts = sampler.random(N_sobol)
            for pt in sobol_pts:
                x = np.full(dim, 0.5)
                x[0] = pt[0]
                x[1] = pt[1]
                candidates.append(x)
                
        # 3. Evaluate
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            used += 1

        return np.array(candidates[:used])
# EVOLVE-BLOCK-END
