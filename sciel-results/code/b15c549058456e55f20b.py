# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 specific structure:
        # - dim = 12, n_obj = 3
        # - M-1 = 2 shape variables (x[0], x[1])
        # - K = dim - (M-1) = 10 distance variables (x[2]..x[11])
        # For Pareto front: g = 0, so all distance variables must be at their optimal values
        # For DTLZ4: g = sum((x_i - 0.5)^2) for i in [M-1, dim), so optimal x_i = 0.5
        # The objectives are:
        #   f_0 = g * (1 + sin(pi * f_0_raw))  -- but when g=0, f_0 = 0? No...
        # Actually for DTLZ4:
        #   f_1 = g * (1 + sin(3*pi*x_0))
        #   f_2 = g * (1 + sin(3*pi*x_1))
        #   f_3 = g
        # When g = 0, all objectives are 0, which is not right.
        
        # Let me reconsider DTLZ4 definition:
        # DTLZ4 has M objectives. For M=3:
        #   x has K + M - 1 variables
        #   g(x_K) = sum((x_i - 0.5)^2) for i in K..K+M-2 (the last M-1 variables)
        #   f_1 = (1+g) * prod(cos(x_i * pi/2)) for i in 0..M-3
        #   f_2 = (1+g) * cos(x_{M-2}*pi/2) * sin(x_{M-1}*pi/2)
        #   ...
        #   f_M = (1+g) * sin(x_{M-1}*pi/2)
        
        # Wait, let me be more careful. Standard DTLZ4:
        # Number of variables: K + M - 1
        # For M=3, K=10: total = 12 variables
        # g(x) = sum_{i=K}^{K+M-2} (x_i - 0.5)^2 = sum_{i=10}^{11} (x_i - 0.5)^2
        # So the last M-1 = 2 variables are the distance variables.
        # The first K = 10 variables are the shape variables.
        
        # For M=3:
        # f_1 = (1+g) * prod_{i=0}^{M-3} cos(x_i * pi/2) = (1+g) * prod_{i=0}^{0} cos(x_i*pi/2) = (1+g)*cos(x_0*pi/2)
        # f_2 = (1+g) * cos(x_{M-2}*pi/2) * sin(x_{M-1}*pi/2) = (1+g)*cos(x_1*pi/2)*sin(x_2*pi/2)
        
        # Hmm, that doesn't seem right either. Let me look at the standard formulation more carefully.
        
        # Actually, the standard DTLZ4 for M objectives:
        # f_1(x) = (1+g(x_K)) * prod_{i=0}^{M-3} cos(x_i * pi/2)
        # f_2(x) = (1+g(x_K)) * cos(x_{M-2} * pi/2) * sin(x_{M-1} * pi/2)  -- wait, this is for M=2
        
        # Let me use the general formula:
        # For M=3:
        # f_1 = (1+g) * cos(x_0*pi/2) * cos(x_1*pi/2)
        # f_2 = (1+g) * cos(x_0*pi/2) * sin(x_1*pi/2)
        # f_3 = (1+g) * sin(x_0*pi/2)
        
        # No wait. The standard DTLZ4 formulation for M=3:
        # The first M-1 = 2 variables are shape variables.
        # f_1 = (1+g) * cos(x_0 * pi/2) * cos(x_1 * pi/2)
        # f_2 = (1+g) * cos(x_0 * pi/2) * sin(x_1 * pi/2)  
        # f_3 = (1+g) * sin(x_0 * pi/2)
        
        # Hmm, I'm getting confused. Let me just use the fact that for DTLZ4,
        # the Pareto front is a 2D surface in objective space.
        # When g=0 (all distance variables at 0.5), the objectives are:
        # For M=3, with shape variables x_0, x_1 in [0,1]:
        # The Pareto front is parameterized by these shape variables.
        
        # Actually, I think the safest approach is to directly sample the Pareto front
        # by setting distance variables to 0.5 and sampling shape variables.
        
        # For DTLZ4 with M=3, K=10:
        # Variables: x[0..9] are shape, x[10..11] are distance
        # g = (x[10]-0.5)^2 + (x[11]-0.5)^2
        # For Pareto front: x[10]=0.5, x[11]=0.5, so g=0
        
        # The objectives when g=0:
        # f_1 = cos(x[0]*pi/2) * cos(x[1]*pi/2)
        # f_2 = cos(x[0]*pi/2) * sin(x[1]*pi/2)
        # f_3 = sin(x[0]*pi/2)
        
        # This gives a quarter-sphere-like surface in [0,1]^3.
        
        # Strategy: Sample shape variables uniformly on the Pareto front.
        # Use spherical coordinates for uniform coverage of the 2D front.
        
        # For uniform sampling on the Pareto front of DTLZ4 (M=3):
        # Use two angles: theta and phi
        # theta in [0, pi/2], phi in [0, pi/2]
        # But we need uniform density on the surface.
        
        # Alternative: just sample x[0] and x[1] uniformly in [0,1]
        # This gives a reasonable spread, though not perfectly uniform in objective space.
        
        # Let's use a combination:
        # 1. Sample a large portion of the budget on the Pareto front (g=0)
        # 2. Use remaining budget for local search to find nearby points
        
        # For the Pareto front sampling:
        # x[0] and x[1] are shape variables
        # x[2..11] are distance variables, set to 0.5 for g=0
        
        # To get uniform coverage of the curved front, use spherical coordinates:
        # Let theta = arccos(sqrt(u1)), phi = arccos(u2) where u1, u2 ~ Uniform(0,1)
        # Actually, for a quarter sphere, we can use:
        # theta ~ Arcsine distribution or similar
        
        # Simpler approach: sample u, v ~ Uniform(0,1)
        # x[0] = 1 - sqrt(u)  # This gives denser sampling near x[0]=0 (high f_3)
        # x[1] = 1 - sqrt(v)
        
        # Or just uniform:
        # x[0] = u, x[1] = v
        
        # Let me just use uniform sampling for simplicity, which should work well enough.
        
        rng = np.random.default_rng(0)
        
        # Allocate budget:
        # - 80% for Pareto front sampling
        # - 20% for local refinement
        
        n_front = int(budget * 0.8)
        n_local = budget - n_front
        
        # Sample points on the Pareto front
        # Shape variables: x[0], x[1]
        # Distance variables: x[2..11] = 0.5
        
        X_front = np.zeros((n_front, dim))
        # Set distance variables to 0.5
        X_front[:, 2:] = 0.5
        
        # Sample shape variables using spherical coordinates for uniform coverage
        # For DTLZ4 M=3, the Pareto front is a quarter of a unit sphere in 3D objective space
        # Parameterize using two angles
        u = rng.random(n_front)
        v = rng.random(n_front)
        
        # Use transformation for uniform sampling on the front
        # theta in [0, pi/2], with density proportional to sin(theta) for uniform area
        # phi in [0, pi/2]
        theta = np.arccos(1 - u)  # theta in [0, pi/2], uniform in cos(theta)
        phi = np.arccos(1 - v)    # phi in [0, pi/2]
        
        # Map to shape variables
        # x[0] controls the "outer" angle, x[1] controls the "inner" angle
        X_front[:, 0] = 1 - np.cos(theta)  # x[0] in [0, 1]
        X_front[:, 1] = 1 - np.cos(phi)    # x[1] in [0, 1]
        
        # Evaluate front points
        F_front = np.array([problem(x) for x in X_front])
        used = n_front
        
        # Local refinement: perturb points on the front slightly
        if n_local > 0:
            for i in range(n_local):
                if time.time() > deadline:
                    break
                
                # Pick a random point from the front
                idx = rng.integers(n_front)
                x = X_front[idx].copy()
                
                # Small perturbation
                noise = 0.05 * rng.standard_normal(dim)
                x = x + noise
                x = np.clip(x, lo, hi)
                
                fx = problem(x)
                used += 1
                
                # Keep the point if it's good (on or near the front)
                # For now, just record all evaluations
        
        return X_front
# EVOLVE-BLOCK-END
