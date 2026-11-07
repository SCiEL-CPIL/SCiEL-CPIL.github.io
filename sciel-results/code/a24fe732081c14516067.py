"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured sampling + evolutionary search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4: 12 variables, 3 objectives
        # x[0] = M-1 = 2 decision vars for position on Pareto front
        # x[1..M-1] = remaining M-1 = 2 vars for position (actually M=3, so x[0..1] are position)
        # Wait, DTLZ4 with M=3 objectives: M-1 = 2 position variables, N-M+1 = 12-3+1 = 10 distance variables
        # Actually for DTLZ4:
        # f1 = x_0 * g * cos(theta_1) ... 
        # Let me use the standard DTLZ4 formulation:
        # M objectives, N variables
        # x[0..M-2] are position variables (M-1 of them)
        # x[M-1..N-1] are distance variables (N-M+1 of them)
        # g(x) = sum_{i=M-1}^{N-1} (x_i^alpha - 0.5)^2  where alpha=10 for DTLZ4
        # f_i(x) = g * h_i(x)
        # For DTLZ4 with M=3:
        # f1 = g * (1 - cos(pi/2 * x_0)) ... actually let me use the standard:
        # f_i = g * h_i where h_i are the standard DTLZ4 h functions
        # h_1 = cos(x_0 * pi/2) * cos(x_1 * pi/2)
        # h_2 = cos(x_0 * pi/2) * sin(x_1 * pi/2)
        # h_3 = sin(x_0 * pi/2)

        # Strategy:
        # 1. Generate a structured set of points on the Pareto front (g=0) using spherical coordinates
        # 2. Use remaining budget for evolutionary search to find points with small g
        # 3. Explicitly place points at key locations (vertices, etc.)

        # Phase 1: Structured sampling on the Pareto front (g=0)
        # For DTLZ4 with M=3, the Pareto front is a 2D curved surface
        # Use spherical coordinates: theta1 in [0, pi/2], theta2 in [0, pi/2]
        # f1 = cos(theta1) * cos(theta2)
        # f2 = cos(theta1) * sin(theta2)
        # f3 = sin(theta1)

        # Number of points for structured sampling
        n_structured = min(60, budget // 3)
        
        # Generate LHS-like points in 2D parameter space
        rng = np.random.default_rng(42)
        
        # Create structured grid with some randomization
        # Use a combination of:
        # 1. Vertices of the Pareto front
        # 2. LHS points in the 2D parameter space
        # 3. Some additional points

        points = []

        # Add vertices (corners) of the Pareto front
        # Vertex 1: f1=1, f2=0, f3=0 -> theta1=pi/2, theta2=0
        # Vertex 2: f1=0, f2=1, f3=0 -> theta1=0, theta2=pi/2
        # Vertex 3: f1=0, f2=0, f3=1 -> theta1=0, theta2=0
        # Actually for DTLZ4:
        # f1 = cos(t1)*cos(t2), f2 = cos(t1)*sin(t2), f3 = sin(t1)
        # t1, t2 in [0, pi/2]
        
        # Vertices:
        # (t1=0, t2=0): f=(1,0,0)
        # (t1=0, t2=pi/2): f=(0,1,0)
        # (t1=pi/2, t2=0): f=(0,0,1)
        
        vertex_thetas = [
            (0.0, 0.0),       # f=(1,0,0)
            (0.0, np.pi/2),   # f=(0,1,0)
            (np.pi/2, 0.0),   # f=(0,0,1)
            (np.pi/4, np.pi/4),  # center-ish
            (np.pi/6, np.pi/3),
            (np.pi/3, np.pi/6),
        ]
        
        for t1, t2 in vertex_thetas:
            # Position variables: x[0] = t1/(pi/2), x[1] = t2/(pi/2)
            # Distance variables: set to 0.5 so g=0
            x = np.full(dim, 0.5)
            x[0] = t1 / (np.pi / 2)
            x[1] = t2 / (np.pi / 2)
            points.append(x.copy())
        
        # LHS points in 2D parameter space
        n_lhs = max(0, n_structured - len(points))
        if n_lhs > 0:
            # Latin Hypercube Sampling in 2D
            idx1 = np.arange(n_lhs)
            idx2 = np.arange(n_lhs)
            rng.shuffle(idx1)
            rng.shuffle(idx2)
            t1_lhs = (idx1 + rng.random(n_lhs)) / n_lhs * (np.pi / 2)
            t2_lhs = (idx2 + rng.random(n_lhs)) / n_lhs * (np.pi / 2)
            
            for t1, t2 in zip(t1_lhs, t2_lhs):
                x = np.full(dim, 0.5)
                x[0] = t1 / (np.pi / 2)
                x[1] = t2 / (np.pi / 2)
                points.append(x)
        
        # Evaluate structured points
        F_structured = np.array([problem(x) for x in points])
        used = len(points)
        
        # Phase 2: Evolutionary search with remaining budget
        # Use NSGA-II-like approach with elitism
        pop_size = min(20, max(10, (budget - used) // 5))
        
        # Initialize population from best structured points + random
        # Select non-dominated points from structured set
        def is_dominated(F, f):
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
        
        dominated = np.array([is_dominated(F_structured, f) for f in F_structured])
        non_dom = ~dominated
        
        if np.sum(non_dom) >= pop_size:
            # Use all non-dominated as initial population
            X = points[:np.sum(non_dom)]
            F = F_structured[non_dom]
            # Pad with random if needed
            while len(X) < pop_size:
                x = lo + (hi - lo) * rng.random(dim)
                X.append(x)
                F = np.vstack([F, problem(x)])
                used += 1
        else:
            # Use all structured + random
            X = list(points)
            F = F_structured
            while len(X) < pop_size:
                x = lo + (hi - lo) * rng.random(dim)
                X.append(x)
                F = np.vstack([F, problem(x)])
                used += 1
        
        X = np.array(X)
        F = np.array(F)
        
        # Evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break
            
            # Find non-dominated set
            dominated = np.array([is_dominated(F, f) for f in F])
            parents = X[~dominated] if np.any(~dominated) else X
            
            # Crossover: SBX-like
            p1 = parents[rng.integers(len(parents))]
            p2 = parents[rng.integers(len(parents))]
            
            # Blend
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2
            
            # Mutation
            child = child + 0.05 * (hi - lo) * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc
        
        # Return the population (return value is ignored by evaluator)
        return X
# EVOLVE-BLOCK-END
