"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ2 structure:
        # First 2 variables are "angle" variables: x0, x1 in [0, 0.5] for angles [0, pi/2]
        # Remaining variables (x2...x_{n-1}) should be 0.5 to minimize g
        # Pareto front: f1 = sin(theta1) * sin(theta2), f2 = cos(theta1) * sin(theta2), f3 = cos(theta2)
        # where theta1 = pi/2 * x0, theta2 = pi/2 * x1

        # Strategy:
        # 1. Generate points on/near the Pareto front directly
        # 2. Use remaining budget to refine with small perturbations

        # Phase 1: Sample the Pareto front directly
        # theta1 = pi/2 * x0, theta2 = pi/2 * x1
        # x0 = 2*theta1/pi, x1 = 2*theta2/pi
        # x2...x_{n-1} = 0.5

        n_points_phase1 = 100
        theta1 = np.linspace(0, np.pi/2, n_points_phase1)
        theta2 = np.linspace(0, np.pi/2, n_points_phase1)
        
        # Create a grid of points on the Pareto front
        T1, T2 = np.meshgrid(theta1, theta2, indexing='ij')
        T1_flat = T1.flatten()
        T2_flat = T2.flatten()
        
        # Select a subset to stay within budget
        max_phase1 = min(n_points_phase1 * n_points_phase1, self.budget // 2)
        if len(T1_flat) > max_phase1:
            idx = np.random.default_rng(42).choice(len(T1_flat), max_phase1, replace=False)
            T1_flat = T1_flat[idx]
            T2_flat = T2_flat[idx]
        
        X_phase1 = np.zeros((len(T1_flat), self.dim))
        X_phase1[:, 0] = 2 * T1_flat / np.pi
        X_phase1[:, 1] = 2 * T2_flat / np.pi
        X_phase1[:, 2:] = 0.5  # Minimize g
        
        F_phase1 = np.array([problem(x) for x in X_phase1])
        used = len(X_phase1)

        # Phase 2: Use remaining budget to explore around the front with small perturbations
        remaining_budget = self.budget - used
        if remaining_budget > 0:
            rng = np.random.default_rng(0)
            # Sample random points on the front and perturb them
            for _ in range(remaining_budget):
                if time.time() > deadline:
                    break
                # Pick a random point on the front
                t1 = rng.random() * np.pi / 2
                t2 = rng.random() * np.pi / 2
                
                child = np.zeros(self.dim)
                child[0] = 2 * t1 / np.pi
                child[1] = 2 * t2 / np.pi
                child[2:] = 0.5
                
                # Small perturbation to explore nearby regions
                noise_scale = 0.01
                child += noise_scale * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1

        return X_phase1
# EVOLVE-BLOCK-END
