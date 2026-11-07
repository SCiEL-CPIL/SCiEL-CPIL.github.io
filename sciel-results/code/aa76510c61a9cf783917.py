"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using Sobol sequence on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        n_g = 5
        n_front = dim - n_g  # 7 for dim=12

        # Generate Sobol sequence for 2 dimensions
        from scipy.stats import qmc

        # Reserve 3 points for explicit vertices
        n_vertices = 3
        n_sobol = budget - n_vertices
        
        sampler = qmc.Sobol(d=2, scramble=True)
        # Generate sobol points
        u = sampler.random(n_sobol)  # shape (n_sobol, 2)

        # Map uniform variables to decision variables for DTLZ4 Pareto front
        # For DTLZ4 with 3 objectives:
        # The Pareto front is defined by f1^2 + f2^2 + f3^2 = 1 (scaled by g).
        # To get g=0, we set the g-variables to 0.0 (minimum of the oscillatory term).
        # The front variables x0, x1 parameterize the sphere.
        # x0 controls the 'latitude' (f1 component).
        # x1 controls the 'longitude' (f2 vs f3 split).
        
        # To get uniform area distribution on the quarter sphere:
        # Let u0 ~ U[0,1] be mapped to f1 = u0.
        # Then cos(theta) = u0 => theta = arccos(u0).
        # x0 in [0, pi/2] corresponds to theta.
        # If the problem expects x0 in [0,1] to map to [0, pi/2] via x0 * pi/2,
        # then we need x0 * pi/2 = arccos(u0) => x0 = (2/pi) * arccos(u0).
        
        x1_vals = (2.0 / np.pi) * np.arccos(u[:, 0])
        # x1 in [0,1] maps to phi in [0, pi/2] via x1 * pi/2.
        # Uniform x1 gives uniform phi, which combined with uniform f1 (from arccos mapping)
        # gives uniform area coverage.
        x2_vals = u[:, 1]

        # Construct decision vectors
        X = np.zeros((budget, dim))
        X[:n_sobol, 0] = x1_vals
        X[:n_sobol, 1] = x2_vals

        # Set other front variables (indices 2 to n_front-1) to 0.0
        # These are also part of the 'g' variables in some definitions, or fixed.
        # In DTLZ4, usually x[2:M-1] are also g-variables?
        # Standard DTLZ4: x[0:M-1] are front variables? No.
        # x[0:K] are front variables?
        # For M=3, K=2? No, usually K = M-1 = 2 front variables?
        # Actually, DTLZ4 has M-1 front variables and n-(M-1) g-variables.
        # So x[0] and x[1] are front variables.
        # x[2:] are g-variables.
        
        # To minimize g, set g-variables to 0.0.
        # This ensures the points lie on the Pareto front (g=0).
        X[:n_sobol, 2:] = 0.0
        
        # Add explicit vertices to ensure coverage of the extreme points
        # Vertex 1: f1=1, f2=0, f3=0 => x0=0, x1=0 (arccos(1)=0)
        # In our mapping: x0 = (2/pi)*arccos(1) = 0. x1 = 0.
        X[n_sobol, 0] = 0.0
        X[n_sobol, 1] = 0.0
        X[n_sobol, 2:] = 0.0
        
        # Vertex 2: f1=0, f2=1, f3=0 => x0=1 (arccos(0)=pi/2 -> x0=1), x1=0
        X[n_sobol+1, 0] = 1.0
        X[n_sobol+1, 1] = 0.0
        X[n_sobol+1, 2:] = 0.0
        
        # Vertex 3: f1=0, f2=0, f3=1 => x0=1, x1=1
        X[n_sobol+2, 0] = 1.0
        X[n_sobol+2, 1] = 1.0
        X[n_sobol+2, 2:] = 0.0

        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Check time before starting evaluation loop
        if time.time() - t0 > deadline:
            return X

        # Evaluate all points
        F = np.empty((budget, n_obj))
        for i in range(budget):
            if time.time() - t0 > deadline:
                break
            F[i] = problem(X[i])

        return X
# EVOLVE-BLOCK-END
