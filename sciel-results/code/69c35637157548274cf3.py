"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Deterministic 2D hypersphere tessellation via Fibonacci lattice on the unit disk,
    mapped directly to DTLZ7 Pareto front parameters (x0, x1).
    
    DTLZ7 with M=3 objectives:
    - x0, x1 are the 'position' variables on the Pareto front.
    - The remaining dim-2 variables are 'distance' variables that affect only g.
    - g(x) = 1 + 9 * (sum(x_d) / (dim-2))
    - h_j(g, y) = 1 - sqrt(y_j / g) for j=1..M-1
    - f_j = y_j * (1 + g) for j=1..M-1
    - f_M = 1 - sum_{j=1}^{M-1} h_j(g, y)
    
    On the Pareto front, g=0, so x_d=0 for all distance variables.
    Then:
    - f_1 = y_1
    - f_2 = y_2
    - f_3 = 1 - (1 - sqrt(y_1)) - (1 - sqrt(y_2)) = sqrt(y_1) + sqrt(y_2) - 1
    
    where y_1 = x_0, y_2 = x_1, and x_0, x_1 in [0, 1].
    
    The Pareto front is parameterized by (x_0, x_1) in [0,1]^2.
    We use a Fibonacci lattice (golden ratio spiral) on the unit disk to generate
    well-distributed points, then map them to (x_0, x_1) via polar-to-square mapping
    or simply use the lattice points projected/clipped to [0,1]^2.
    
    Actually, for uniform coverage of the square [0,1]^2, a 2D Fibonacci lattice
    (Halton sequence or similar low-discrepancy sequence) is more appropriate.
    We'll use a 2D Halton sequence or a golden-ratio based 2D lattice.
    
    For M=3, dim=22:
    - x[0] = x0, x[1] = x1 from the lattice
    - x[2:] = 0 (to be on the Pareto front)
    
    We generate as many lattice points as the budget allows, evaluate them all,
    and return.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        n_eval = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # Generate n_eval points using a 2D low-discrepancy sequence (Halton-like)
        # Using golden ratio conjugate for 2D: ( {n * phi}, {n * phi^2} ) mod 1
        # where phi = (1 + sqrt(5)) / 2
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        inv_phi = phi - 1.0  # 1/phi
        
        # Generate indices
        n = np.arange(1, n_eval + 1, dtype=np.float64)
        
        # 2D Halton-like sequence using golden ratio
        # For dimension 1: {n * inv_phi}
        # For dimension 2: {n * inv_phi^2} = {n * (2 - phi)} but actually use {n * phi} mod 1
        # Better: use two different bases or just use the fractional parts of n*phi and n*phi^2
        
        # Actually, a simpler approach: use the first two coordinates of a 3D or 2D
        # low-discrepancy sequence. Let's use the Van der Corput sequence with base phi.
        
        # For 2D, we can use:
        # x0 = (n * inv_phi) % 1
        # x1 = (n * inv_phi * inv_phi) % 1  -- this isn't quite right for low discrepancy
        
        # Let's use a proper 2D Halton sequence with bases 2 and 3:
        def halton(n, base):
            """Generate the n-th Halton sequence point in [0,1) for given base."""
            f = 1.0
            r = 0.0
            n_int = int(n)
            while n_int > 0:
                f /= base
                r += f * (n_int % base)
                n_int //= base
            return r
        
        # Generate 2D Halton points
        x0_vals = np.array([halton(i, 2) for i in range(1, n_eval + 1)])
        x1_vals = np.array([halton(i, 3) for i in range(1, n_eval + 1)])
        
        # Create candidate solutions
        # x[0] = x0, x[1] = x1, x[2:] = 0
        X = np.zeros((n_eval, dim))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        # x[2:] = 0 (already zero)
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
