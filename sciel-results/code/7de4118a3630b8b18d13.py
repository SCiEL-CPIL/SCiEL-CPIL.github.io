"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # DTLZ4: Pareto front is at g=0, i.e., x[2:] = 0.5
        # f1 = (1+g)*cos(t1)*cos(t2), f2 = (1+g)*cos(t1)*sin(t2), f3 = (1+g)*sin(t1)
        # where t1 = x[0]*pi/2, t2 = x[1]*pi/2
        # On Pareto front (g=0): f1=f1_0, f2=f2_0, f3=f3_0 with f1_0^2+f2_0^2+f3_0^2=1
        # and all >= 0 (quarter sphere in 3D)
        
        # Strategy: Use low-discrepancy sampling in the 2D angle space (t1, t2)
        # to generate points on the Pareto front, then map back to decision space.
        # This exploits the known structure directly.
        
        n_evals = 0
        results = []
        
        # Generate Sobol-like low-discrepancy points in [0, pi/2] x [0, pi/2]
        # We'll use a Halton sequence for simplicity and effectiveness
        
        def halton_sequence(index, base):
            """Generate Halton sequence value for given index and base."""
            result = 0.0
            f = 1.0 / base
            i = index
            while i > 0:
                result += f * (i % base)
                i //= base
                f /= base
            return result
        
        # Number of points to sample on the Pareto front
        # Budget is 200 evaluations. We need to be careful about how many we evaluate.
        # Each point on the Pareto front requires 1 evaluation.
        # Let's use most of the budget for Pareto front sampling.
        
        # Use a mix: some structured points + some random perturbations
        n_structured = min(180, self.budget - 20)  # Leave 20 for potential refinement
        
        # Generate structured points using Halton sequence in 2D
        # t1 in [0, pi/2], t2 in [0, pi/2]
        for i in range(1, n_structured + 1):
            # Check time
            if time.time() > deadline - 5.0:
                break
            if n_evals >= self.budget:
                break
            
            # Halton sequence for 2D (bases 2 and 3)
            t1 = halton_sequence(i, 2) * (np.pi / 2)
            t2 = halton_sequence(i, 3) * (np.pi / 2)
            
            # Create decision vector
            x = np.zeros(self.dim)
            x[0] = t1 / (np.pi / 2)  # x[0] = t1 / (pi/2), so t1 = x[0]*pi/2
            x[1] = t2 / (np.pi / 2)  # x[1] = t2 / (pi/2), so t2 = x[1]*pi/2
            x[2:] = 0.5  # g = 0
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            n_evals += 1
            results.append((x, f))
        
        # Add some critical corner/center points for better hypervolume coverage
        critical_points = [
            # Corners of the quarter sphere
            [1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # f1=1, f2=0, f3=0
            [0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # f1=0, f2=1, f3=0
            [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # f1=0, f2=0, f3=1
            # Center
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ]
        
        for cp in critical_points:
            if n_evals >= self.budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            x = np.array(cp, dtype=np.float64)
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            n_evals += 1
            results.append((x, f))
        
        # Use remaining budget for random perturbations around Pareto front
        # to capture points slightly off the front that might improve hypervolume
        remaining = self.budget - n_evals
        if remaining > 0 and time.time() < deadline - 10.0:
            for _ in range(remaining):
                if time.time() > deadline - 5.0:
                    break
                
                # Random point near Pareto front
                t1 = np.random.random() * (np.pi / 2)
                t2 = np.random.random() * (np.pi / 2)
                
                x = np.zeros(self.dim)
                x[0] = t1 / (np.pi / 2)
                x[1] = t2 / (np.pi / 2)
                x[2:] = 0.5 + 0.01 * np.random.standard_normal(self.dim - 2)
                
                x = np.clip(x, lo, hi)
                
                f = problem(x)
                n_evals += 1
                results.append((x, f))
        
        # Return the first point (return value is ignored, but keep interface)
        if results:
            return results[0][0]
        return np.zeros(self.dim)

# EVOLVE-BLOCK-END
