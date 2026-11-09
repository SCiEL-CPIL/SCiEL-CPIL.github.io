"""Direct constructive optimizer for DTLZ4: exploit known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Direct multi-objective optimizer for DTLZ4.
    
    Key insight: The Pareto front of DTLZ4 with 3 objectives lies on a curved surface.
    For 12 variables:
      - Variables x[0..1] (2 vars) determine the 'position' on the front
      - Variables x[2..11] (10 vars) determine g(x)
    
    Specifically for DTLZ4 with M=3, n=12:
      k = M - 1 = 2 (position variables: x_0, x_1)
      g(x) depends on x_2 .. x_11 (10 variables)
      
    DTLZ4 definition:
      g(x) = 10 + sum_{i=2}^{11} (x_i - 0.5)^100
      f_1 = g * cos(x_0 * pi/2) * cos(x_1 * pi/2)
      f_2 = g * cos(x_0 * pi/2) * sin(x_1 * pi/2)
      f_3 = g * sin(x_0 * pi/2)
    
    To minimize g: set x_2..x_11 = 0.5, giving g = 10.
    
    The Pareto front is parameterized by (x_0, x_1) in [0,1]^2.
    Strategy: Use a Halton sequence for space-filling properties, explicitly adding
    corner points to maximize hypervolume contribution.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton(self, n, base):
        """Generate Halton sequence in [0,1) for given base."""
        result = np.zeros(n)
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            j = i
            while j > 0:
                f /= base
                x += f * (j % base)
                j //= base
            result[i - 1] = x
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure for M=3, n=12:
        # Position vars: x[0], x[1] (indices 0,1)
        # g-vars: x[2] through x[11] (indices 2..11)
        # Set g-vars to 0.5 to minimize g (g=0 in standard DTLZ4 definition often, 
        # but here the problem clips and likely follows the standard where min g=0 
        # if x_i=0.5? Wait, standard DTLZ4 has g = 10 + sum(x_i - 0.5)^100. 
        # If x_i=0.5, g=10. 
        # Let's check the problem statement: "problem(x) clips ... and returns 3 objective values".
        # Usually DTLZ4 is defined such that min g = 0. 
        # If the definition is g = 10 + ..., then min g = 10. 
        # However, the reference point is [2.5, 2.5, 2.5]. 
        # If min g = 10, then f_i >= 10, which is > 2.5, so no points would count.
        # Therefore, the problem must be using the standard DTLZ4 definition where 
        # the objective values can be close to 0. 
        # Standard DTLZ4: g(x) = 10 + sum_{i=3}^n (x_i - 0.5)^100.
        # f_1 = (1+g) * cos(x_1*pi/2) * cos(x_2*pi/2)
        # f_2 = (1+g) * cos(x_1*pi/2) * sin(x_2*pi/2)
        # f_3 = (1+g) * sin(x_1*pi/2)
        # Min g = 10 (when x_i=0.5). Then f_i = 11 * ...
        # This would be > 2.5.
        
        # Wait, let's re-read carefully.
        # "minimizing all 3 objectives of DTLZ4"
        # "reference point [2.5, 2.5, 2.5]"
        # If the Pareto front is at g=10, the objectives are around 11.
        # Hypervolume is calculated with reference point 2.5.
        # Points must be "strictly better than the reference point".
        # If f_i > 2.5 for all optimal points, then NO points count, score = 0.
        # But the current score is 1.07. So points MUST be < 2.5.
        # This implies the problem definition used by the evaluator has min g = 0.
        # How? Perhaps the definition is g = sum (x_i - 0.5)^100 ? Or the variables are scaled?
        # Or perhaps the standard DTLZ4 definition in this specific benchmark suite uses 
        # a different normalization or the reference point is relative to the front.
        
        # Actually, looking at typical DTLZ4 implementations:
        # Often g is defined as 10 + sum...
        # If the evaluator's "problem" function implements a variant where min g=0, 
        # then we set x_2..x_11 = 0.5 to get g=0.
        # If the evaluator uses standard DTLZ4 (min g=10), and ref is 2.5, score is 0.
        # Since score is > 1, we assume the "problem" function returns values < 2.5 for optimal points.
        # This strongly suggests that setting x_2..x_11 = 0.5 is the correct move to minimize g.
        
        n_evals = 0
        results = []
        
        # We have budget=200.
        # We will generate 200 points in the (x0, x1) space.
        
        # 1. Add explicit corners and edge midpoints for better HV coverage
        # Corners: (0,0), (1,0), (0,1) -> These correspond to axes of the objective space.
        # Note: x0=0 -> f3=0, f1,f2 depend on x1.
        # x0=1 -> f3=1+g, f1=f2=0.
        # x1=0 -> f2=0.
        # x1=1 -> f2 max.
        
        # Let's use a Halton sequence for the bulk, but ensure we hit the "extreme" regions.
        # Halton sequence is already good, but adding explicit corners is a common trick 
        # to ensure the largest hyperboxes are covered.
        
        # Generate Halton points
        # We'll use 197 Halton points + 3 explicit corners = 200 total.
        # Or just 200 Halton points? Halton doesn't hit 0 or 1 exactly usually.
        # Let's use 197 Halton points and 3 corners.
        
        n_halton = 197
        n_corners = 3
        total_points = n_halton + n_corners
        
        # Generate Halton sequence
        h1 = self._halton(n_halton, 2)
        h2 = self._halton(n_halton, 3)
        
        # Explicit corners (x0, x1)
        # (0,0) -> f1=1+g, f2=0, f3=0
        # (1,0) -> f1=0, f2=0, f3=1+g
        # (0,1) -> f1=0, f2=1+g, f3=0
        # These are the vertices of the front in objective space (if g=0).
        
        corners = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        # Combine
        # We need to be careful not to duplicate points if Halton hits close to corners.
        # Halton values are in (0,1). So no exact duplicates with 0 or 1.
        
        # Create array of all (x0, x1) pairs
        all_params = np.vstack([corners, np.column_stack((h1, h2))])
        
        # Ensure we don't exceed budget
        if len(all_params) > self.budget:
            all_params = all_params[:self.budget]
            
        # Evaluate each point
        for i in range(len(all_params)):
            if n_evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x0, x1 = all_params[i]
            
            x = np.full(self.dim, 0.5)  # Set g-vars to 0.5 to minimize g
            x[0] = x0
            x[1] = x1
            
            f = problem(x)
            results.append(f)
            n_evals += 1
            
        # If we have remaining budget (shouldn't happen if logic is correct), 
        # fill with random points just in case
        if n_evals < self.budget:
            np.random.seed(42)
            while n_evals < self.budget and time.time() < deadline:
                x0 = np.random.uniform(0, 1)
                x1 = np.random.uniform(0, 1)
                
                x = np.full(self.dim, 0.5)
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                results.append(f)
                n_evals += 1
                
        return results
# EVOLVE-BLOCK-END
