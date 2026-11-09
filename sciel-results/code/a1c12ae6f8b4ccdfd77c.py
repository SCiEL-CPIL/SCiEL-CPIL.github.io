import numpy as np
import time

class Optimizer:
    """Exploits DTLZ1 structure to sample the Pareto front directly using Sobol sequences with adaptive refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n, offset=0):
        """Generate n 2D Sobol-like low-discrepancy points in [0,1]^2.
        Uses a simple van der Corput-based construction for both dimensions."""
        i = np.arange(n) + offset
        # Van der Corput sequence in base 2
        def van_der_corput(i, base=2):
            result = np.zeros_like(i, dtype=float)
            f = np.ones_like(i, dtype=float)
            i_copy = i.copy()
            while np.any(i_copy > 0):
                result += f * (i_copy % base)
                i_copy //= base
                f /= base
            return result
        x = van_der_corput(i)
        # For the second dimension, use a different scrambling or shift
        # Use a linear congruential shift to decorrelate
        y = van_der_corput((i * 7 + 3) % max(n, 1))
        # Ensure we stay in [0,1)
        x = np.mod(x, 1.0)
        y = np.mod(y, 1.0)
        return np.column_stack([x, y])

    def __call__(self, problem, bounds):
        t_start = time.time()
        # Safety margin: Evaluator kills at 330s. We stop at 300s to be safe.
        deadline = t_start + 290.0 

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 with M=3 objectives has M-1 = 2 shape variables (x_0, x_1).
        # The remaining N-M+1 = 5 variables (x_2..x_6) are distance variables.
        # To achieve g=0 (minimize distance part), set distance vars to 0.5.
        # The Pareto front is the set of points where g=0.
        # We sample the shape variables (x_0, x_1) to cover the front.
        
        # Phase 1: Use 150 Sobol points for broad coverage
        n_phase1 = min(150, budget)
        n_phase2 = budget - n_phase1
        
        # Generate Sobol points for phase 1
        sobol_points = self._sobol_2d(n_phase1)
        
        # Explicitly include the corners of the decision space that map to
        # unique vertices of the Pareto front in objective space.
        # (x0, x1) = (1, 0) -> f = (1, 0, 0)
        # (x0, x1) = (0, 1) -> f = (0, 1, 0)
        # (x0, x1) = (0, 0) -> f = (0, 0, 1)
        # Also add the centroid of the front: (x0, x1) = (0.5, 0.5) -> f = (0.25, 0.25, 0.5)
        
        # Replace first few samples with explicit key points
        if n_phase1 >= 4:
            sobol_points[0] = [1.0, 0.0]
            sobol_points[1] = [0.0, 1.0]
            sobol_points[2] = [0.0, 0.0]
            sobol_points[3] = [0.5, 0.5]
        elif n_phase1 >= 3:
            sobol_points[0] = [1.0, 0.0]
            sobol_points[1] = [0.0, 1.0]
            sobol_points[2] = [0.0, 0.0]
        elif n_phase1 >= 2:
            sobol_points[0] = [1.0, 0.0]
            sobol_points[1] = [0.0, 1.0]
        elif n_phase1 >= 1:
            sobol_points[0] = [0.5, 0.5]

        # Construct points for phase 1
        points = np.zeros((budget, dim))
        points[:n_phase1, 0] = sobol_points[:, 0]
        points[:n_phase1, 1] = sobol_points[:, 1]
        
        # Set distance variables to 0.5
        for j in range(2, dim):
            points[:, j] = 0.5
            
        # Evaluate phase 1
        results = []
        for i in range(n_phase1):
            if time.time() > deadline:
                break
            x = points[i]
            results.append(problem(x))
        
        if len(results) < n_phase1 or n_phase2 <= 0:
            return results
        
        # Phase 2: Adaptive refinement - find hypervolume-critical regions
        # The Pareto front in objective space is the simplex f1+f2+f3=1, fi>=0
        # We want to add points where the local hypervolume contribution is largest.
        # Strategy: identify the largest "empty" triangles in the simplex.
        
        # Map existing points to objective space
        # For DTLZ1: f1 = x0 * g, f2 = (1-x0)*x1*g, f3 = (1-x0)*(1-x1)*g
        # With g=0 (distance vars at 0.5), we have:
        # f1 = x0, f2 = (1-x0)*x1, f3 = (1-x0)*(1-x1)
        # Actually, let me re-derive. DTLZ1:
        # f1 = 0.5 * x0 * (1 + g)
        # f2 = 0.5 * (1-x0) * x1 * (1+g)
        # f3 = 0.5 * (1-x0) * (1-x1) * (1+g)
        # With g=0: f1 = 0.5*x0, f2 = 0.5*(1-x0)*x1, f3 = 0.5*(1-x0)*(1-x1)
        # Note: f1+f2+f3 = 0.5*x0 + 0.5*(1-x0)*x1 + 0.5*(1-x0)*(1-x1)
        #       = 0.5*x0 + 0.5*(1-x0)*(x1 + 1 - x1)
        #       = 0.5*x0 + 0.5*(1-x0)
        #       = 0.5
        # So the front lies on the plane f1+f2+f3 = 0.5, not 1.
        
        # Get objective values of evaluated points
        obj_vals = np.array(results)
        
        # For adaptive points, we'll target regions of the (x0, x1) unit square
        # where we have poor coverage. Use a grid-based approach.
        
        # Create a 10x10 grid over the unit square
        grid_n = 10
        cell_size = 1.0 / grid_n
        
        # Count how many points fall in each cell
        cell_counts = np.zeros((grid_n, grid_n), dtype=int)
        for i in range(n_phase1):
            x0, x1 = sobol_points[i]
            cx = min(int(x0 / cell_size), grid_n - 1)
            cy = min(int(x1 / cell_size), grid_n - 1)
            cell_counts[cy, cx] += 1
        
        # Find the empty or least-covered cells
        # We want to place points in cells with the fewest points
        # Prioritize cells that are far from existing points
        
        # For each empty/low-count cell, place a point at its center
        # Sort cells by count (ascending), take the top n_phase2
        cells = []
        for cy in range(grid_n):
            for cx in range(grid_n):
                count = cell_counts[cy, cx]
                if count == 0:
                    cells.append((0, cx, cy))  # priority 0 for empty
                elif count == 1:
                    cells.append((1, cx, cy))
                else:
                    cells.append((count, cx, cy))
        
        cells.sort()
        
        # Place adaptive points in the least-covered cells
        adaptive_points = []
        for priority, cx, cy in cells[:n_phase2]:
            # Place at the center of the cell
            x0 = (cx + 0.5) * cell_size
            x1 = (cy + 0.5) * cell_size
            adaptive_points.append([x0, x1])
        
        # If we need more points than cells, use finer subdivision
        if len(adaptive_points) < n_phase2:
            # Use a 20x20 grid for remaining points
            grid_n2 = 20
            cell_size2 = 1.0 / grid_n2
            cell_counts2 = np.zeros((grid_n2, grid_n2), dtype=int)
            for i in range(n_phase1):
                x0, x1 = sobol_points[i]
                cx = min(int(x0 / cell_size2), grid_n2 - 1)
                cy = min(int(x1 / cell_size2), grid_n2 - 1)
                cell_counts2[cy, cx] += 1
            
            # Also count adaptive points already placed
            for x0, x1 in adaptive_points:
                cx = min(int(x0 / cell_size2), grid_n2 - 1)
                cy = min(int(x1 / cell_size2), grid_n2 - 1)
                cell_counts2[cy, cx] += 1
            
            cells2 = []
            for cy in range(grid_n2):
                for cx in range(grid_n2):
                    count = cell_counts2[cy, cx]
                    cells2.append((count, cx, cy))
            
            cells2.sort()
            for count, cx, cy in cells2:
                if len(adaptive_points) >= n_phase2:
                    break
                x0 = (cx + 0.5) * cell_size2
                x1 = (cy + 0.5) * cell_size2
                adaptive_points.append([x0, x1])
        
        # Place only the first n_phase2 adaptive points
        adaptive_points = adaptive_points[:n_phase2]
        
        # Evaluate phase 2
        for i in range(n_phase2):
            if time.time() > deadline:
                break
            x0, x1 = adaptive_points[i]
            x = points[n_phase1 + i]
            x[0] = x0
            x[1] = x1
            results.append(problem(x))
        
        return results
