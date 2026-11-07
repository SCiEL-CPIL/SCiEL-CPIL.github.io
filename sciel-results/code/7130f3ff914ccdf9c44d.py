"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic adaptive grid refinement multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        budget = self.budget
        dim = self.dim

        # Phase 1: Coarse grid evaluation
        # Use a grid of size 4 in each dimension -> 4^4 = 256 points
        # But we only have 200 budget, so use 3^4 = 81 points for coarse
        # Actually, let's be smarter. Let's use a grid that fits in budget.
        # 4^4 = 256 > 200. 3^4 = 81 <= 200.
        # Let's try 3^4 = 81 for the first pass.
        
        n_coarse = 3
        # Generate coarse grid points
        axes = [np.linspace(lo[i], hi[i], n_coarse) for i in range(dim)]
        grid = np.array(np.meshgrid(*axes, indexing='ij')).T.reshape(-1, dim)
        
        # Evaluate coarse grid
        F_coarse = np.array([problem(x) for x in grid])
        used = len(grid)
        
        # Keep track of all evaluated points and their objectives
        all_points = grid.copy()
        all_F = F_coarse.copy()
        
        # Phase 2: Adaptive refinement
        # We have remaining budget. We will refine cells that are non-dominated or promising.
        # A simple strategy: 
        # 1. Identify non-dominated points in the coarse grid.
        # 2. For each non-dominated point, subdivide its surrounding cell (or the cell it belongs to) 
        #    into smaller cells and evaluate new points.
        # 
        # A better approach for "adaptive grid refinement":
        # Maintain a list of cells (hyper-cubes) that are candidates for refinement.
        # Initially, the whole space is one cell? No, we already evaluated a grid.
        # Let's think of the coarse grid as dividing the space into (n_coarse-1)^dim cells.
        # Each cell has 2^dim corners. We evaluated the grid points (corners).
        # We can refine a cell by evaluating its center and/or midpoints of edges.
        
        # Let's use a priority queue of cells to refine.
        # Priority: based on the "promise" of the cell.
        # Since we don't have a model, we can use the best objective value in the cell's corners.
        # Or simply refine cells containing non-dominated points.
        
        # Let's define cells by their lower and upper bounds.
        # The coarse grid divides each dimension into n_coarse-1 intervals.
        # Number of cells = (n_coarse-1)^dim = 2^4 = 16 cells.
        
        # For each cell, we know the objective values at its 16 corners (from the coarse grid).
        # We can compute the "best" corner in each cell.
        # Then we prioritize cells with good best corners.
        
        # Let's create a list of cells to refine.
        # Each cell is defined by an index tuple (i0, i1, i2, i3) where ik in [0, n_coarse-2].
        # The cell spans [lo + i_k * span_k/(n_coarse-1), lo + (i_k+1) * span_k/(n_coarse-1)] in dimension k.
        
        cells = []
        for idx in np.ndindex(*(n_coarse - 1,) * dim):
            # Get the corners of this cell
            # Corner indices in the grid: for each dimension k, the corner is at idx[k] or idx[k]+1
            corner_indices = []
            for offset in np.ndindex(*(2,) * dim):
                ci = tuple(idx[k] + offset[k] for k in range(dim))
                # Flatten index
                flat = 0
                for k in range(dim - 1, -1, -1):
                    flat = flat * (n_coarse) + ci[k]
                corner_indices.append(flat)
            
            F_corners = F_coarse[corner_indices]
            # Find the best corner (using sum of objectives as a simple scalarization)
            best_corner_idx = np.argmin(np.sum(F_corners, axis=1))
            best_val = F_corners[best_corner_idx]
            
            # Check if this cell contains a non-dominated point
            # A point is non-dominated if no other point in the entire set dominates it.
            # For now, let's just use the best corner value as a priority.
            priority = np.sum(best_val)
            cells.append((priority, idx, best_val))
        
        # Sort cells by priority (lower is better)
        cells.sort(key=lambda x: x[0])
        
        # Now, iteratively refine the best cells.
        # Refinement strategy: subdivide a cell into 2^dim sub-cells and evaluate the center of the original cell?
        # Or evaluate new grid points within the cell.
        # Let's evaluate the center of the cell first. If it's good, subdivide further.
        
        # To keep it simple and deterministic:
        # For each cell in priority order, if we have budget left:
        # 1. Evaluate the center of the cell.
        # 2. If the center is non-dominated (or better than the best corner), subdivide the cell into 2^dim sub-cells
        #    and add them to the refinement queue with updated priorities.
        
        # However, managing a dynamic queue of sub-cells is complex.
        # Alternative: Multi-level grid.
        # Level 0: 3^4 grid (81 points). Used 81.
        # Level 1: 5^4 grid? 625 points. Too many.
        # We have 200 - 81 = 119 points left.
        
        # Let's try a different approach:
        # Use the remaining budget to do a local search around the non-dominated points found in the coarse grid.
        # Or, use a finer global grid but only evaluate a subset.
        
        # Actually, the directive says: "iteratively subdivides the 4D parameter space into hypercubes, 
        # evaluating only the centroids and retaining the finest grid cells whose objective values are non-dominated"
        
        # Let's implement a simple recursive subdivision.
        # Start with the whole domain as one cell.
        # But we already have a coarse grid. Let's use that as the base.
        
        # Let's restart the refinement logic.
        # We have 119 evaluations left.
        # We will refine the 16 coarse cells.
        # For each cell, we can evaluate its center. That's 16 evaluations.
        # Then we can subdivide the most promising cells.
        
        # Let's do this:
        # 1. Evaluate centers of all 16 coarse cells. (16 evals)
        # 2. Combine with coarse grid points.
        # 3. Identify non-dominated points.
        # 4. For each non-dominated point, it lies in some coarse cell.
        # 5. Subdivide that coarse cell into 16 sub-cells (2x2x2x2).
        # 6. Evaluate the centers of these sub-cells.
        # 7. Repeat until budget is exhausted.
        
        # This is a bit complex to implement correctly. Let's simplify.
        
        # Alternative Simple Adaptive Grid:
        # Use a fixed sequence of grid resolutions that fit in the budget.
        # Res 3: 81 points.
        # Res 5: 625 points (too many).
        # Res 4: 256 points (too many for 200 budget total, but 256-81=175 > 119).
        
        # Let's just use a random search but with a deterministic seed? No, the directive says deterministic grid.
        
        # Let's try to fit a 4^4 grid by skipping some points?
        # Or use a 3^4 grid and then a local refinement.
        
        # Given the complexity, let's stick to a simpler deterministic strategy:
        # 1. Evaluate a 3^4 grid (81 points).
        # 2. Find non-dominated points.
        # 3. For each non-dominated point, perform a local grid search in a small neighborhood.
        #    Neighborhood: +/- 0.1 * span in each dimension.
        #    Grid: 3^4 = 81 points per neighborhood? Too many.
        #    Grid: 2^4 = 16 points per neighborhood (midpoints of edges).
        
        # Let's do this:
        # After the 3^4 grid, identify the top K non-dominated points.
        # For each, evaluate 15 new points around it (midpoints of the 2x2x2x2 sub-cells centered at it? No, just a small grid).
        
        # Actually, let's just use a 4^4 grid but evaluate it in a specific order and stop when budget runs out.
        # A 4^4 grid has 256 points. We can evaluate 200 of them.
        # Which 200? The ones closest to the center? Or a space-filling curve?
        
        # Let's use a Halton sequence or similar quasi-random sequence, but deterministic.
        # The directive says "adaptive grid refinement".
        
        # Let's implement a simple version of the directive:
        # 1. Start with 1 cell (the whole domain).
        # 2. Evaluate the center.
        # 3. Subdivide into 16 cells.
        # 4. Evaluate centers of all 16 cells. (Total 17)
        # 5. Keep the best 8 cells (based on objective value).
        # 6. Subdivide those 8 into 16 sub-cells each (128 cells).
        # 7. Evaluate centers of these 128 cells? (17 + 128 = 145).
        # 8. Keep the best 16 cells.
        # 9. Subdivide those 16 into 16 sub-cells each (256 cells).
        # 10. We have 200 - 145 = 55 evaluations left.
        # 11. Evaluate 55 centers from the 256 cells, prioritizing those with best parent cells.
        
        # This seems like a solid plan.
        
        # Reset and implement this plan.
        
        # Step 1: Whole domain center
        center0 = (lo + hi) / 2
        F0 = np.array([problem(center0)])
        used = 1
        all_points = [center0]
        all_F = F0
        
        # Current cells: list of (center, F, bounds)
        # We will maintain a list of active cells to refine.
        # Each cell has a center, an F value, and bounds (lo, hi).
        
        # Level 1: Subdivide the whole domain into 16 cells.
        # The 16 cells are defined by splitting each dimension at the midpoint.
        # Centers are the midpoints of each sub-interval.
        
        def subdivide_cell(cell_lo, cell_hi):
            """Yield the 16 sub-cells of a 4D cell."""
            mids = (cell_lo + cell_hi) / 2
            # Generate all 16 combinations of low/high for each dimension
            for combo in np.ndindex(*(2,) * dim):
                new_lo = np.array([cell_lo[i] if combo[i] == 0 else mids[i] for i in range(dim)])
                new_hi = np.array([mids[i] if combo[i] == 0 else cell_hi[i] for i in range(dim)])
                new_center = (new_lo + new_hi) / 2
                yield new_lo, new_hi, new_center
        
        # Evaluate Level 1 cells
        level1_cells = []
        for new_lo, new_hi, new_center in subdivide_cell(lo, hi):
            if used >= budget:
                break
            F_new = np.array([problem(new_center)])
            used += 1
            all_points.append(new_center)
            all_F = np.vstack([all_F, F_new])
            level1_cells.append((new_lo, new_hi, new_center, F_new[0]))
        
        # Sort Level 1 cells by sum of objectives
        level1_cells.sort(key=lambda x: np.sum(x[3]))
        
        # Keep the best 8 cells
        best_cells = level1_cells[:8]
        
        # Level 2: Subdivide the best 8 cells
        level2_cells = []
        for cell in best_cells:
            if used >= budget:
                break
            for new_lo, new_hi, new_center in subdivide_cell(cell[0], cell[1]):
                if used >= budget:
                    break
                F_new = np.array([problem(new_center)])
                used += 1
                all_points.append(new_center)
                all_F = np.vstack([all_F, F_new])
                level2_cells.append((new_lo, new_hi, new_center, F_new[0]))
        
        # Sort Level 2 cells
        level2_cells.sort(key=lambda x: np.sum(x[3]))
        
        # Keep the best 16 cells
        best_cells = level2_cells[:16]
        
        # Level 3: Subdivide the best 16 cells
        level3_cells = []
        for cell in best_cells:
            if used >= budget:
                break
            for new_lo, new_hi, new_center in subdivide_cell(cell[0], cell[1]):
                if used >= budget:
                    break
                F_new = np.array([problem(new_center)])
                used += 1
                all_points.append(new_center)
                all_F = np.vstack([all_F, F_new])
                level3_cells.append((new_lo, new_hi, new_center, F_new[0]))
        
        # If we still have budget, evaluate more points from the best level 3 cells
        level3_cells.sort(key=lambda x: np.sum(x[3]))
        for cell in level3_cells:
            if used >= budget:
                break
            # Subdivide this cell again?
            # We are running out of budget, so let's just stop.
            pass
        
        return all_points
# EVOLVE-BLOCK-END
