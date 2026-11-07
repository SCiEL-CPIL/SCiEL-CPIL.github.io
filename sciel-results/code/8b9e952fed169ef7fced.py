"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Grid Refinement Multi-Objective Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We maintain a list of active "cells" (hyper-rectangles) to be evaluated or refined.
        # Each cell is defined by its low and high bounds.
        # We also keep track of evaluated points to avoid duplicates and for final return.
        
        active_cells = [(lo, hi)]
        evaluated_points = []
        used = 0
        
        # To maximize hypervolume, we want to spread points out.
        # A pure grid approach might be too rigid if the front is curved.
        # An adaptive approach: 
        # 1. Start with 1 cell (the whole space).
        # 2. Evaluate the center of the most "promising" or largest cell?
        #    Actually, for global coverage, we should expand the grid uniformly first.
        
        # Let's use a strategy similar to "Halving" but adaptive.
        # Initialize with a coarse grid? 
        # With 200 evaluations and 5 dims, a full grid is impossible (2^5=32, 3^5=243).
        # 200 is close to 3^5. 
        # However, a static 3x3x3x3x3 grid uses 243 evals, which is > 200.
        # A 2x2x2x2x2 grid uses 32 evals.
        
        # Strategy:
        # 1. Generate a base grid of size 2^dim (32 points). Evaluate all.
        # 2. For each subsequent evaluation, pick the cell with the largest "potential" 
        #    (e.g., largest volume, or largest distance to nearest evaluated point in objective space) 
        #    and split it, evaluating the center of the new sub-cells?
        #    Actually, simpler: maintain a list of candidate points to evaluate.
        #    Start with the 32 grid points.
        #    After evaluating them, for each cell in the grid, we can add midpoints of its edges?
        
        # Let's try a "Space Filling" approach using a priority queue of cells.
        # But we need to be careful about the number of points.
        
        # Alternative: Latin Hypercube Sampling (LHS) is good for coverage but doesn't adapt.
        # Deterministic Grid with Adaptive Refinement:
        
        # Step 1: Coarse Grid (2 per dim -> 32 points)
        # Step 2: Refine the cells that contain non-dominated points? Or just refine uniformly?
        
        # Let's stick to a robust, simple structure:
        # 1. Evaluate a 2^5 grid (32 points).
        # 2. For each of the 32 cells, we have 1 point (the center? No, the grid points are vertices).
        #    Let's define cells by their vertices.
        #    Actually, let's just generate a list of points to evaluate.
        
        # Points to evaluate:
        # 1. All vertices of the 2^5 grid (32 points).
        # 2. The center of the whole domain (1 point).
        # 3. Midpoints of edges?
        
        # Let's use a recursive subdivision.
        # Queue of cells: (low, high, depth)
        # We evaluate the center of a cell.
        # If we have budget left, we split the cell with the largest volume into 2^dim sub-cells?
        # No, that explodes.
        
        # Better: 
        # 1. Evaluate centers of 2^dim cells (32 points).
        # 2. Identify the non-dominated set.
        # 3. For each non-dominated point, its "cell" in the coarse grid is promising.
        #    Subdivide that cell into 2^dim sub-cells and evaluate their centers.
        #    This focuses search on promising regions.
        
        # Let's implement this.
        
        # Initial coarse grid: 2 intervals per dimension.
        # Centers of these 2^5 = 32 cells.
        
        def get_centers(lo, hi, n_splits=2):
            """Generate centers of a grid of n_splits^dim cells."""
            # lo, hi are arrays of length dim
            # We create 1D points for each dim
            points_1d = []
            for i in range(self.dim):
                # n_splits intervals
                # Centers are at (i + 0.5) / n_splits * (hi-lo) + lo
                centers = []
                for j in range(n_splits):
                    center = lo[i] + (j + 0.5) / n_splits * (hi[i] - lo[i])
                    centers.append(center)
                points_1d.append(centers)
            
            # Cartesian product
            grid_points = np.array(np.meshgrid(*points_1d)).T.reshape(-1, self.dim)
            return grid_points
        
        # 1. Evaluate coarse grid
        coarse_points = get_centers(lo, hi, n_splits=2)
        
        # We need to track which cells these points belong to for refinement.
        # A point p in coarse grid corresponds to cell index tuple (c0, c1, ..., c4) where ci in {0, 1}
        # We can store the cell bounds for each evaluated point.
        
        # Structure: list of dicts or tuples: (point, f_value, cell_lo, cell_hi)
        # Let's keep it simple:
        # active_cells: list of (lo, hi)
        # We will evaluate the center of each active cell.
        # If budget allows, we pick the "best" active cell (based on its center's objective value?) 
        # and split it.
        
        # Actually, a simpler and very effective method for small budgets is:
        # 1. Evaluate 32 coarse centers.
        # 2. Sort them by a scalarization (e.g., sum of normalized objectives) or just keep all.
        # 3. For the next 168 evaluations, we perform adaptive refinement.
        #    We maintain a priority queue of cells.
        #    Priority = Volume of the cell? Or "Potential" = distance to nearest non-dominated point?
        #    Let's use Volume. Larger cells have more unexplored space.
        #    When we pick a cell, we evaluate its center.
        #    Then we split it into 2^dim sub-cells and add them to the queue.
        #    Wait, if we split into 2^5=32 sub-cells, we add 32 cells to the queue for 1 evaluation.
        #    This is inefficient if we don't evaluate the sub-cells.
        
        # Revised Strategy:
        # 1. Start with 1 cell (the whole space).
        # 2. Loop while budget > 0:
        #    a. Pick the cell with the largest volume (or some heuristic).
        #    b. Evaluate its center.
        #    c. Remove this cell from the queue.
        #    d. Split this cell into 2^dim sub-cells.
        #    e. Add the sub-cells to the queue.
        #    f. Decrement budget.
        
        # This is a standard adaptive grid search (like Halving or VLSM).
        # It guarantees that the maximum gap between evaluated points shrinks as 1/sqrt(N) in each dimension?
        # Actually, it creates a hierarchical grid.
        
        # Let's implement this.
        
        # Queue: list of (volume, lo, hi)
        # Using a list is fine for 200 items, but a heap is better.
        # We want to pop the LARGEST volume, so we use a min-heap with negative volume.
        
        import heapq
        
        # Initial cell
        initial_vol = np.prod(hi - lo)
        # Heap element: (-volume, counter, lo, hi)
        # Counter to avoid comparing arrays if volumes are equal
        counter = 0
        heap = []
        heapq.heappush(heap, (-initial_vol, counter, lo, hi))
        counter += 1
        
        all_points = []
        all_f = []
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Pop the cell with the largest volume
            neg_vol, _, cell_lo, cell_hi = heapq.heappop(heap)
            
            # Calculate center
            center = (cell_lo + cell_hi) / 2.0
            
            # Evaluate
            f_val = problem(center)
            used += 1
            
            all_points.append(center)
            all_f.append(f_val)
            
            # Check if we have budget to split
            if used < self.budget:
                # Split into 2^dim sub-cells
                # For each dimension, we split the interval [lo, hi] into [lo, mid] and [mid, hi]
                # We need to generate all 2^dim combinations.
                
                # To avoid creating 32 new arrays every time (which is fine for 200 iterations),
                # we can iterate through the binary choices.
                
                # Optimization: Only split if the cell volume is significant?
                # No, just split.
                
                # Generate sub-cells
                # dim=5, so 32 sub-cells.
                # We can use itertools.product
                import itertools
                
                sub_cells = []
                for bits in itertools.product([0, 1], repeat=self.dim):
                    new_lo = cell_lo.copy()
                    new_hi = cell_hi.copy()
                    
                    for i in range(self.dim):
                        mid = (cell_lo[i] + cell_hi[i]) / 2.0
                        if bits[i] == 0:
                            new_hi[i] = mid
                        else:
                            new_lo[i] = mid
                            
                    # Calculate volume
                    vol = np.prod(new_hi - new_lo)
                    if vol > 0:
                        heapq.heappush(heap, (-vol, counter, new_lo, new_hi))
                        counter += 1

        # Return the best set of points?
        # The task says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # Wait, "return value of __call__ is ignored".
        # "More than 200 evaluations... makes the whole result invalid"
        # "Scoring... per seed: exact hypervolume of all recorded vectors..."
        
        # So the evaluator records the objective vectors returned by `problem(x)`?
        # "Every returned objective vector is recorded"
        # This implies the evaluator wraps `problem` or `__call__` returns something?
        # "the return value of __call__ is ignored"
        # "Every returned objective vector is recorded" -> This likely means the evaluator intercepts the calls to `problem`?
        # Or does `__call__` return the list of points?
        # "Write class Optimizer ... and __call__(self, problem, bounds) ... minimizing all 3 objectives"
        # "Every returned objective vector is recorded"
        # Usually in these tasks, the evaluator monkey-patches `problem` to record calls.
        # So I just need to make sure I call `problem` at most 200 times.
        # The return value of `__call__` doesn't matter for scoring, only the recorded evaluations.
        
        return all_points
# EVOLVE-BLOCK-END
