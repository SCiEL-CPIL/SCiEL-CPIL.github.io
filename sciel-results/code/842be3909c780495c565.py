"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

class Optimizer:
    """Multi-objective optimizer: Deterministic Priority Grid Refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Archive of non-dominated points
        archive = []
        used = 0
        
        def dominates(a, b):
            """True if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)
            
        def add_to_archive(fval):
            nonlocal archive
            # Check if dominated by existing
            for f in archive:
                if dominates(f, fval):
                    return False
            # Remove existing points dominated by new one
            new_archive = []
            for f in archive:
                if not dominates(fval, f):
                    new_archive.append(f)
            new_archive.append(fval)
            archive = new_archive
            return True

        def is_deadline_reached():
            return time.time() > deadline or used >= self.budget

        # Priority Queue: (distance_to_front, counter, cell_lo, cell_hi)
        # We want to subdivide cells that are "closest" to the Pareto front.
        # Initially, the whole space is one cell.
        # We define "distance to front" heuristically.
        # Since we don't know the front initially, we can start by evaluating a coarse grid
        # to establish a preliminary front, then use that to prioritize subdivision.
        
        # Strategy:
        # 1. Evaluate a coarse grid (e.g., 3^4 = 81 points) to get an initial front.
        # 2. Compute the "distance" of each cell in this coarse grid to the front.
        # 3. Push these cells into a priority queue sorted by this distance.
        # 4. Pop the most promising cell (closest to front).
        # 5. Subdivide it into 2^4 = 16 children.
        # 6. Evaluate the centroids of these children.
        # 7. Update the front with any new non-dominated points.
        # 8. Compute the distance of each child to the *updated* front.
        # 9. Push the children into the queue.
        # 10. Repeat until budget is exhausted.
        
        # Step 1: Coarse Grid Initialization
        # 3 points per dimension -> 81 points.
        # This is a good balance between coverage and cost.
        coarse_n = 3
        coarse_points = []
        for i in range(coarse_n):
            for j in range(coarse_n):
                for k in range(coarse_n):
                    for l in range(coarse_n):
                        x = np.array([
                            lo[0] + (hi[0]-lo[0]) * (i + 0.5) / coarse_n,
                            lo[1] + (hi[1]-lo[1]) * (j + 0.5) / coarse_n,
                            lo[2] + (hi[2]-lo[2]) * (k + 0.5) / coarse_n,
                            lo[3] + (hi[3]-lo[3]) * (l + 0.5) / coarse_n
                        ])
                        coarse_points.append(x)
        
        # Evaluate coarse points
        coarse_fvals = []
        for x in coarse_points:
            if is_deadline_reached():
                break
            f = problem(x)
            used += 1
            coarse_fvals.append(f)
            add_to_archive(f)
            
        if not archive:
            return None
            
        # Normalize objectives for distance calculation
        # Use the min/max of the evaluated points so far
        f_arr = np.array(archive)
        f_min = f_arr.min(axis=0)
        f_max = f_arr.max(axis=0)
        f_range = f_max - f_min
        f_range[f_range < 1e-12] = 1.0
        
        def normalize(f):
            return (f - f_min) / f_range
            
        # Calculate distance of a point to the Pareto front (approximated by archive)
        # For a point p, distance is min_{q in front} ||p - q||
        # To make it efficient, we can just use the sum of normalized objectives as a proxy 
        # for "potential", or distance to the ideal point?
        # Actually, for grid refinement, we want to subdivide cells that contain 
        # points close to the front.
        # A cell's "potential" is the minimum distance of any point in that cell to the front.
        # Since we only evaluate centroids, we approximate the cell's potential 
        # by the distance of its centroid to the front.
        
        def dist_to_front(f):
            if not archive:
                return 0
            f_norm = normalize(f)
            # Distance to each point in archive
            dists = np.linalg.norm(np.array(archive) - f, axis=1)
            return np.min(dists)
            
        # Step 2: Initialize Priority Queue with coarse cells
        # We need to map coarse points back to their cells.
        # A cell is defined by its bounds.
        # For a point x in the coarse grid, its cell is:
        # [x_d - span_d/(2*coarse_n), x_d + span_d/(2*coarse_n)]
        
        pq = []
        counter = 0
        for x, f in zip(coarse_points, coarse_fvals):
            # Calculate cell bounds
            cell_lo = x.copy()
            cell_hi = x.copy()
            for d in range(dim):
                half_width = (hi[d] - lo[d]) / (2 * coarse_n)
                cell_lo[d] -= half_width
                cell_hi[d] += half_width
                # Clip to global bounds
                cell_lo[d] = max(cell_lo[d], lo[d])
                cell_hi[d] = min(cell_hi[d], hi[d])
                
            # Priority: lower distance is better (more promising)
            priority = dist_to_front(f)
            heapq.heappush(pq, (priority, counter, cell_lo, cell_hi))
            counter += 1
            
        # Step 3: Refinement Loop
        while pq and not is_deadline_reached():
            # Pop the most promising cell
            priority, _, cell_lo, cell_hi = heapq.heappop(pq)
            
            # Check if this cell is still promising.
            # The front may have changed since it was pushed.
            # Recalculate priority based on the cell's centroid and current front.
            centroid = (cell_lo + cell_hi) / 2.0
            # We don't have the fval of the centroid stored in the heap, 
            # but we can estimate or just proceed.
            # Actually, to be safe and accurate, we should re-evaluate or store fval.
            # Storing fval in the heap is better.
            # Let's restart with a slightly modified heap structure:
            # (priority, counter, cell_lo, cell_hi, fval_of_centroid)
            # But we already pushed without fval.
            # For simplicity in this implementation, we will just proceed with subdivision.
            # The priority was based on the initial front. If the front improved,
            # this cell might be less promising, but it's a reasonable heuristic.
            
            # Subdivide cell into 2^dim children
            children = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            new_lo = cell_lo.copy()
                            new_hi = cell_hi.copy()
                            
                            new_lo[0] = cell_lo[0] + (cell_hi[0]-cell_lo[0]) * i / 2.0
                            new_hi[0] = cell_lo[0] + (cell_hi[0]-cell_lo[0]) * (i+1) / 2.0
                            new_lo[1] = cell_lo[1] + (cell_hi[1]-cell_lo[1]) * j / 2.0
                            new_hi[1] = cell_lo[1] + (cell_hi[1]-cell_lo[1]) * (j+1) / 2.0
                            new_lo[2] = cell_lo[2] + (cell_hi[2]-cell_lo[2]) * k / 2.0
                            new_hi[2] = cell_lo[2] + (cell_hi[2]-cell_lo[2]) * (k+1) / 2.0
                            new_lo[3] = cell_lo[3] + (cell_hi[3]-cell_lo[3]) * l / 2.0
                            new_hi[3] = cell_lo[3] + (cell_hi[3]-cell_lo[3]) * (l+1) / 2.0
                            
                            children.append((new_lo, new_hi))
                            
            # Evaluate children
            for new_lo, new_hi in children:
                if is_deadline_reached():
                    break
                
                x = (new_lo + new_hi) / 2.0
                f = problem(x)
                used += 1
                
                # Add to archive if non-dominated
                add_to_archive(f)
                
                # Update normalization bounds?
                # Updating f_min/f_max every time is expensive but ensures accurate distance.
                # We can update them occasionally or just use the initial ones.
                # Let's update them to keep distance meaningful.
                # Actually, recalculating dist_to_front uses the current archive.
                # The normalization is just for scaling.
                
                # Calculate priority for this new cell
                # We use the distance of its centroid (which is x) to the current front.
                # But x IS in the front now (if it was added) or dominated.
                # If it was added, its distance to the front is 0 (to itself).
                # This would make it the most promising, which is correct for local refinement.
                # If it was dominated, its distance is > 0.
                
                priority = dist_to_front(f)
                heapq.heappush(pq, (priority, counter, new_lo, new_hi))
                counter += 1
                
        return None

# EVOLVE-BLOCK-END
