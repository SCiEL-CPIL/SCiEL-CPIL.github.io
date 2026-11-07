import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0
        eval_count = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f, eval_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Grid Scan to locate approximate region
        # Use a deterministic grid with 3 points per dimension (27 total in 5D)
        grid_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        for m in range(3):
                            x = np.array([i*5-5, j*5-5, k*5-5, l*5-5, m*5-5], dtype=float)
                            grid_points.append(x)
        
        # Evaluate all grid points
        for x in grid_points:
            if calls_made >= budget:
                break
            if (time.time() - start_time) > deadline:
                break
            evaluate(x)
        
        # Phase 2: Adaptive Bisection Search
        # We maintain a priority queue of regions to explore
        # Each region is defined by lower and upper bounds
        # We prioritize regions where function values vary significantly
        
        from heapq import heappush, heappop, heapify
        
        # Region: (priority, center, radius, f_center)
        # Priority = max_f_in_region - min_f_in_region (higher = more interesting)
        # We'll use a simpler approach: track regions and their best/worst values
        
        # Initialize with the full domain
        full_lower = np.full(n, -5.0)
        full_upper = np.full(n, 5.0)
        full_center = np.zeros(n)
        
        # Evaluate center
        f_center = evaluate(full_center)
        
        # Create initial region
        # Region representation: [priority, id, lower, upper, center, f_center, best_f_in_region]
        region_id = 0
        regions = []
        
        def create_region(lower, upper, center, f_center, best_f_in_region=None):
            nonlocal region_id
            if best_f_in_region is None:
                best_f_in_region = f_center
            region = {
                'id': region_id,
                'lower': lower.copy(),
                'upper': upper.copy(),
                'center': center.copy(),
                'f_center': f_center,
                'best_f': best_f_in_region,
                'worst_f': f_center,
                'eval_count': 1
            }
            region_id += 1
            return region
        
        initial_region = create_region(full_lower, full_upper, full_center, f_center)
        regions.append(initial_region)
        
        # Priority function: we want to explore regions with high variance first
        # But we don't know the variance yet. Let's use a different strategy:
        # Always bisect the region containing the current best point, and also
        # explore neighboring regions to find the discontinuity
        
        # Strategy: 
        # 1. Find which coarse grid cell contains best_x
        # 2. Subdivide that cell into 2^n subcells
        # 3. Evaluate centers of subcells
        # 4. Recursively subdivide the best subcell
        
        # First, find the coarse grid cell containing best_x
        def find_coarse_cell(x):
            cell = []
            for d in range(n):
                if x[d] < -2.5:
                    cell.append(0)
                elif x[d] < 2.5:
                    cell.append(1)
                else:
                    cell.append(2)
            return cell
        
        best_cell = find_coarse_cell(best_x)
        
        # Now subdivide this cell
        # Cell bounds:
        cell_lower = np.array([c * 5 - 5 for c in best_cell])
        cell_upper = cell_lower + 5.0
        
        # Generate all 2^n subcell centers
        subcell_centers = []
        for mask in range(2 ** n):
            center = cell_lower.copy()
            for d in range(n):
                if mask & (1 << d):
                    center[d] += 2.5
                else:
                    center[d] += 0.0
            # Actually, let's do this properly:
            # Each dimension is split into 2 halves
            center = cell_lower.copy()
            for d in range(n):
                if mask & (1 << d):
                    center[d] = cell_lower[d] + 2.5
                else:
                    center[d] = cell_lower[d] + 0.0
            # Wait, that's not right either. Let me think...
            # If cell_lower[d] = -5, then the two halves are [-5, -2.5] and [-2.5, 0]
            # Centers: -3.75 and -1.25
            # So: if bit is 0, center = lower + 1.25; if bit is 1, center = lower + 3.75
            
            center = cell_lower.copy()
            for d in range(n):
                if mask & (1 << d):
                    center[d] = cell_lower[d] + 3.75
                else:
                    center[d] = cell_lower[d] + 1.25
            subcell_centers.append(center)
        
        # Evaluate all subcell centers
        subcell_values = []
        for center in subcell_centers:
            if calls_made >= budget:
                break
            if (time.time() - start_time) > deadline:
                break
            f_val = evaluate(center)
            subcell_values.append((center, f_val))
        
        # Find the best subcell
        if subcell_values:
            best_subcell = min(subcell_values, key=lambda x: x[1])
            best_subcell_center = best_subcell[0]
            best_subcell_f = best_subcell[1]
            
            # Determine which subcell this is (for further subdivision)
            # We need to track the subcell index
            # Let's reorganize: keep track of subcells with their indices
            
            # Actually, let's use a different approach:
            # Maintain a list of "active" regions, each with bounds
            # Start with the coarse cell, then subdivide
            
            active_regions = []
            
            # Create the coarse cell region
            coarse_region = {
                'lower': cell_lower.copy(),
                'upper': cell_upper.copy(),
                'center': (cell_lower + cell_upper) / 2.0,
                'f_center': f_center,
                'depth': 0
            }
            active_regions.append(coarse_region)
            
            # Now iteratively subdivide the most promising region
            max_depth = 8  # Maximum subdivision depth
            
            while calls_made < budget and (time.time() - start_time) < deadline:
                if not active_regions:
                    break
                
                # Find the region with the best f_center
                best_region_idx = 0
                best_f = np.inf
                for i, region in enumerate(active_regions):
                    if region['f_center'] < best_f:
                        best_f = region['f_center']
                        best_region_idx = i
                
                region = active_regions[best_region_idx]
                
                if region['depth'] >= max_depth:
                    # Remove this region, it's fully subdivided
                    active_regions.pop(best_region_idx)
                    continue
                
                # Subdivide this region into 2^n subregions
                region_lower = region['lower']
                region_upper = region['upper']
                region_center = region['center']
                region_depth = region['depth']
                
                # Remove the current region
                active_regions.pop(best_region_idx)
                
                # Generate 2^n subregions
                new_regions = []
                for mask in range(2 ** n):
                    sub_lower = region_lower.copy()
                    sub_upper = region_upper.copy()
                    sub_center = np.zeros(n)
                    
                    half_width = (region_upper - region_lower) / 2.0
                    
                    for d in range(n):
                        if mask & (1 << d):
                            # Upper half
                            sub_lower[d] = region_lower[d] + half_width[d]
                            sub_center[d] = region_lower[d] + half_width[d] + half_width[d] / 2
                        else:
                            # Lower half
                            sub_upper[d] = region_lower[d] + half_width[d]
                            sub_center[d] = region_lower[d] + half_width[d] / 2
                    
                    # Evaluate the center of this subregion
                    if calls_made >= budget:
                        break
                    if (time.time() - start_time) > deadline:
                        break
                    
                    f_sub = evaluate(sub_center)
                    
                    sub_region = {
                        'lower': sub_lower,
                        'upper': sub_upper,
                        'center': sub_center,
                        'f_center': f_sub,
                        'depth': region_depth + 1
                    }
                    new_regions.append(sub_region)
                
                # Add new regions to active list
                active_regions.extend(new_regions)
                
                # If we've exhausted the budget or time, break
                if calls_made >= budget or (time.time() - start_time) > deadline:
                    break
        
        # Phase 3: Fine-tuning around the best point found
        if calls_made < budget and (time.time() - start_time) < deadline:
            # Use a local search with shrinking step sizes
            step_sizes = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
            
            current_x = best_x.copy()
            current_f = best_f
            
            for step in step_sizes:
                if calls_made >= budget or (time.time() - start_time) > deadline:
                    break
                
                improved = True
                max_iter = 10
                iter_count = 0
                
                while improved and calls_made < budget and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    
                    for d_idx in range(n):
                        if calls_made >= budget or (time.time() - start_time) > deadline:
                            break
                        
                        # Try positive direction
                        x_pos = current_x.copy()
                        x_pos[d_idx] += step
                        x_pos = np.clip(x_pos, -5, 5)
                        f_pos = func(x_pos)
                        calls_made += 1
                        
                        if f_pos < current_f:
                            current_x = x_pos
                            current_f = f_pos
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                            improved = True
                            continue
                        
                        # Try negative direction
                        x_neg = current_x.copy()
                        x_neg[d_idx] -= step
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = func(x_neg)
                        calls_made += 1
                        
                        if f_neg < current_f:
                            current_x = x_neg
                            current_f = f_neg
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                            improved = True
                            continue
        
        return best_x, best_f
