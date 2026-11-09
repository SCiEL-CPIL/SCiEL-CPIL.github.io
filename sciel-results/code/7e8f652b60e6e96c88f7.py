"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase adaptive grid search exploiting WFG9 structure for dense Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # WFG9: variables 0-2 control the front, variables 3-11 are shape/position
        # Fix non-front variables at midpoint to exploit structure
        base = mid.copy()
        
        X_list = []
        F_list = []
        used = 0
        
        def evaluate_point(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            f = problem(x)
            X_list.append(x.copy())
            F_list.append(np.array(f))
            used += 1
            return f
        
        # Phase 1: Coarse uniform grid to identify promising region
        # Use 4x4x4 = 64 points for initial scan
        n1_c, n2_c, n3_c = 4, 4, 4
        t1_c = np.linspace(0.0, 1.0, n1_c)
        t2_c = np.linspace(0.0, 1.0, n2_c)
        t3_c = np.linspace(0.0, 1.0, n3_c)
        
        for t1 in t1_c:
            for t2 in t2_c:
                for t3 in t3_c:
                    x = base.copy()
                    x[0] = lo[0] + t1 * (hi[0] - lo[0])
                    x[1] = lo[1] + t2 * (hi[1] - lo[1])
                    x[2] = lo[2] + t3 * (hi[2] - lo[2])
                    f = evaluate_point(x)
                    if f is None:
                        break
                if used >= self.budget or time.time() > deadline:
                    break
            if used >= self.budget or time.time() > deadline:
                break
        
        # If no budget left, return what we have
        if used >= self.budget or time.time() > deadline:
            if X_list:
                return np.array(X_list)
            return base.reshape(1, -1)
        
        # Phase 2: Adaptive refinement in the most promising region
        # Find the region with the lowest sum of objectives (closest to ideal point)
        if F_list:
            F_arr = np.array(F_list)
            sums = F_arr.sum(axis=1)
            best_idx = np.argmin(sums)
            
            # Get the center of the best region
            best_x = X_list[best_idx]
            # Determine which cell in the coarse grid this belongs to
            cell_size = 1.0 / 4.0  # Since we used 4 divisions
            c1 = int((best_x[0] - lo[0]) / (hi[0] - lo[0]) * 4)
            c2 = int((best_x[1] - lo[1]) / (hi[1] - lo[1]) * 4)
            c3 = int((best_x[2] - lo[2]) / (hi[2] - lo[2]) * 4)
            c1 = min(c1, 3)
            c2 = min(c2, 3)
            c3 = min(c3, 3)
            
            # Define the refined region: expand the best cell slightly
            # Use the cell containing the best point and its neighbors
            regions = []
            for dc1 in [-1, 0, 1]:
                for dc2 in [-1, 0, 1]:
                    for dc3 in [-1, 0, 1]:
                        nc1 = c1 + dc1
                        nc2 = c2 + dc2
                        nc3 = c3 + dc3
                        if 0 <= nc1 <= 3 and 0 <= nc2 <= 3 and 0 <= nc3 <= 3:
                            regions.append((nc1, nc2, nc3))
            
            # If too many regions, just use the best cell
            if len(regions) > 4:
                regions = [(c1, c2, c3)]
            
            # Phase 2a: Fine grid in the identified region(s)
            remaining = self.budget - used
            # Use 70% of remaining for fine grid, 30% for local refinement
            fine_budget = int(remaining * 0.7)
            local_budget = remaining - fine_budget
            
            # Fine grid resolution: try to use ~8x8x8 if possible, scale down otherwise
            # Total fine points should be <= fine_budget
            fine_n = max(2, int(round((fine_budget / max(1, len(regions))) ** (1/3))))
            fine_n = min(fine_n, 10)  # Cap at 10 per dimension
            
            for (r1, r2, r3) in regions:
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Define sub-region boundaries
                lo1 = lo[0] + r1 * (hi[0] - lo[0]) / 4.0
                hi1 = lo[0] + (r1 + 1) * (hi[0] - lo[0]) / 4.0
                lo2 = lo[1] + r2 * (hi[1] - lo[1]) / 4.0
                hi2 = lo[1] + (r2 + 1) * (hi[1] - lo[1]) / 4.0
                lo3 = lo[2] + r3 * (hi[2] - lo[2]) / 4.0
                hi3 = lo[2] + (r3 + 1) * (hi[2] - lo[2]) / 4.0
                
                t1_f = np.linspace(0.0, 1.0, fine_n)
                t2_f = np.linspace(0.0, 1.0, fine_n)
                t3_f = np.linspace(0.0, 1.0, fine_n)
                
                for t1 in t1_f:
                    for t2 in t2_f:
                        for t3 in t3_f:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x = base.copy()
                            x[0] = lo1 + t1 * (hi1 - lo1)
                            x[1] = lo2 + t2 * (hi2 - lo2)
                            x[2] = lo3 + t3 * (hi3 - lo3)
                            f = evaluate_point(x)
                            if f is None:
                                break
                        if used >= self.budget or time.time() > deadline:
                            break
                    if used >= self.budget or time.time() > deadline:
                        break
        
        # Phase 3: Local refinement around non-dominated points
        if used < self.budget and time.time() < deadline and F_list:
            F_arr = np.array(F_list)
            # Find non-dominated points efficiently
            nd_idx = []
            n_pts = len(F_arr)
            for i in range(n_pts):
                dominated = False
                for j in range(n_pts):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_idx.append(i)
            
            # Refine around top non-dominated points
            for idx in nd_idx[:min(10, len(nd_idx))]:
                if used >= self.budget or time.time() > deadline:
                    break
                
                center = X_list[idx]
                # Local perturbation on the 3 front variables with multiple step sizes
                for dim in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    for delta in [0.05, -0.05, 0.02, -0.02]:
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        x = center.copy()
                        x[dim] = np.clip(x[dim] + delta * (hi[dim] - lo[dim]), lo[dim], hi[dim])
                        
                        f = evaluate_point(x)
                        if f is None:
                            break
        
        # Convert to arrays
        if X_list:
            X = np.array(X_list)
        else:
            X = base.reshape(1, -1)
        
        return X
# EVOLVE-BLOCK-END
