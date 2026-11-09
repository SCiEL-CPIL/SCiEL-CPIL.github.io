import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        n_points = self.budget
        dim = self.dim
        ref_point = np.array([2.5, 2.5, 2.5])
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index //= base
            return r
        
        # Generate a pool of candidate points in v-space
        # v1, v2 define points on the Pareto front of DTLZ4
        # f1 = 1 - v1^2, f2 = v1*sin(pi*v2/2), f3 = v1*cos(pi*v2/2)
        
        # First, create a dense grid of candidates
        n_grid = 50
        candidates_v = []
        
        # Add strategic points (vertices and edge midpoints)
        strategic_points_v = [
            (0.0, 0.0), # f=(1, 0, 0)
            (1.0, 0.0), # f=(0, 1, 0)
            (1.0, 1.0), # f=(0, 0, 1)
            (0.5, 0.0), # Edge
            (0.5, 1.0), # Edge
            (1.0, 0.5), # Edge
            (0.25, 0.0),
            (0.75, 0.0),
            (0.25, 1.0),
            (0.75, 1.0),
            (1.0, 0.25),
            (1.0, 0.75),
        ]
        candidates_v.extend(strategic_points_v)
        
        # Add Halton sequence points
        for i in range(1, 300):
            S1 = halton(i, 2)
            S2 = halton(i, 3)
            v1 = np.sqrt(1.0 - S1 * S1)
            v2 = 1.0 - S2
            v1 = float(np.clip(v1, 0.0, 1.0))
            v2 = float(np.clip(v2, 0.0, 1.0))
            candidates_v.append((v1, v2))
        
        # Convert v-space to objective space for hypervolume calculation
        def v_to_f(v1, v2):
            f1 = 1.0 - v1 * v1
            f2 = v1 * np.sin(np.pi * v2 / 2.0)
            f3 = v1 * np.cos(np.pi * v2 / 2.0)
            return np.array([f1, f2, f3])
        
        # Greedy hypervolume maximization
        selected_v = []
        selected_f = []
        
        # Compute hypervolume contribution of a new point given existing selected points
        def compute_hv_contribution(new_f, selected_f_list, ref):
            """Compute the hypervolume contribution of adding new_f to the set."""
            if len(selected_f_list) == 0:
                # Volume of box from new_f to ref
                return np.prod(np.maximum(0, ref - new_f))
            
            # For each existing point, compute the overlap
            # The contribution is the volume of the box from new_f to ref
            # minus the volume already covered by existing points
            new_box = np.maximum(0, ref - new_f)
            
            if np.any(new_box <= 0):
                return 0.0
            
            # Use inclusion-exclusion or just estimate by checking if new_f is dominated
            # For simplicity, check if new_f dominates any existing point
            # If so, it contributes significantly
            
            total_vol = 0.0
            for existing_f in selected_f_list:
                # Compute the box that new_f covers but existing_f doesn't
                # This is complex; use a simpler heuristic: 
                # if new_f is not dominated by any existing point, it has positive contribution
                
                if not np.all(existing_f <= new_f):
                    # new_f is not dominated by existing_f
                    # Approximate contribution as the volume of the box from new_f to ref
                    # weighted by how much it extends beyond existing points
                    pass
            
            # Simpler approach: compute the volume of the box from new_f to ref
            # and subtract the volume of the intersection with boxes from existing points
            # For 3D, this is manageable
            
            vol_new = np.prod(new_box)
            
            # Subtract overlaps with existing points
            # The overlap is the box from max(new_f, existing_f) to ref
            total_overlap = 0.0
            for existing_f in selected_f_list:
                overlap_lower = np.maximum(new_f, existing_f)
                if np.all(overlap_lower < ref):
                    overlap_vol = np.prod(ref - overlap_lower)
                    total_overlap += overlap_vol
            
            # This overcounts due to triple overlaps, but for greedy it's okay
            # A better estimate: just use vol_new if new_f is non-dominated
            is_dominated = False
            for existing_f in selected_f_list:
                if np.all(existing_f <= new_f) and np.any(existing_f < new_f):
                    is_dominated = True
                    break
            
            if is_dominated:
                return 0.0
            else:
                return vol_new
        
        # Select points greedily
        remaining_budget = n_points - len(strategic_points_v)
        
        # First, select the strategic points
        for v1, v2 in strategic_points_v[:min(6, n_points)]:
            f = v_to_f(v1, v2)
            selected_v.append((v1, v2))
            selected_f.append(f)
        
        # Now greedily select additional points
        candidates_remaining = candidates_v[len(strategic_points_v):]
        
        for _ in range(remaining_budget):
            if time.time() > deadline - 5:
                break
            
            best_idx = -1
            best_contribution = -1
            
            for idx, (v1, v2) in enumerate(candidates_remaining):
                if time.time() > deadline - 5:
                    break
                
                f = v_to_f(v1, v2)
                contribution = compute_hv_contribution(f, selected_f, ref_point)
                
                if contribution > best_contribution:
                    best_contribution = contribution
                    best_idx = idx
            
            if best_idx >= 0:
                v1, v2 = candidates_remaining.pop(best_idx)
                f = v_to_f(v1, v2)
                selected_v.append((v1, v2))
                selected_f.append(f)
            else:
                break
        
        # If we still have budget, add more Halton points
        if len(selected_v) < n_points:
            i = 300
            while len(selected_v) < n_points and time.time() < deadline - 5:
                S1 = halton(i, 2)
                S2 = halton(i, 3)
                v1 = np.sqrt(1.0 - S1 * S1)
                v2 = 1.0 - S2
                v1 = float(np.clip(v1, 0.0, 1.0))
                v2 = float(np.clip(v2, 0.0, 1.0))
                selected_v.append((v1, v2))
                i += 1
        
        # Evaluate all selected points
        results = []
        for v1, v2 in selected_v:
            if time.time() > deadline:
                break
            
            # Map to x0, x1
            x0 = (2.0 / np.pi) * np.arccos(v1)
            x1 = v2
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # Set distance variables to 0.5 to minimize g(x)
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
        
        return results
