"""Multi-objective optimizer for RE34: Hypervolume-guided local search with exact HV."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq


class Optimizer:
    """Hypervolume-guided local search for RE34."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        # Exact 3D Hypervolume calculation
        def hypervolume_3d(points, ref_point):
            if len(points) == 0:
                return 0.0
            
            # Filter points that are strictly better than ref in all objectives
            valid_mask = np.all(points < ref_point, axis=1)
            points = points[valid_mask]
            if len(points) == 0:
                return 0.0
            
            # Remove dominated points to speed up
            # Sort by objective 0
            points = points[np.argsort(points[:, 0])]
            
            # Filter dominated
            front = [points[0]]
            for i in range(1, len(points)):
                p = points[i]
                # Check if p is dominated by any point in front
                # Since sorted by obj 0, we only need to check if there is a point in front 
                # with obj1 <= p[1] AND obj2 <= p[2]
                # We can maintain the minimum obj2 for each obj1? 
                # For small N, simple check is fine.
                dominated = False
                for q in front:
                    if q[1] <= p[1] and q[2] <= p[2]:
                        dominated = True
                        break
                if not dominated:
                    front.append(p)
            
            front = np.array(front)
            if len(front) == 0:
                return 0.0
            
            # Calculate HV of the front
            # Sort by obj 0
            front = front[np.argsort(front[:, 0])]
            
            # For each point i in front, the volume it contributes is:
            # (front[i+1, 0] - front[i, 0]) * (max_obj1 - front[i, 1]) * (max_obj2 - front[i, 2])?
            # No, that's not quite right. The standard 3D HV algorithm:
            # Sort by x.
            # Volume = sum over i of (x_{i+1} - x_i) * HV_2D(points[i..end], ref_yz)
            # This is O(N^2) or O(N^3).
            
            # Simpler O(N^2) approach for 3D:
            # The HV is the sum of the volume of the box defined by each point and the reference point,
            # minus the overlaps.
            # Or, we can use the "layer" method.
            
            # Let's use a simple O(N^2) method:
            # For each point p in front, the unique volume it contributes is the volume of the box [p, ref]
            # that is not covered by any other point q in front.
            # This is hard to compute directly.
            
            # Let's use the standard algorithm:
            # Sort by x ascending.
            # Let x_{n+1} = ref[0].
            # For i from 0 to n-1:
            #   The slice [x_i, x_{i+1}] has a 2D HV in the y-z plane.
            #   The points contributing to this slice are those with x <= x_{i+1}.
            #   Specifically, the "active" set is points 0..i.
            #   The 2D HV of the active set's y-z coordinates is computed.
            #   The volume is (x_{i+1} - x_i) * HV_2D(active_set_yz, ref_yz).
            
            # We need an efficient 2D HV calculator.
            def hv_2d(pts, ref_yz):
                if len(pts) == 0:
                    return 0.0
                # pts is Nx2
                # Sort by y
                pts = pts[np.argsort(pts[:, 0])]
                # The 2D HV is the sum of rectangles.
                # For each point i, the rectangle width is (y_{i+1} - y_i) and height is (ref_z - z_i)?
                # No.
                # Sort by y.
                # Maintain the minimum z so far?
                # Actually, for 2D, if we sort by y, the HV is sum of (y_{i+1} - y_i) * (ref_z - min_z_so_far)?
                # No, that's for the area under the curve.
                # The HV is the area of the union of boxes [p_i, ref].
                # If we sort by y ascending, and keep track of the minimum z value seen so far (which is the "front" in z),
                # then the contribution of point i is (y_{i+1} - y_i) * (ref_z - min_z_so_far).
                # Wait, if a point has a larger z, it doesn't help if a previous point has a smaller z.
                # So we want the minimum z for each y.
                
                # Sort by y
                order = np.argsort(pts[:, 0])
                sorted_pts = pts[order]
                
                min_z = np.inf
                total_hv = 0.0
                for i in range(len(sorted_pts)):
                    y_i = sorted_pts[i, 0]
                    z_i = sorted_pts[i, 1]
                    
                    # The width of this slice is up to the next point's y, or ref_y
                    y_next = sorted_pts[i+1, 0] if i + 1 < len(sorted_pts) else ref_yz[0]
                    
                    # The height is determined by the minimum z in the set of points with y <= y_next.
                    # Since we are iterating in increasing y, the "front" is the minimum z seen so far.
                    # But wait, if a new point has a larger z, it doesn't change the min_z.
                    # If it has a smaller z, it updates min_z.
                    
                    # Actually, the region dominated by the set of points up to y_next is determined by the min z.
                    # So the area of the slice [y_i, y_next] is (y_next - y_i) * (ref_z - min_z_so_far).
                    
                    if z_i < min_z:
                        min_z = z_i
                    
                    if min_z < ref_yz[1]:
                        total_hv += (y_next - y_i) * (ref_yz[1] - min_z)
                    else:
                        # If min_z >= ref_z, no more contribution
                        break
                        
                return total_hv

            # Now back to 3D
            total_hv = 0.0
            # We need to maintain the "active" set of y-z points.
            # As we sweep x, we add points to the active set.
            # We need to compute HV_2D of the active set efficiently.
            # Since N is small (<= 200), we can just recompute HV_2D from scratch for each x-slice.
            # O(N^2) overall.
            
            for i in range(len(front)):
                x_i = front[i, 0]
                x_next = front[i+1, 0] if i + 1 < len(front) else ref_point[0]
                
                # Active set is front[0..i]
                active_yz = front[0:i+1, 1:3]
                
                hv_2d_val = hv_2d(active_yz, ref_point[1:3])
                
                if x_next > x_i:
                    total_hv += (x_next - x_i) * hv_2d_val
                    if hv_2d_val == 0:
                        # No more contribution possible?
                        # Not necessarily, but if min z is already >= ref_z, then yes.
                        pass
                        
            return total_hv

        # Deterministic initialization using Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = []
            bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            for i in range(1, n + 1):
                p = [halton(i, b) for b in bases[:dim]]
                points.append(p)
            return np.array(points)

        init_pop_size = min(20, self.budget // 10)
        if init_pop_size < 5:
            init_pop_size = 5

        halton_pts = halton_sequence(init_pop_size, self.dim)
        X_init = lo + (hi - lo) * halton_pts
        F_init = np.array([problem(x) for x in X_init])
        
        X = X_init
        F = F_init
        n_evals = init_pop_size

        # Initial HV
        current_hv = hypervolume_3d(F, ref)

        # Local search loop
        # We will maintain a list of non-dominated points.
        # At each step, we pick a point from the front, mutate it, and if it improves HV, we accept it.
        # To speed up, we can use a "best-improvement" or "first-improvement" strategy.
        # Given the small budget, we can try multiple mutations per point.
        
        # Adaptive step size
        base_step = 0.1 * (hi - lo)
        
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            front_idx = []
            for i in range(len(F)):
                is_dom = False
                for j in range(len(F)):
                    if i != j:
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            is_dom = True
                            break
                if not is_dom:
                    front_idx.append(i)
            
            if not front_idx:
                front_idx = list(range(len(F)))
            
            # Pick a point to mutate
            # Strategy: Pick the point with the highest HV contribution?
            # Or pick randomly from the front?
            # Let's pick randomly to explore.
            idx = front_idx[np.random.randint(len(front_idx))]
            
            # Mutate
            # Use a Gaussian mutation
            # Step size can be adapted based on the "density" of the front
            # For now, use a fixed step size with some randomness
            step = base_step * np.random.uniform(0.5, 1.5)
            child = X[idx] + step * np.random.randn(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            n_evals += 1
            
            # Check if child is non-dominated
            is_dom = False
            for j in range(len(F)):
                if np.all(F[j] <= fc) and np.any(F[j] < fc):
                    is_dom = True
                    break
            
            if not is_dom:
                # Add child to population
                # We need to decide which point to replace.
                # If we add it, we should remove a point that is now dominated or has low contribution.
                # For simplicity, let's just add it and let the front update handle it.
                # But we have a fixed budget, so we must replace.
                
                # Find the point with the lowest HV contribution?
                # Or just replace a random dominated point?
                
                # Let's find dominated points
                dom_idx = []
                for i in range(len(F)):
                    is_dominated = False
                    for j in range(len(F)):
                        if i != j:
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                is_dominated = True
                                break
                    if is_dominated:
                        dom_idx.append(i)
                
                if dom_idx:
                    # Replace a random dominated point
                    replace_idx = dom_idx[np.random.randint(len(dom_idx))]
                else:
                    # Replace the point with the smallest box volume (proxy for contribution)
                    vols = (ref[0] - F[:, 0]) * (ref[1] - F[:, 1]) * (ref[2] - F[:, 2])
                    replace_idx = np.argmin(vols)
                
                X[replace_idx] = child
                F[replace_idx] = fc
                
                # Update HV
                new_hv = hypervolume_3d(F, ref)
                if new_hv > current_hv:
                    current_hv = new_hv
                # If new_hv <= current_hv, we might want to revert?
                # But we've already spent the evaluation.
                # Let's just keep it if it's non-dominated, as it might help in the future.
        
        return X
# EVOLVE-BLOCK-END
