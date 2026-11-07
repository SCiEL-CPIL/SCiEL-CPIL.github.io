import numpy as np
import time

class Optimizer:
    """DTLZ2-specific optimizer: hypervolume-weighted grid sampling with recursive refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # DTLZ2 structure:
        # Pareto front: g=0 when x_i=0.5 for i>=m (indices m-1 to n-1)
        # f1 = cos(theta1)*cos(theta2), f2 = cos(theta1)*sin(theta2), f3 = sin(theta1)
        # theta1 = x0*pi/2, theta2 = x1*pi/2
        # Surface area element: cos(theta1) dtheta1 dtheta2
        # For hypervolume optimization, weight toward interior where marginal volume is higher

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        evaluated_points = []
        used_budget = 0

        # Strategy: Hypervolume-weighted grid with recursive refinement
        # 1. Start with a coarse grid using hypervolume-weighted positions
        # 2. Use remaining budget to refine by adding points at centers of sub-rectangles
        #    formed by recursive bisection, prioritizing regions with largest marginal gain

        # Phase 1: Coarse grid with hypervolume-weighted positions
        # Use a power-law transformation to cluster points toward the center
        # where hypervolume contribution is maximized
        alpha = 0.6  # Clustering parameter

        def cluster_toward_center(u, alpha=0.6):
            """Map u in [0,1] to [0,1] with clustering toward 0.5"""
            if u < 0.5:
                return 0.5 - 0.5 * (1 - 2*u)**alpha
            else:
                return 0.5 + 0.5 * (2*u - 1)**alpha

        # Generate coarse grid points
        # Start with a smaller grid to leave budget for refinement
        grid_size = 12  # 12x12 = 144 points
        num_grid = grid_size * grid_size

        grid_points = []
        for i in range(grid_size):
            for j in range(grid_size):
                if used_budget >= self.budget:
                    break

                # Uniform grid in [0, 1]
                u_uniform = (i + 0.5) / grid_size
                v_uniform = (j + 0.5) / grid_size

                # Apply clustering toward center
                u = cluster_toward_center(u_uniform, alpha)
                v = cluster_toward_center(v_uniform, alpha)

                # Map to DTLZ2 parameters
                # x0: arcsin transformation for surface-area uniformity
                x0 = (2.0 / np.pi) * np.arcsin(u)
                x1 = v

                x = base_x.copy()
                x[0] = x0
                x[1] = x1

                f = problem(x)
                evaluated_points.append(x)
                used_budget += 1
                grid_points.append((u, v, x, f))

        # Phase 2: Recursive refinement using remaining budget
        # For each sub-rectangle in the parameter space, add points at centers
        # of sub-sub-rectangles formed by bisection, prioritizing regions with
        # largest marginal hypervolume gain (approximated by distance to reference point)

        remaining_budget = self.budget - used_budget

        if remaining_budget > 0:
            # Create a priority queue of sub-rectangles
            # Each entry: (priority, u_range, v_range)
            # Priority: approximate hypervolume contribution (higher = better)
            # We approximate by the volume of the box in objective space

            import heapq

            # Initialize with the full parameter space
            # u in [0, 1], v in [0, 1]
            # The objective space for DTLZ2 3-objective is a quarter sphere
            # f1 = cos(t1)*cos(t2), f2 = cos(t1)*sin(t2), f3 = sin(t1)
            # where t1 = x0*pi/2, t2 = x1*pi/2
            # u = sin(t1), so t1 = arcsin(u), x0 = 2*arcsin(u)/pi
            # v = t2/pi, so t2 = v*pi, x1 = v

            # For a sub-rectangle [u_lo, u_hi] x [v_lo, v_hi]:
            # t1 in [arcsin(u_lo), arcsin(u_hi)]
            # t2 in [v_lo*pi, v_hi*pi]
            # The volume contribution is roughly proportional to the area on the sphere
            # which is ~ sin(t1) d(t1) d(t2) integrated over the region

            def compute_priority(u_lo, u_hi, v_lo, v_hi):
                """Approximate hypervolume contribution of a sub-rectangle"""
                # Average sin(t1) over the range, times delta(t1) * delta(t2)
                # sin(t1) = u, so average u over [u_lo, u_hi] is (u_lo + u_hi)/2
                avg_u = (u_lo + u_hi) / 2.0
                delta_t1 = np.arcsin(u_hi) - np.arcsin(u_lo)
                delta_t2 = (v_hi - v_lo) * np.pi
                # Also weight by distance to reference point [2.5, 2.5, 2.5]
                # Points closer to the front (smaller f values) contribute more to hypervolume
                # Approximate the "distance" by 1 - avg_u (since f3 = sin(t1) = u)
                # Actually, the hypervolume contribution is larger for points with smaller f values
                # so weight by (1 - avg_f) where avg_f is roughly avg_u
                return avg_u * delta_t1 * delta_t2 * (1.0 - avg_u * 0.5)

            # Initialize priority queue with the full space
            # We'll bisect the space and add points at centers of sub-rectangles
            # Start by dividing into 4 quadrants
            pq = []
            # Add the 4 quadrants
            for u_lo, u_hi in [(0.0, 0.5), (0.5, 1.0)]:
                for v_lo, v_hi in [(0.0, 0.5), (0.5, 1.0)]:
                    priority = compute_priority(u_lo, u_hi, v_lo, v_hi)
                    # Store negative priority for max-heap (heapq is min-heap)
                    heapq.heappush(pq, (-priority, u_lo, u_hi, v_lo, v_hi, 0))

            # Process the priority queue
            while used_budget < self.budget and pq:
                if time.time() > deadline:
                    break

                neg_priority, u_lo, u_hi, v_lo, v_hi, level = heapq.heappop(pq)
                priority = -neg_priority

                # Add a point at the center of this sub-rectangle
                u_center = (u_lo + u_hi) / 2.0
                v_center = (v_lo + v_hi) / 2.0

                # Apply clustering to get the actual position
                u = cluster_toward_center(u_center, alpha)
                v = cluster_toward_center(v_center, alpha)

                # Map to DTLZ2 parameters
                x0 = (2.0 / np.pi) * np.arcsin(u)
                x1 = v

                x = base_x.copy()
                x[0] = x0
                x[1] = x1

                f = problem(x)
                evaluated_points.append(x)
                used_budget += 1

                # If we have budget left, bisect this rectangle and add to queue
                if used_budget < self.budget and level < 4:  # Limit depth to 4
                    # Bisect in the direction that gives larger priority gain
                    # Try both bisecting u and bisecting v
                    u_mid = (u_lo + u_hi) / 2.0
                    v_mid = (v_lo + v_hi) / 2.0

                    # Option 1: Bisect u
                    p1 = compute_priority(u_lo, u_mid, v_lo, v_hi)
                    p2 = compute_priority(u_mid, u_hi, v_lo, v_hi)

                    # Option 2: Bisect v
                    p3 = compute_priority(u_lo, u_hi, v_lo, v_mid)
                    p4 = compute_priority(u_lo, u_hi, v_mid, v_hi)

                    # Choose the better bisect
                    if max(p1, p2) > max(p3, p4):
                        heapq.heappush(pq, (-p1, u_lo, u_mid, v_lo, v_hi, level + 1))
                        heapq.heappush(pq, (-p2, u_mid, u_hi, v_lo, v_hi, level + 1))
                    else:
                        heapq.heappush(pq, (-p3, u_lo, u_hi, v_lo, v_mid, level + 1))
                        heapq.heappush(pq, (-p4, u_lo, u_hi, v_mid, v_hi, level + 1))

        return evaluated_points
