"""Multi-objective optimizer on WFG4 using hypervolume-indicator-driven sequential search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-indicator-driven sequential multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        rng = np.random

        # Archive: list of objective vectors (non-dominated)
        archive = []
        archive_X = []  # corresponding decision variables

        def update_archive(f, x):
            """Add f to archive if non-dominated, remove dominated archive points.
            Returns True if f was added."""
            # Check if f is dominated by any archive point
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove archive points dominated by f
            new_archive = []
            new_archive_X = []
            for i, af in enumerate(archive):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append(af)
                    new_archive_X.append(archive_X[i])
            new_archive.append(f.copy())
            new_archive_X.append(x.copy())
            archive[:] = new_archive
            archive_X[:] = new_archive_X
            return True

        def hypervolume_3d(points, ref):
            """Compute exact 3D hypervolume for a set of points vs reference point.
            Only points strictly better than ref in all objectives count."""
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            if len(valid) == 1:
                p = valid[0]
                return float(np.prod(ref - p))

            # Sort by first objective
            valid_sorted = sorted(valid, key=lambda p: p[0])
            n = len(valid_sorted)
            hv = 0.0

            for i in range(n):
                pi = valid_sorted[i]
                w0 = ref[0] - pi[0]
                if w0 <= 0:
                    continue

                # Compute 2D hypervolume of points from i to n-1 in dims 1,2
                # Reference for 2D is (ref[1], ref[2])
                # We need the area of the union of rectangles [p[1], ref[1]] x [p[2], ref[2]]
                # for p in valid_sorted[i:]
                
                # A simple O(K^2) method for the 2D slice is fast enough for K <= 200
                # But we can do better.
                # Let's just use a simple grid or sorting method for 2D.
                # For 2D HV: sort by y, sweep z? No, just sort by y and keep min z?
                # Actually, for 2D HV with ref (Ry, Rz):
                # Sort points by y. 
                # For each point, its contribution is (Ry - y_i) * (Rz - max_z_so_far) 
                # where max_z_so_far is the minimum z value seen so far? No.
                # The hypervolume is the area of the union of boxes.
                # Since it's 2D, it's the area of the union of rectangles [y_i, Ry] x [z_i, Rz].
                # Sort by y_i ascending.
                # Keep track of the minimum z value seen so far (which defines the "bottom" of the union in the z-direction? No, the "top" in the minimization sense, i.e. the best value).
                # Wait, the rectangle is from y_i to Ry and z_i to Rz.
                # If we sort by y_i, the new rectangle extends from y_i to Ry.
                # The part that is NOT covered by previous rectangles (which had smaller y, so extended further left? No, previous had smaller y, so they cover [y_prev, Ry]. Since y_i > y_prev, [y_i, Ry] is a subset of [y_prev, Ry] in the y-dimension? No.
                # Let's visualize. 
                # P1: y=1, z=5. Rect: [1, 10]x[5, 10].
                # P2: y=2, z=3. Rect: [2, 10]x[3, 10].
                # Union: 
                # y in [1,2]: z in [5,10] (from P1). Area = 1 * 5 = 5.
                # y in [2,10]: z in [3,10] (from P2, which is better). Area = 8 * 7 = 56.
                # Total 61.
                # Algorithm:
                # Sort by y ascending.
                # min_z = inf
                # area = 0
                # prev_y = 0 (or -inf)
                # For each point (y, z) in sorted order:
                #   if z < min_z:
                #      area += (y - prev_y) * (Rz - min_z)
                #      min_z = z
                #   prev_y = y
                # After loop:
                #   area += (Ry - prev_y) * (Rz - min_z)
                
                pts_2d = [(p[1], p[2]) for p in valid_sorted[i:]]
                pts_2d.sort(key=lambda p: p[0])
                
                area_2d = 0.0
                min_z = ref[2]
                prev_y = 0.0 # This should be the start of the range, which is effectively -inf but we only care about the width between points.
                # Actually, the "width" in y for the segment after point j is y_{j+1} - y_j.
                # The height is Rz - min_z_seen_so_far.
                
                # Let's refine:
                # The union is a staircase.
                # Sort by y.
                # Keep track of the best (min) z seen so far.
                # The area contributed by the segment from y_current to y_next is (y_next - y_current) * (Rz - min_z_current).
                # The segment from the last y to Ry is (Ry - y_last) * (Rz - min_z_last).
                
                current_min_z = ref[2]
                last_y = 0.0 # Not quite, we need to handle the first point correctly.
                
                # Let's just iterate.
                # The "step" in y is from the current point's y to the next point's y.
                # The height is determined by the min z of all points up to the current one?
                # No, the point with the smallest y covers the largest range in y.
                
                # Correct 2D HV algorithm:
                # 1. Sort points by y ascending.
                # 2. Iterate. Keep `best_z` (minimum z so far).
                # 3. The area to the right of point i (from y_i to y_{i+1}) is (y_{i+1} - y_i) * (Rz - best_z).
                # 4. Initialize best_z = Rz (infinity).
                # 5. When we see a point with z < best_z, update best_z.
                # 6. The area from y_0 to y_1 uses the best_z after seeing point 0.
                
                best_z = ref[2]
                prev_y = 0.0 # Dummy, we will calculate widths based on differences.
                
                # Actually, simpler:
                # Area = sum over segments.
                # Segment 1: y in [y_0, y_1]. Height = Rz - min(z_0).
                # Segment 2: y in [y_1, y_2]. Height = Rz - min(z_0, z_1).
                # ...
                # Segment N: y in [y_{N-1}, Ry]. Height = Rz - min(z_0...z_{N-1}).
                
                # Let's implement this.
                ys = [p[0] for p in pts_2d]
                zs = [p[1] for p in pts_2d]
                
                # Add Ry to ys to handle the last segment
                # But we need to associate the min_z with the segment.
                
                total_area = 0.0
                current_min_z = ref[2]
                
                for k in range(len(pts_2d)):
                    y_k = pts_2d[k][0]
                    z_k = pts_2d[k][1]
                    
                    # The segment from previous y to y_k? No.
                    # The segment from y_k to y_{k+1} (or Ry) uses the min_z including z_k.
                    
                    # So we update min_z first?
                    # If I have points (1, 5) and (2, 3).
                    # k=0: y=1, z=5. min_z becomes 5.
                    # Next y is 2. Segment [1, 2]. Area = (2-1)*(Rz-5).
                    # k=1: y=2, z=3. min_z becomes 3.
                    # Next y is Ry. Segment [2, Ry]. Area = (Ry-2)*(Rz-3).
                    
                    # So:
                    # 1. Update min_z with z_k.
                    # 2. Determine next_y: if k < len-1, next_y = pts_2d[k+1][0], else next_y = ref[1].
                    # 3. width = next_y - y_k.
                    # 4. area += width * (ref[2] - min_z).
                    
                    if z_k < current_min_z:
                        current_min_z = z_k
                    
                    next_y = pts_2d[k+1][0] if k + 1 < len(pts_2d) else ref[1]
                    width = next_y - y_k
                    if width > 0:
                        total_area += width * (ref[2] - current_min_z)
                
                hv += w0 * total_area

            return hv

        # Phase 1: Front-biased initialization
        n_init = min(80, budget)
        used = 0

        for i in range(n_init):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Sample parameter t uniformly
            t = rng.random()

            # Construct x based on WFG4 front structure
            x = np.zeros(dim)
            x[0] = lo[0] + span[0] * t

            # For remaining variables, use a mix of front-biased and random values
            for d in range(1, dim):
                u = rng.random()
                if rng.random() < 0.5:
                    x[d] = lo[d] + span[d] * u
                else:
                    x[d] = (lo[d] + hi[d]) / 2.0 + span[d] * 0.3 * rng.standard_normal()
                    x[d] = np.clip(x[d], lo[d], hi[d])

            f = problem(x)
            used += 1
            update_archive(f, x)

        # Phase 2: Sequential search with local perturbations
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            if len(archive) == 0:
                x = lo + span * rng.random(dim)
                f = problem(x)
                used += 1
                update_archive(f, x)
                continue

            # 20% chance of global random, 80% local perturbation
            if rng.random() < 0.2:
                x = lo + span * rng.random(dim)
            else:
                idx = rng.randint(len(archive))
                base_x = archive_X[idx]
                # Perturb with Gaussian noise, scaled by span
                # Step size decreases slightly as we use more budget?
                # Keep it simple: fixed small step
                step = 0.1
                x = base_x + step * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)

            f = problem(x)
            used += 1
            update_archive(f, x)

        return archive_X if archive_X else [np.zeros(dim)]

# EVOLVE-BLOCK-END
