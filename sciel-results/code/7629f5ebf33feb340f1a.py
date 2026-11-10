# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import itertools

TOL = 1e-7


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares with pairwise disjoint interiors inside the
    unit square [0,1]^2 so as to MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    t0 = time.time()
    DEADLINE = t0 + 580.0  # safety margin
    n = 10

    def _is_feasible_vec(centers, sides, tol=TOL):
        """
        Vectorized feasibility check.
        Returns True if all squares are inside [0,1]^2 and pairwise non-overlapping.
        """
        cx = centers[:, 0]
        cy = centers[:, 1]
        s = sides
        
        # Check bounds
        if np.any(s <= 0):
            return False
        if np.any(cx - s / 2.0 < -tol) or np.any(cx + s / 2.0 > 1.0 + tol):
            return False
        if np.any(cy - s / 2.0 < -tol) or np.any(cy + s / 2.0 > 1.0 + tol):
            return False
        
        # Check overlaps
        cx_diff = np.abs(cx[:, None] - cx[None, :])
        cy_diff = np.abs(cy[:, None] - cy[None, :])
        s_sum = (s[:, None] + s[None, :]) / 2.0 - tol
        
        mask = ~np.eye(n, dtype=bool)
        
        overlap_x = cx_diff < s_sum
        overlap_y = cy_diff < s_sum
        
        if np.any(overlap_x & overlap_y & mask):
            return False
            
        return True

    def _score(centers, sides):
        if not _is_feasible_vec(centers, sides):
            return -np.inf
        return np.sum(sides)

    def _make_layout_30():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        idx = 0
        # Row 0: 4 squares of 0.25
        s1 = 0.25
        for j in range(4):
            centers[idx, 0] = j * s1 + s1 / 2
            centers[idx, 1] = s1 / 2
            sides[idx] = s1
            idx += 1
        # Row 1: 3 squares of 1/3
        s2 = 1.0 / 3.0
        y1 = s1 + s2 / 2
        for j in range(3):
            centers[idx, 0] = j * s2 + s2 / 2
            centers[idx, 1] = y1
            sides[idx] = s2
            idx += 1
        # Row 2: 3 squares of 1/3
        s3 = 1.0 / 3.0
        y2 = s1 + s2 + s3 / 2
        for j in range(3):
            centers[idx, 0] = j * s3 + s3 / 2
            centers[idx, 1] = y2
            sides[idx] = s3
            idx += 1
        return centers, sides

    def _max_side_at(cx, cy, centers, sides, i, tol=TOL):
        """
        Compute the maximum side length for square i centered at (cx,cy)
        given the positions and sizes of all other squares.
        """
        # Boundary constraints
        max_s = min(2.0 * cx, 2.0 * (1.0 - cx), 2.0 * cy, 2.0 * (1.0 - cy))
        if max_s < 0:
            max_s = 0.0
        
        for j in range(len(sides)):
            if j == i:
                continue
            dx = abs(cx - centers[j, 0])
            dy = abs(cy - centers[j, 1])
            # To avoid overlap with j:
            # |cx - cx_j| >= (s_i + s_j)/2 - tol  OR  |cy - cy_j| >= (s_i + s_j)/2 - tol
            # We want to find max s_i.
            # If we rely on x-separation: s_i <= 2*dx - s_j + 2*tol
            # If we rely on y-separation: s_i <= 2*dy - s_j + 2*tol
            # The constraint is (s_i <= bound_x) OR (s_i <= bound_y) is NOT correct.
            # The condition for NO overlap is:
            # NOT (|dx| < (s_i+s_j)/2 - tol AND |dy| < (s_i+s_j)/2 - tol)
            # This is equivalent to:
            # |dx| >= (s_i+s_j)/2 - tol  OR  |dy| >= (s_i+s_j)/2 - tol
            # Let t = (s_i+s_j)/2 - tol.
            # We need |dx| >= t OR |dy| >= t.
            # t <= |dx|  => (s_i+s_j)/2 - tol <= |dx| => s_i <= 2|dx| - s_j + 2tol
            # t <= |dy|  => s_i <= 2|dy| - s_j + 2tol
            # So if s_i <= min(2|dx|, 2|dy|) - s_j + 2tol, is it safe?
            # No. If s_i is small enough to satisfy ONE of the conditions, it's safe.
            # Condition: (s_i <= 2|dx| - s_j + 2tol) OR (s_i <= 2|dy| - s_j + 2tol)
            # This is always true if s_i <= max(2|dx| - s_j + 2tol, 2|dy| - s_j + 2tol).
            # Wait. If s_i is very large, both conditions fail.
            # We need at least one to hold.
            # So s_i must be <= max(bound_x, bound_y).
            
            bound_x = 2.0 * dx - sides[j] + 2.0 * tol
            bound_y = 2.0 * dy - sides[j] + 2.0 * tol
            bound = max(bound_x, bound_y)
            
            if bound < max_s:
                max_s = bound
        
        if max_s < 0:
            max_s = 0.0
        return max_s

    def _grow_all_squares(centers, sides, tol=TOL, max_rounds=50):
        """
        Iteratively grow all squares to their maximum given current positions.
        """
        n = len(sides)
        for _ in range(max_rounds):
            improved = False
            for i in range(n):
                new_s = _max_side_at(centers[i, 0], centers[i, 1], centers, sides, i, tol)
                if new_s > sides[i] + 1e-12:
                    sides[i] = new_s
                    improved = True
            if not improved:
                break
        return sides

    def _local_search(centers_init, sides_init, max_iters=5000, seed=42):
        rng = np.random.default_rng(seed)
        best_c = centers_init.copy()
        best_s = sides_init.copy()
        best_val = _score(best_c, best_s)
        
        if not np.isfinite(best_val):
            return best_c, best_s, best_val

        it = 0
        step = 0.05
        while it < max_iters:
            if time.time() > DEADLINE - 5:
                break
            it += 1
            
            # Perturb a random subset of squares
            k = rng.integers(1, 4)
            idx = rng.choice(n, size=k, replace=False)
            
            new_c = best_c.copy()
            new_s = best_s.copy()
            
            for i in idx:
                new_c[i, 0] += rng.uniform(-step, step)
                new_c[i, 1] += rng.uniform(-step, step)
                
            # Clamp centers to keep squares inside [0,1]^2
            for i in range(n):
                half = new_s[i] / 2.0
                new_c[i, 0] = max(half, min(1.0 - half, new_c[i, 0]))
                new_c[i, 1] = max(half, min(1.0 - half, new_c[i, 1]))
                
            # Try to grow squares
            new_s = _grow_all_squares(new_c, new_s)
            
            if _is_feasible_vec(new_c, new_s):
                new_val = np.sum(new_s)
                if new_val > best_val:
                    best_c = new_c
                    best_s = new_s
                    best_val = new_val
                    # Maybe increase step slightly on success? Or keep.
                    # step = min(0.1, step * 1.02)
                else:
                    # Simulated annealing acceptance?
                    # For now, just reject non-improving moves to keep it simple and fast.
                    # Or accept with small prob.
                    pass
            else:
                # Infeasible after growth. This can happen if positions are bad.
                # We could try to resolve overlaps, but that's complex.
                # Just reject.
                pass
                
            # Cool down / adapt step
            if it % 100 == 0:
                step *= 0.95
            if step < 1e-4:
                break
                
        return best_c, best_s, best_val

    # Main logic
    seeds = []
    
    # Seed 1: The 3.0 grid
    c1, s1 = _make_layout_30()
    seeds.append((c1, s1))
    
    # Seed 2: Random perturbation of Seed 1
    c2, s2 = _make_layout_30()
    rng_seed = np.random.default_rng(123)
    for i in range(n):
        c2[i, 0] += rng_seed.uniform(-0.02, 0.02)
        c2[i, 1] += rng_seed.uniform(-0.02, 0.02)
        s2[i] += rng_seed.uniform(-0.02, 0.02)
    # Clamp
    for i in range(n):
        half = s2[i] / 2.0
        c2[i, 0] = max(half, min(1.0 - half, c2[i, 0]))
        c2[i, 1] = max(half, min(1.0 - half, c2[i, 1]))
    
    if not _is_feasible_vec(c2, s2):
        # Fallback to seed 1 if perturbation broke it
        c2, s2 = c1.copy(), s1.copy()
    else:
        # Try to grow
        s2 = _grow_all_squares(c2, s2)
        if not _is_feasible_vec(c2, s2):
             c2, s2 = c1.copy(), s1.copy()
    seeds.append((c2, s2))
    
    best_centers, best_sides = seeds[0]
    best_score = _score(best_centers, best_sides)
    
    # Run local search from each seed
    for i, (c_seed, s_seed) in enumerate(seeds):
        if time.time() > DEADLINE - 30:
            break
        c_res, s_res, val_res = _local_search(c_seed, s_seed, seed=42 + i)
        if val_res > best_score:
            best_centers, best_sides, best_score = c_res, s_res, val_res

    # Final check
    if not _is_feasible_vec(best_centers, best_sides):
        best_centers, best_sides = _make_layout_30()
        best_score = _score(best_centers, best_sides)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
