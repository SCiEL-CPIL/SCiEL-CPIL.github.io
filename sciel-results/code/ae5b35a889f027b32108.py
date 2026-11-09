# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 axis-aligned squares packed
    inside the unit square [0,1]^2.
    
    Strategy:
    1. Generate multiple diverse initial guesses (grid, corner+center, etc.)
    2. Refine each with scipy.optimize.minimize using penalty-based objective
    3. Return the best valid configuration found
    
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 630s kill

    def is_valid(centers, sides):
        """Check if configuration is valid per task contract."""
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
                return False
            if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if t <= 0:
                    continue
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    return False
        return True

    def objective(x):
        """Penalty-based objective: minimize -sum(sides) + penalties."""
        s = x[:n]
        cx = x[n:2 * n]
        cy = x[2 * n:3 * n]
        
        # Clip sides to positive for numerical stability
        s_eff = np.maximum(s, 1e-8)
        
        # Boundary penalties
        pen_bound = 0.0
        for i in range(n):
            si = s_eff[i]
            pen_bound += max(0, cx[i] + si / 2 - 1) ** 2
            pen_bound += max(0, si / 2 - cx[i]) ** 2
            pen_bound += max(0, cy[i] + si / 2 - 1) ** 2
            pen_bound += max(0, si / 2 - cy[i]) ** 2
        
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                si = s_eff[i]
                sj = s_eff[j]
                t = (si + sj) / 2
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < t and dy < t:
                    depth = min(t - dx, t - dy)
                    pen_overlap += depth ** 2
        
        sum_s = np.sum(s_eff)
        return -sum_s + 10000.0 * (pen_bound + pen_overlap)

    def generate_grid_guess(seed):
        """Standard k×k grid packing."""
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        # Add small noise
        rng = np.random.RandomState(seed)
        centers += rng.normal(0, 0.005, centers.shape)
        sides += rng.normal(0, 0.002, sides.shape)
        sides = np.maximum(sides, 0.1)
        return centers, sides

    def generate_corner_center_guess(seed):
        """4 large corner squares + 6 smaller central squares."""
        rng = np.random.RandomState(seed)
        s_large = 0.32
        s_small = 0.14
        
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # 4 corners
        corners = [
            [s_large / 2, s_large / 2],
            [1 - s_large / 2, s_large / 2],
            [s_large / 2, 1 - s_large / 2],
            [1 - s_large / 2, 1 - s_large / 2]
        ]
        for i in range(4):
            centers[i] = corners[i]
            sides[i] = s_large
        
        # 6 center squares in 2×3 grid
        avail_w = 1 - 2 * s_large
        avail_h = 1 - 2 * s_large
        
        grid_w = 2 * s_small
        grid_h = 3 * s_small
        
        start_x = s_large + (avail_w - grid_w) / 2
        start_y = s_large + (avail_h - grid_h) / 2
        
        k = 0
        for row in range(3):
            for col in range(2):
                idx = 4 + k
                cx = start_x + col * s_small + s_small / 2
                cy = start_y + row * s_small + s_small / 2
                centers[idx] = [cx, cy]
                sides[idx] = s_small
                k += 1
        
        # Add noise
        centers += rng.normal(0, 0.01, centers.shape)
        sides += rng.normal(0, 0.005, sides.shape)
        sides = np.maximum(sides, 0.05)
        
        return centers, sides

    def generate_diagonal_large_guess(seed):
        """2 large diagonal corner squares + 8 smaller squares."""
        rng = np.random.RandomState(seed)
        
        s_large = 0.40
        s_small = 0.15
        
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # 2 large diagonal corners
        centers[0] = [s_large / 2, s_large / 2]
        sides[0] = s_large
        centers[1] = [1 - s_large / 2, 1 - s_large / 2]
        sides[1] = s_large
        
        # 8 small squares: 4 in bottom-right, 4 in top-left
        br_x0 = s_large
        br_y0 = 0
        for r in range(2):
            for c in range(2):
                idx = 2 + r * 2 + c
                cx = br_x0 + c * s_small + s_small / 2
                cy = br_y0 + r * s_small + s_small / 2
                centers[idx] = [cx, cy]
                sides[idx] = s_small
        
        tl_x0 = 0
        tl_y0 = 1 - s_large
        for r in range(2):
            for c in range(2):
                idx = 6 + r * 2 + c
                cx = tl_x0 + c * s_small + s_small / 2
                cy = tl_y0 + r * s_small + s_small / 2
                centers[idx] = [cx, cy]
                sides[idx] = s_small
        
        # Add noise
        centers += rng.normal(0, 0.008, centers.shape)
        sides += rng.normal(0, 0.004, sides.shape)
        sides = np.maximum(sides, 0.05)
        
        return centers, sides

    def generate_mixed_edge_guess(seed):
        """4 corners medium, 4 edges small, 2 center smallest."""
        rng = np.random.RandomState(seed)
        
        s_corner = 0.30
        s_edge = 0.20
        s_center = 0.15
        
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        corners = [
            [s_corner / 2, s_corner / 2],
            [1 - s_corner / 2, s_corner / 2],
            [s_corner / 2, 1 - s_corner / 2],
            [1 - s_corner / 2, 1 - s_corner / 2]
        ]
        for i in range(4):
            centers[i] = corners[i]
            sides[i] = s_corner
        
        edges = [
            [0.5, s_edge / 2],
            [0.5, 1 - s_edge / 2],
            [s_edge / 2, 0.5],
            [1 - s_edge / 2, 0.5],
        ]
        for i in range(4):
            centers[4 + i] = edges[i]
            sides[4 + i] = s_edge
        
        centers[8] = [0.35, 0.5]
        sides[8] = s_center
        centers[9] = [0.65, 0.5]
        sides[9] = s_center
        
        centers += rng.normal(0, 0.008, centers.shape)
        sides += rng.normal(0, 0.004, sides.shape)
        sides = np.maximum(sides, 0.05)
        
        return centers, sides

    def optimize_from(centers, sides, max_iters=300):
        """Run scipy optimization from given initial guess."""
        x0 = np.concatenate([sides, centers[:, 0], centers[:, 1]])
        
        bounds = []
        for i in range(n):
            bounds.append((0.01, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        
        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        x_opt = result.x
        s_opt = x_opt[:n]
        cx_opt = x_opt[n:2 * n]
        cy_opt = x_opt[2 * n:3 * n]
        centers_opt = np.stack([cx_opt, cy_opt], axis=1)
        
        return centers_opt, s_opt, result.fun

    # Generate multiple initial guesses
    generators = [
        generate_grid_guess,
        generate_corner_center_guess,
        generate_diagonal_large_guess,
        generate_mixed_edge_guess,
    ]
    
    best_value = -np.inf
    best_centers = None
    best_sides = None
    
    # Always keep the baseline grid as fallback
    baseline_centers, baseline_sides = generate_grid_guess(0)
    if is_valid(baseline_centers, baseline_sides):
        best_value = np.sum(baseline_sides)
        best_centers = baseline_centers.copy()
        best_sides = baseline_sides.copy()
    
    # Try each generator with multiple seeds
    for gen_idx, gen in enumerate(generators):
        for seed in range(4):
            if time.time() > deadline - 60:
                break
                
            try:
                centers_init, sides_init = gen(seed)
            except Exception:
                continue
            
            # Quick validity check before optimizing
            if not is_valid(centers_init, sides_init):
                # Try to repair by projecting to bounds
                for i in range(n):
                    s = sides_init[i]
                    centers_init[i, 0] = np.clip(centers_init[i, 0], s / 2, 1 - s / 2)
                    centers_init[i, 1] = np.clip(centers_init[i, 1], s / 2, 1 - s / 2)
                if not is_valid(centers_init, sides_init):
                    continue
            
            # Run optimization
            try:
                centers_opt, sides_opt, obj_val = optimize_from(centers_init, sides_init)
            except Exception:
                continue
            
            # Check validity
            if is_valid(centers_opt, sides_opt):
                val = np.sum(sides_opt)
                if val > best_value:
                    best_value = val
                    best_centers = centers_opt.copy()
                    best_sides = sides_opt.copy()
                    print(f"  New best: {val:.6f} (gen={gen_idx}, seed={seed})")
        
        if time.time() > deadline - 60:
            break
    
    # Final fallback
    if best_centers is None or best_sides is None:
        best_centers, best_sides = baseline_centers, baseline_sides
    
    return best_centers, best_sides


# EVOLVE-BLOCK-END
