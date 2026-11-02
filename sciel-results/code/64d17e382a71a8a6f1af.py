"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog, minimize

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    """
    
    def solve_radii_lp(centers, W, H, delaunay_edges=None):
        """Solve LP for optimal radii given fixed centers and box size.
        Uses Delaunay edges to reduce constraint count if provided."""
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= d_ij
        # Use Delaunay edges for efficiency, but include all pairs for safety
        if delaunay_edges is not None:
            pairs = set()
            for i, j in delaunay_edges:
                pairs.add((min(i,j), max(i,j)))
            # Also add close pairs (within 2x average distance)
            avg_dist = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    avg_dist += d
                    count += 1
            avg_dist /= count if count > 0 else 1
            for i in range(n):
                for j in range(i+1, n):
                    if (i,j) not in pairs:
                        d = np.linalg.norm(centers[i] - centers[j])
                        if d < 2.5 * avg_dist:
                            pairs.add((i,j))
        else:
            pairs = set()
            for i in range(n):
                for j in range(i+1, n):
                    pairs.add((i,j))
        
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
        
        # Boundary constraints: r_i <= dist_to_boundary
        for i in range(n):
            # Left
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][0])
            # Right
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(W - centers[i][0])
            # Bottom
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][1])
            # Top
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(H - centers[i][1])
            
        if not A_ub:
            return np.zeros(n)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x
        except Exception:
            pass
        return None

    def get_delaunay_edges(centers):
        """Get Delaunay triangulation edges from centers."""
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    j = (i + 1) % 3
                    a, b = int(simplex[i]), int(simplex[j])
                    edges.add((min(a,b), max(a,b)))
            return list(edges)
        except Exception:
            return None

    def optimize_packing(centers, W, H, max_iters=200, tol=1e-7):
        """
        Iteratively optimize centers using gradient-based approach.
        Uses Delaunay triangulation to identify tight constraints.
        """
        centers = centers.copy()
        n = len(centers)
        best_val = -np.inf
        best_centers = centers.copy()
        best_radii = None
        
        step_size = 0.05
        stall_count = 0
        prev_val = -np.inf
        
        for it in range(max_iters):
            # Get Delaunay edges
            d_edges = get_delaunay_edges(centers)
            
            # Solve LP for radii
            radii = solve_radii_lp(centers, W, H, d_edges)
            if radii is None:
                break
            
            val = np.sum(radii)
            if val > best_val:
                best_val = val
                best_centers = centers.copy()
                best_radii = radii.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            if stall_count > 30:
                # Perturb to escape local minima
                centers += np.random.normal(0, 0.005, size=centers.shape)
                centers[:, 0] = np.clip(centers[:, 0], 1e-6, W - 1e-6)
                centers[:, 1] = np.clip(centers[:, 1], 1e-6, H - 1e-6)
                stall_count = 0
                step_size = max(0.005, step_size * 0.7)
                continue
            
            # Compute gradient of sum(radii) w.r.t. centers
            # For tight pair constraint: r_i + r_j = d_ij
            # dr_i/dc_i + dr_j/c_j = gradient of d_ij
            # For boundary: r_i = dist, dr_i/dc_i = gradient of dist
            
            grad = np.zeros_like(centers)
            
            # Pair gradients
            if d_edges is not None:
                for i, j in d_edges:
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    if d < 1e-10:
                        continue
                    # Check if tight
                    if abs(radii[i] + radii[j] - d) < 1e-4:
                        # Gradient of d_ij w.r.t. center_i is d_vec/d
                        # Gradient of d_ij w.r.t. center_j is -d_vec/d
                        grad[i] += d_vec / d
                        grad[j] -= d_vec / d
            
            # Boundary gradients
            for i in range(n):
                if abs(radii[i] - centers[i][0]) < 1e-4:
                    grad[i][0] += 1.0
                if abs(radii[i] - (W - centers[i][0])) < 1e-4:
                    grad[i][0] -= 1.0
                if abs(radii[i] - centers[i][1]) < 1e-4:
                    grad[i][1] += 1.0
                if abs(radii[i] - (H - centers[i][1])) < 1e-4:
                    grad[i][1] -= 1.0
            
            # Apply gradient step
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                centers += step_size * grad / grad_norm
            
            # Clamp to box with small margin
            centers[:, 0] = np.clip(centers[:, 0], 1e-6, W - 1e-6)
            centers[:, 1] = np.clip(centers[:, 1], 1e-6, H - 1e-6)
            
            # Decay step size
            step_size *= 0.99
            
            # Convergence check
            if abs(val - prev_val) < tol and it > 50:
                break
            prev_val = val
        
        if best_radii is not None:
            return best_val, best_centers, best_radii
        return -np.inf, None, None

    def run_optimization(W, H, initial_centers):
        """Run local optimization for a specific box size and initial config."""
        val, centers, radii = optimize_packing(initial_centers, W, H, max_iters=150)
        return val, centers, radii

    # Aspect ratios to try (W/H)
    # Perimeter 2(W+H) = 4 => W+H=2
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 1.3, 0.77, 1.1, 0.9, 1.4, 0.7, 1.6, 0.6, 1.8, 0.5]
    
    best_val_global = -np.inf
    best_result_global = None
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Hexagonal packing pattern (better than grid for circles)
    def make_hex_pattern(n_circles, W, H):
        """Create hexagonal packing pattern."""
        pts = []
        # Try different hex spacings
        for spacing_mult in [0.9, 1.0, 1.1, 1.2]:
            # Hex packing: rows offset
            row_height = spacing_mult * 0.866  # sqrt(3)/2
            col_width = spacing_mult
            row = 0
            while len(pts) < n_circles:
                x_offset = (row % 2) * col_width / 2
                col = 0
                while len(pts) < n_circles:
                    x = col_width/2 + x_offset + col * col_width
                    y = row_height/2 + row * row_height
                    if x < W and y < H:
                        pts.append([x, y])
                    col += 1
                row += 1
            if len(pts) >= n_circles:
                return np.array(pts[:n_circles])
        return None
    
    # 2. Grid patterns
    def make_grid_pattern(n_circles, W, H, nx, ny):
        """Create grid pattern."""
        pts = []
        for j in range(ny):
            for i in range(nx):
                if len(pts) >= n_circles:
                    break
                x = (i + 0.5) * W / nx
                y = (j + 0.5) * H / ny
                pts.append([x, y])
        if len(pts) >= n_circles:
            return np.array(pts[:n_circles])
        return None
    
    # 3. Random
    def make_random_pattern(n_circles, W, H, seed):
        rng = np.random.RandomState(seed)
        return rng.rand(n_circles, 2) * np.array([W, H])
    
    for aspect in aspects:
        H = 2.0 / (aspect + 1.0)
        W = 2.0 * aspect / (aspect + 1.0)
        
        # Generate configs for this aspect
        local_configs = []
        
        # Hex patterns
        hex_pts = make_hex_pattern(N, W, H)
        if hex_pts is not None:
            local_configs.append(hex_pts)
        
        # Grid patterns
        for nx, ny in [(5,5), (4,6), (6,4), (7,3), (3,7), (7,2), (2,7), (6,3), (3,6)]:
            grid_pts = make_grid_pattern(N, W, H, nx, ny)
            if grid_pts is not None:
                local_configs.append(grid_pts)
        
        # Random with different seeds
        for seed in [42, 123, 456, 789]:
            local_configs.append(make_random_pattern(N, W, H, seed))
        
        # Run optimization for each config
        for config in local_configs:
            val, centers, radii = run_optimization(W, H, config)
            
            if val > best_val_global:
                best_val_global = val
                result = np.zeros((N, 3))
                result[:, :2] = centers
                result[:, 2] = radii
                best_result_global = result
    
    if best_result_global is not None:
        return best_result_global
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
