# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import linprog, minimize

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    
    Approach: Use Delaunay triangulation to identify relevant circle-circle
    constraints, solve LP for radii. Optimize centers using scipy.optimize.minimize
    (Powell method) to maximize the sum of radii.
    """
    N = 21
    
    def get_delaunay_pairs(centers):
        """Get Delaunay neighbor pairs from center positions."""
        try:
            tri = Delaunay(centers)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        edges.add((min(a, b), max(a, b)))
            return list(edges)
        except Exception:
            # Fallback: all pairs if Delaunay fails
            return [(i, j) for i in range(len(centers)) for j in range(i + 1, len(centers))]

    def solve_lp(centers, W, H, delaunay_edges):
        """
        Solve LP for optimal radii given fixed centers and box size.
        Returns sum of radii or 0 if infeasible.
        """
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= d_ij (only Delaunay edges)
        for i, j in delaunay_edges:
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
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return -res.fun  # res.fun is min(-sum(r)) = -sum(r), so -res.fun = sum(r)
            else:
                return 0.0
        except Exception:
            return 0.0

    def objective(centers_flat, W, H):
        """
        Objective function for scipy.optimize.minimize.
        Minimizes -sum(radii).
        """
        centers = centers_flat.reshape((N, 2))
        
        # Check if centers are within bounds
        if np.any(centers[:, 0] < 1e-6) or np.any(centers[:, 0] > W - 1e-6) or \
           np.any(centers[:, 1] < 1e-6) or np.any(centers[:, 1] > H - 1e-6):
            return 1e6 # Large penalty

        delaunay_edges = get_delaunay_pairs(centers)
        val = solve_lp(centers, W, H, delaunay_edges)
        return -val

    def solve_lp_with_radii(centers, W, H, delaunay_edges):
        """Solve LP and return radii."""
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        for i, j in delaunay_edges:
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
        
        for i in range(n):
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][0])
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(W - centers[i][0])
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(centers[i][1])
            row = np.zeros(n); row[i] = 1
            A_ub.append(row); b_ub.append(H - centers[i][1])
            
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

    def optimize_packing(centers_init, W, H):
        """
        Optimize centers using Powell's method.
        """
        centers_init = np.array(centers_init, dtype=float)
        centers_init[:, 0] = np.clip(centers_init[:, 0], 1e-5, W - 1e-5)
        centers_init[:, 1] = np.clip(centers_init[:, 1], 1e-5, H - 1e-5)
        
        x0 = centers_init.flatten()
        
        try:
            res = minimize(objective, x0, args=(W, H), method='Powell', 
                           options={'maxiter': 100, 'maxfev': 500, 'disp': False})
            best_centers = res.x.reshape((N, 2))
            best_centers[:, 0] = np.clip(best_centers[:, 0], 1e-5, W - 1e-5)
            best_centers[:, 1] = np.clip(best_centers[:, 1], 1e-5, H - 1e-5)
            
            delaunay_edges = get_delaunay_pairs(best_centers)
            radii = solve_lp_with_radii(best_centers, W, H, delaunay_edges)
            if radii is not None:
                return np.sum(radii), best_centers, radii
            else:
                return 0.0, best_centers, np.zeros(N)
        except Exception:
            return 0.0, centers_init, np.zeros(N)

    # Reduced set of aspects and configs for speed
    aspects = [1.0, 1.2, 0.8, 1.5, 0.67, 1.1, 0.9]
    
    best_val_global = -np.inf
    best_result_global = None
    
    configs = []
    
    # 5x5 grid
    gx, gy = 5, 5
    cx = np.linspace(0.5/gx, 1 - 0.5/gx, gx)
    cy = np.linspace(0.5/gy, 1 - 0.5/gy, gy)
    grid_pts = np.array([[x, y] for y in cy for x in cx])
    configs.append(grid_pts[:21])
    
    # Hexagonal-like pattern
    hex_pts = []
    for row in range(5):
        for col in range(5):
            x = (col + 0.5) / 5.0 + (0.5 / 5.0) * (row % 2)
            y = (row + 0.5) / 5.0
            hex_pts.append([x, y])
    configs.append(np.array(hex_pts[:21]))
    
    # Random seeds
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        configs.append(np.random.rand(21, 2))

    for aspect in aspects:
        H = 2.0 / (aspect + 1.0)
        W = 2.0 * aspect / (aspect + 1.0)
        
        for config in configs:
            centers_init = config * np.array([W, H])
            
            val, centers, radii = optimize_packing(centers_init, W, H)
            
            if val > best_val_global:
                best_val_global = val
                result = np.zeros((21, 3))
                result[:, :2] = centers
                result[:, 2] = radii
                best_result_global = result

    if best_result_global is not None:
        result = best_result_global.copy()
        if result[:, 0].min() < 0:
            result[:, 0] -= result[:, 0].min()
        if result[:, 1].min() < 0:
            result[:, 1] -= result[:, 1].min()
        return result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
