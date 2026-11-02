"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import itertools

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    The bounding rectangle must have perimeter at most 4.
    """
    
    def solve_lp_radii(centers, W, H):
        """Solve LP for optimal radii given fixed centers and box size."""
        n = len(centers)
        c = -np.ones(n)
        A_ub = []
        b_ub = []
        
        # Pair constraints: r_i + r_j <= d_ij
        # For efficiency, we can precompute distances if centers don't change, 
        # but here centers are input, so we compute on the fly.
        for i in range(n):
            for j in range(i + 1, n):
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
                return res.x
        except Exception:
            pass
        return None

    def objective(centers_flat, W, H):
        """Objective function for minimization: negative sum of radii."""
        centers = centers_flat.reshape(-1, 2)
        # Ensure centers are within bounds (LP will handle radii, but centers should be reasonable)
        # If a center is outside, LP might fail or give 0 radii. 
        # We add a penalty if centers are outside to guide the optimizer.
        penalty = 0.0
        if np.any(centers[:, 0] < 0) or np.any(centers[:, 0] > W) or np.any(centers[:, 1] < 0) or np.any(centers[:, 1] > H):
            penalty = 100.0 + np.sum((np.clip(centers[:, 0], 0, W) - centers[:, 0])**2 + 
                                      (np.clip(centers[:, 1], 0, H) - centers[:, 1])**2)
        
        radii = solve_lp_radii(centers, W, H)
        if radii is None:
            return 1000.0 + penalty
        
        val = -np.sum(radii)
        return val + penalty

    def optimize_centers(W, H, initial_centers, max_iter=1000):
        """Optimize centers for fixed W, H using Nelder-Mead."""
        initial_flat = initial_centers.flatten()
        # Nelder-Mead parameters
        options = {
            'maxiter': max_iter,
            'xatol': 1e-5,
            'fatol': 1e-5,
            'adaptive': True
        }
        try:
            res = minimize(objective, initial_flat, args=(W, H), method='Nelder-Mead', options=options)
            if res.success or res.fun < 100: # Accept if it improved or is reasonable
                best_centers = res.x.reshape(-1, 2)
                # Clamp to box just in case
                best_centers[:, 0] = np.clip(best_centers[:, 0], 0, W)
                best_centers[:, 1] = np.clip(best_centers[:, 1], 0, H)
                radii = solve_lp_radii(best_centers, W, H)
                if radii is not None:
                    return np.sum(radii), best_centers, radii
        except Exception:
            pass
        return -np.inf, None, None

    # Generate initial configurations
    def generate_grid_config(rows, cols, W, H):
        """Generate a grid configuration scaled to W, H."""
        xs = np.linspace(W / (2*cols), W - W / (2*cols), cols)
        ys = np.linspace(H / (2*rows), H - H / (2*rows), rows)
        points = []
        for y in ys:
            for x in xs:
                points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            return points[:N]
        else:
            # Pad with random if needed (shouldn't happen for 21)
            pad = np.random.rand(N - len(points), 2) * np.array([W, H])
            return np.vstack([points, pad])

    def generate_random_config(W, H, seed):
        np.random.seed(seed)
        return np.random.rand(N, 2) * np.array([W, H])

    best_val_global = -np.inf
    best_result_global = None
    
    # Aspect ratios to try
    # W + H = 2. Let aspect = W/H. Then W = 2*aspect/(1+aspect), H = 2/(1+aspect)
    # Try a range of aspects
    aspects = np.linspace(0.5, 2.0, 30)
    
    # Initial configurations types
    # 1. 5x5 grid
    # 2. 4x6 grid (approx 24, take 21)
    # 3. 7x3 grid
    # 4. Random
    
    configs_generators = [
        lambda W, H: generate_grid_config(5, 5, W, H),
        lambda W, H: generate_grid_config(4, 6, W, H),
        lambda W, H: generate_grid_config(7, 3, W, H),
        lambda W, H: generate_random_config(W, H, 42),
        lambda W, H: generate_random_config(W, H, 123),
    ]
    
    for aspect in aspects:
        H = 2.0 / (aspect + 1.0)
        W = 2.0 * aspect / (aspect + 1.0)
        
        for gen in configs_generators:
            initial_centers = gen(W, H)
            
            val, centers, radii = optimize_centers(W, H, initial_centers, max_iter=500)
            
            if val > best_val_global:
                best_val_global = val
                result = np.zeros((N, 3))
                result[:, :2] = centers
                result[:, 2] = radii
                best_result_global = result

    if best_result_global is not None:
        return best_result_global
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
