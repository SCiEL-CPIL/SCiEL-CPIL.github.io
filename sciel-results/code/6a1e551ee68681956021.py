import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    d = 2
    start_time = time.time()
    max_time = 600.0
    safety_margin = 30.0

    def get_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def get_dmin_dmax(points):
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        return d_min, d_max

    def objective(points_flat):
        """Objective to minimize: -R = -(d_min/d_max)^2"""
        points = points_flat.reshape(n, d)
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 1.0
        R = (d_min / d_max) ** 2
        return -R

    def objective_grad(points_flat):
        """Gradient of -R using subgradients of d_min and d_max."""
        points = points_flat.reshape(n, d)
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-15:
            return np.zeros_like(points_flat)
        
        R = (d_min / d_max) ** 2
        
        # Find which pairs give d_min and d_max
        # pdist returns distances in a specific order for n points
        # We need to map back to point indices
        idx_min = np.argmin(dists)
        idx_max = np.argmax(dists)
        
        # Convert pdist index to (i, j) pair
        def pdist_to_pair(idx, n):
            # pdist uses compact format: distance between points i and j (i<j)
            # index = i*n - i*(i+1)/2 + (j - i - 1)
            # Solve for i, j
            k = idx
            i = 0
            while k >= n - i - 1:
                k -= (n - i - 1)
                i += 1
            j = i + 1 + k
            return i, j
        
        i_min, j_min = pdist_to_pair(idx_min, n)
        i_max, j_max = pdist_to_pair(idx_max, n)
        
        # Gradient of d_min with respect to points
        # d_min = ||p_i - p_j|| for the closest pair
        # ∂d_min/∂p_k = (p_i - p_j)/d_min if k==i, (p_j - p_i)/d_min if k==j, 0 otherwise
        # Gradient of d_max similarly
        
        grad = np.zeros_like(points_flat)
        
        if d_min > 1e-15:
            diff_min = points[i_min] - points[j_min]
            grad[i_min * d:(i_min+1) * d] += diff_min / d_min
            grad[j_min * d:(j_min+1) * d] -= diff_min / d_min
        
        if d_max > 1e-15:
            diff_max = points[i_max] - points[j_max]
            grad[i_max * d:(i_max+1) * d] += diff_max / d_max
            grad[j_max * d:(j_max+1) * d] -= diff_max / d_max
        
        # R = (d_min/d_max)^2
        # ∂R/∂x = 2*(d_min/d_max)^2 * (1/d_min * ∂d_min/∂x - 1/d_max * ∂d_max/∂x)
        # = 2*R * (grad_dmin/d_min - grad_dmax/d_max)
        
        # Actually, let's be more careful:
        # R = d_min^2 / d_max^2
        # ∂R/∂x = 2*d_min/d_max^2 * ∂d_min/∂x - 2*d_min^2/d_max^3 * ∂d_max/∂x
        # = 2*R/d_min * ∂d_min/∂x - 2*R/d_max * ∂d_max/∂x
        
        # We want gradient of -R
        if d_min > 1e-15 and d_max > 1e-15:
            factor_min = 2 * R / d_min
            factor_max = 2 * R / d_max
            
            # Recompute gradients properly
            grad_R = np.zeros_like(points_flat)
            
            diff_min = points[i_min] - points[j_min]
            grad_R[i_min * d:(i_min+1) * d] += diff_min / d_min
            grad_R[j_min * d:(j_min+1) * d] -= diff_min / d_min
            
            diff_max = points[i_max] - points[j_max]
            grad_R[i_max * d:(i_max+1) * d] += diff_max / d_max
            grad_R[j_max * d:(j_max+1) * d] -= diff_max / d_max
            
            grad_final = factor_min * grad_R  # This is wrong, need separate terms
            
            # Let me redo:
            # ∂R/∂x = (2*d_min/d_max^2) * ∂d_min/∂x - (2*d_min^2/d_max^3) * ∂d_max/∂x
            term1 = np.zeros_like(points_flat)
            term2 = np.zeros_like(points_flat)
            
            term1[i_min * d:(i_min+1) * d] += diff_min / d_min
            term1[j_min * d:(j_min+1) * d] -= diff_min / d_min
            
            term2[i_max * d:(i_max+1) * d] += diff_max / d_max
            term2[j_max * d:(j_max+1) * d] -= diff_max / d_max
            
            grad_R = (2 * d_min / d_max**2) * term1 - (2 * d_min**2 / d_max**3) * term2
            
            return -grad_R
        else:
            return np.zeros_like(points_flat)

    def simulated_annealing(points, iters=5000, T0=0.5, T_end=0.001):
        """Simulated annealing to escape local minima."""
        best = points.copy()
        best_score = get_score(best)
        current = points.copy()
        current_score = best_score
        
        for i in range(iters):
            if time.time() - start_time > max_time - safety_margin:
                break
            
            T = T0 * (T_end / T0) ** (i / iters)
            
            # Random perturbation
            idx = np.random.randint(n)
            perturbation = np.random.normal(0, T * 0.1, size=d)
            candidate = current.copy()
            candidate[idx] += perturbation
            
            cand_score = get_score(candidate)
            
            # Accept if better, or with probability exp((cand_score - current_score)/T)
            if cand_score > current_score:
                current = candidate
                current_score = cand_score
                if cand_score > best_score:
                    best = candidate.copy()
                    best_score = cand_score
            else:
                # Avoid division by zero in exponent
                delta = cand_score - current_score
                if delta < -100 * T:  # Too bad, reject
                    continue
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current = candidate
                    current_score = cand_score
        
        return best, best_score

    def polish(points):
        """Polish using scipy optimizer with gradient."""
        pts_flat = points.flatten()
        
        result = minimize(
            objective,
            pts_flat,
            jac=objective_grad,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8}
        )
        
        polished = result.x.reshape(n, d)
        return polished

    # Generate diverse initial configurations
    configs = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    configs.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. 4x4 grid
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    gx, gy = np.meshgrid(x, y)
    configs.append(np.column_stack((gx.flatten(), gy.flatten())))
    
    # 3. Hexagonal packing
    hex_pts = []
    for row in range(4):
        for col in range(4):
            xh = col * 1.0 + (0.5 if row % 2 == 1 else 0)
            yh = row * 0.866
            hex_pts.append([xh, yh])
    hex_pts = np.array(hex_pts)
    hex_pts -= hex_pts.mean(axis=0)
    configs.append(hex_pts)
    
    # 4. Two concentric squares (8+8) with rotation
    outer = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                      [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
    inner = outer * 0.5
    theta = np.pi / 4
    c, s = np.cos(theta), np.sin(theta)
    R_rot = np.array([[c, -s], [s, c]])
    inner = inner @ R_rot.T
    configs.append(np.vstack([outer, inner]))
    
    # 5. Concentric rings: 8 on outer, 8 on inner
    outer_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    inner_angles = outer_angles + np.pi/8
    outer_ring = np.column_stack((np.cos(outer_angles), np.sin(outer_angles))) * 1.5
    inner_ring = np.column_stack((np.cos(inner_angles), np.sin(inner_angles))) * 0.7
    configs.append(np.vstack([outer_ring, inner_ring]))
    
    # 6. Random with seed
    np.random.seed(12345)
    configs.append(np.random.randn(n, d) * 1.5)
    
    # 7. Rotated grid
    theta = np.pi / 8
    c, s = np.cos(theta), np.sin(theta)
    R_rot2 = np.array([[c, -s], [s, c]])
    grid_rot = configs[1] @ R_rot2.T
    configs.append(grid_rot)
    
    # 8. Square with points on edges and center
    sq_pts = []
    # 4 corners
    sq_pts.extend([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    # 4 edge midpoints
    sq_pts.extend([[0, -1], [1, 0], [0, 1], [-1, 0]])
    # 4 inner
    sq_pts.extend([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    # 4 more inner
    sq_pts.extend([[0, 0], [-0.5, 0], [0.5, 0], [0, -0.5]])
    configs.append(np.array(sq_pts, dtype=float))
    
    best_points_global = configs[0]
    best_score_global = get_score(configs[0])
    
    # For each config: SA -> Polish -> SA -> Polish
    for ci, config in enumerate(configs):
        if time.time() - start_time > max_time - safety_margin:
            break
        
        current = config.copy()
        
        # Phase 1: Simulated annealing
        sa_pts, sa_score = simulated_annealing(current, iters=3000, T0=0.3, T_end=0.005)
        if sa_score > best_score_global:
            best_score_global = sa_score
            best_points_global = sa_pts.copy()
        
        # Phase 2: Polish with gradient
        if time.time() - start_time > max_time - safety_margin:
            break
        polished = polish(sa_pts)
        pol_score = get_score(polished)
        if pol_score > best_score_global:
            best_score_global = pol_score
            best_points_global = polished.copy()
        
        # Phase 3: Another SA from polished
        if time.time() - start_time > max_time - safety_margin:
            break
        sa2_pts, sa2_score = simulated_annealing(polished, iters=3000, T0=0.1, T_end=0.001)
        if sa2_score > best_score_global:
            best_score_global = sa2_score
            best_points_global = sa2_pts.copy()
        
        # Phase 4: Final polish
        if time.time() - start_time > max_time - safety_margin:
            break
        final_polished = polish(sa2_pts)
        final_score = get_score(final_polished)
        if final_score > best_score_global:
            best_score_global = final_score
            best_points_global = final_polished.copy()
    
    return best_points_global
