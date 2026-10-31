"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time
import itertools
from scipy.spatial import Delaunay

def _pairwise_distances(points):
    """Compute pairwise distance matrix for n points."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def _feasibility_penalty(x, n, w=100.0):
    """Penalty for constraint violations."""
    pts = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]
    
    # Boundary violations
    pen = 0.0
    for i in range(n):
        if pts[i][0] < radii[i]:
            pen += w * (radii[i] - pts[i][0]) ** 2
        if pts[i][0] > 1 - radii[i]:
            pen += w * (pts[i][0] - (1 - radii[i])) ** 2
        if pts[i][1] < radii[i]:
            pen += w * (radii[i] - pts[i][1]) ** 2
        if pts[i][1] > 1 - radii[i]:
            pen += w * (pts[i][1] - (1 - radii[i])) ** 2
    
    # Overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                pen += w * (min_dist - dist) ** 2
    
    return pen

def _objective(x, n):
    """Negative sum of radii plus penalty."""
    radii = x[2 * n:]
    return -np.sum(radii) + _feasibility_penalty(x, n, w=50.0)

def _is_feasible(pts, radii, tol=1e-6):
    """Check if configuration is feasible."""
    for i in range(len(radii)):
        if pts[i][0] < radii[i] - tol or pts[i][0] > 1 - radii[i] + tol:
            return False
        if pts[i][1] < radii[i] - tol or pts[i][1] > 1 - radii[i] + tol:
            return False
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            dist = np.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _sum_radii(radii):
    return float(np.sum(radii))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n = N
    best_solution = None
    best_value = -np.inf
    
    # Multiple random restarts with different initial configurations
    np.random.seed(42)
    
    # Strategy 1: Grid-based initialization with perturbations
    side = int(np.ceil(np.sqrt(n)))
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Create initial grid positions
        pts = []
        for i in range(side):
            for j in range(side):
                if len(pts) >= n:
                    break
                base_x = (i + 0.5) / side
                base_y = (j + 0.5) / side
                # Add random perturbation
                px = base_x + np.random.uniform(-0.05, 0.05)
                py = base_y + np.random.uniform(-0.05, 0.05)
                px = np.clip(px, 0.05, 0.95)
                py = np.clip(py, 0.05, 0.95)
                pts.append([px, py])
        
        pts = np.array(pts[:n])
        
        # Initial radii: small, will be grown
        base_r = 0.5 / side
        radii = np.full(n, base_r * 0.8)
        
        # Build initial x vector
        x0 = np.concatenate([pts.flatten(), radii])
        
        # Optimize using L-BFGS-B with penalty method
        try:
            res = minimize(_objective, x0, args=(n,), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if time.time() > deadline - 15:
                break
            
            # Extract solution
            pts_opt = res.x[:2 * n].reshape(n, 2)
            radii_opt = res.x[2 * n:]
            
            # Ensure radii are positive
            radii_opt = np.maximum(radii_opt, 1e-6)
            
            # Check feasibility and compute value
            if _is_feasible(pts_opt, radii_opt, tol=1e-5):
                val = _sum_radii(radii_opt)
                if val > best_value:
                    best_value = val
                    best_solution = (pts_opt.copy(), radii_opt.copy())
        except Exception as e:
            pass
    
    # Strategy 2: Use Delaunay-based neighbor identification for targeted optimization
    if best_solution is not None and time.time() < deadline - 20:
        pts, radii = best_solution
        try:
            # Find Delaunay triangulation to identify close pairs
            tri = Delaunay(pts)
            
            # For each triangle, we can try to push circles apart or grow them
            # Simple local refinement: for each pair of nearby circles, try to optimize
            improved = True
            max_iters = 5
            iter_count = 0
            while improved and iter_count < max_iters and time.time() < deadline - 10:
                improved = False
                iter_count += 1
                
                # Try growing each radius slightly
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    
                    trial_radii = radii.copy()
                    trial_radii[i] *= 1.02  # Try 2% growth
                    
                    if _is_feasible(pts, trial_radii, tol=1e-5):
                        radii = trial_radii
                        improved = True
                    else:
                        # Try smaller growth
                        for factor in [1.01, 1.005, 1.001]:
                            trial_radii = radii.copy()
                            trial_radii[i] = radii[i] * factor
                            if _is_feasible(pts, trial_radii, tol=1e-5):
                                radii = trial_radii
                                improved = True
                                break
                
                # Try moving points to create more space
                for i in range(n):
                    if time.time() > deadline - 10:
                        break
                    
                    # Try small random moves
                    for _ in range(5):
                        dx = np.random.uniform(-0.02, 0.02)
                        dy = np.random.uniform(-0.02, 0.02)
                        trial_pts = pts.copy()
                        trial_pts[i][0] += dx
                        trial_pts[i][1] += dy
                        
                        # Check bounds
                        if (trial_pts[i][0] < radii[i] or trial_pts[i][0] > 1 - radii[i] or
                            trial_pts[i][1] < radii[i] or trial_pts[i][1] > 1 - radii[i]):
                            continue
                        
                        if _is_feasible(trial_pts, radii, tol=1e-5):
                            # Check if this allows any radius growth
                            new_radii = radii.copy()
                            for j in range(n):
                                # Try to grow j slightly
                                for factor in [1.01, 1.005]:
                                    trial_r = new_radii.copy()
                                    trial_r[j] = new_radii[j] * factor
                                    if _is_feasible(trial_pts, trial_r, tol=1e-5):
                                        new_radii[j] = trial_r[j]
                                        break
                            
                            if _sum_radii(new_radii) > _sum_radii(radii) + 1e-8:
                                pts = trial_pts
                                radii = new_radii
                                improved = True
                                break
        
        except Exception as e:
            pass
    
    # Final polish: SLSQP with tighter constraints
    if best_solution is not None and time.time() < deadline - 5:
        pts, radii = best_solution
        try:
            x0 = np.concatenate([pts.flatten(), radii])
            
            # Define constraints for SLSQP
            def constraints_slsqp(x):
                p = x[:2 * n].reshape(n, 2)
                r = x[2 * n:]
                cons = []
                
                # Boundary constraints
                for i in range(n):
                    cons.append(p[i][0] - r[i])
                    cons.append(1 - p[i][0] - r[i])
                    cons.append(p[i][1] - r[i])
                    cons.append(1 - p[i][1] - r[i])
                
                # Non-overlap constraints
                for i in range(n):
                    for j in range(i + 1, n):
                        dist_sq = (p[i][0] - p[j][0]) ** 2 + (p[i][1] - p[j][1]) ** 2
                        min_dist_sq = (r[i] + r[j]) ** 2
                        cons.append(dist_sq - min_dist_sq)
                
                return cons
            
            # Use SLSQP for final polish
            res2 = minimize(lambda x: -np.sum(x[2 * n:]), x0, method='SLSQP',
                           constraints=[{'type': 'ineq', 'fun': constraints_slsqp}],
                           options={'maxiter': 100, 'ftol': 1e-12})
            
            if res2.success and _is_feasible(res2.x[:2 * n].reshape(n, 2), res2.x[2 * n:], tol=1e-6):
                val = _sum_radii(res2.x[2 * n:])
                if val > best_value:
                    best_value = val
                    best_solution = (res2.x[:2 * n].reshape(n, 2).copy(), res2.x[2 * n:].copy())
        except Exception as e:
            pass
    
    # Construct final output
    if best_solution is None:
        # Fallback to trivial grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:n])
    
    pts, radii = best_solution
    result = np.zeros((n, 3))
    result[:, 0] = pts[:, 0]
    result[:, 1] = pts[:, 1]
    result[:, 2] = radii
    return result

# EVOLVE-BLOCK-END
