import numpy as np
import time
from scipy.optimize import minimize

N = 16
D = 2
DEADLINE = 115.0  # Safety margin before 120s limit

def _pairwise_dists_sq(pts):
    """Compute squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    return d2[iu]

def _exact_score(pts):
    """Compute exact score: (max_dist / min_dist)^2 = max_d2 / min_d2."""
    d2 = _pairwise_dists_sq(pts)
    min_d2 = np.min(d2)
    max_d2 = np.max(d2)
    if min_d2 < 1e-12:
        return 1e10
    return max_d2 / min_d2

def _soft_objective(params, temp=0.01):
    """
    Soft objective using log-sum-exp for smooth min/max.
    Returns soft_max / soft_min of squared distances.
    """
    pts = params.reshape(N, D)
    d2 = _pairwise_dists_sq(pts)
    d2 = np.maximum(d2, 1e-12)
    
    m_max = np.max(d2)
    # Soft max: m + temp * log(sum(exp((d - m)/temp)))
    soft_max = m_max + temp * np.log(np.sum(np.exp((d2 - m_max) / temp)))
    
    m_min = np.min(d2)
    # Soft min: m - temp * log(sum(exp(-(d - m)/temp)))
    soft_min = m_min - temp * np.log(np.sum(np.exp(-(d2 - m_min) / temp)))
    
    if soft_min < 1e-9:
        return 1e10
        
    return soft_max / soft_min

def _generate_grid_pts():
    """Generate 4x4 grid points."""
    gx = np.array([0, 1, 2, 3], dtype=float)
    gy = np.array([0, 1, 2, 3], dtype=float)
    X, Y = np.meshgrid(gx, gy)
    return np.column_stack([X.flatten(), Y.flatten()])

def _generate_hex_pts():
    """Generate 16 points from hexagonal lattice (subset of 19-point cluster)."""
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    
    # We have 19 points. Remove 3 from the outer ring to get 16.
    # Identify outer ring (dist^2 == 4)
    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    # Simply take the first 16 points? No, that might be unbalanced.
    # Better: Remove 3 specific outer points to maintain symmetry if possible.
    # For now, just remove the first 3 outer indices found.
    if len(outer_indices) >= 3:
        remove_idx = list(outer_indices[:3])
        keep_idx = [i for i in range(len(hex_points)) if i not in remove_idx]
        return hex_points[keep_idx]
    
    return hex_points[:16]

def _generate_circles_pts():
    """Two concentric circles: 8 inner, 8 outer."""
    pts = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts.append([1 * np.cos(ang), 1 * np.sin(ang)])
    return np.array(pts)

def _generate_spiral_pts():
    """Golden angle spiral."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    pts = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt(i + 1)
        theta = i * golden_angle
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
    return pts

def construct():
    start_time = time.time()
    
    candidates = []
    
    # 1. 4x4 Grid
    candidates.append(_generate_grid_pts())
    
    # 2. Hexagonal Lattice Subset
    candidates.append(_generate_hex_pts())
    
    # 3. Concentric Circles
    candidates.append(_generate_circles_pts())
    
    # 4. Spiral
    candidates.append(_generate_spiral_pts())
    
    # 5. Random Perturbations of Grid
    rng = np.random.default_rng(42)
    for _ in range(5):
        perturbed = candidates[0] + rng.normal(0, 0.1, (N, D))
        candidates.append(perturbed)
        
    # 6. Random Perturbations of Hex
    for _ in range(5):
        perturbed = candidates[1] + rng.normal(0, 0.1, (N, D))
        candidates.append(perturbed)

    best_pts = None
    best_score = np.inf

    # Optimization Strategy:
    # For each candidate:
    # 1. Coarse L-BFGS with high temp
    # 2. Fine L-BFGS with lower temp
    # 3. Final L-BFGS with very low temp
    # Then global final polish on the best result.

    temps = [0.1, 0.01, 0.001]
    
    for i, init_pts in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        current_params = init_pts.flatten()
        
        try:
            for temp in temps:
                if time.time() - start_time > DEADLINE - 5:
                    break
                res = minimize(
                    lambda p, t=temp: _soft_objective(p, t),
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_params = res.x
                
            pts = current_params.reshape(N, D)
            score = _exact_score(pts)
            
            if score < best_score:
                best_score = score
                best_pts = pts
                
        except Exception:
            continue

    # Final high-precision polish on the best candidate found
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        try:
            # Use a very low temperature for the final polish
            res_final = minimize(
                lambda p: _soft_objective(p, 1e-4),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts_final = res_final.x.reshape(N, D)
            score_final = _exact_score(pts_final)
            
            if score_final < best_score:
                best_pts = pts_final
                best_score = score_final
        except Exception:
            pass

    if best_pts is None:
        best_pts = candidates[0]

    return best_pts

if __name__ == '__main__':
    pts = construct()
    print("Score:", _exact_score(pts))
