"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _penalty_loss(x, radii, eps=1e-4):
    """
    x: (2*N,) flattened centers
    radii: (N,) fixed radii
    Returns scalar loss: sum of soft violations (boundary + overlaps)
    """
    centers = x.reshape(N, 2)
    # Boundary violations: center must be within [r, 1-r]
    lo_viol = np.maximum(0, radii - centers)
    hi_viol = np.maximum(0, centers + radii - 1.0)
    boundary = np.sum(lo_viol**2) + np.sum(hi_viol**2)
    
    # Overlap violations: for each pair, distance < r_i + r_j
    # dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(N, k=1)
    d = centers[i_idx] - centers[j_idx]
    dist2 = np.sum(d**2, axis=1)
    min_dist = radii[i_idx] + radii[j_idx]
    # violation = max(0, min_dist - dist)^2, use soft version
    diff = min_dist**2 - dist2
    soft_viol = np.maximum(0, diff) ** 2
    overlap = np.sum(soft_viol)
    
    return boundary + overlap

def _penalty_grad(x, radii, eps=1e-4):
    """
    Gradient of penalty loss.
    """
    centers = x.reshape(N, 2)
    grad = np.zeros_like(x)
    
    # Boundary gradients
    lo_viol = np.maximum(0, radii - centers)
    hi_viol = np.maximum(0, centers + radii - 1.0)
    # d/dx of (r - x)^2 = -2(r-x) when r-x>0
    grad[:N] += -2.0 * lo_viol  # x component
    grad[N:] += -2.0 * lo_viol  # y component
    # d/dx of (x + r - 1)^2 = 2(x+r-1) when x+r-1>0
    grad[:N] += 2.0 * hi_viol
    grad[N:] += 2.0 * hi_viol
    
    # Overlap gradients
    i_idx, j_idx = np.triu_indices(N, k=1)
    d = centers[i_idx] - centers[j_idx]
    dist2 = np.sum(d**2, axis=1)
    min_dist = radii[i_idx] + radii[j_idx]
    diff = min_dist**2 - dist2
    active = diff > 0
    if np.any(active):
        # soft_viol = diff^2, d/dx diff = d/dx (min_dist^2 - dist^2) = -2*d
        # d/dx soft_viol = 2*diff * (-2*d) = -4*diff*d
        factor = -4.0 * diff[active]
        d_active = d[active]
        # Accumulate into i and j
        for k in range(len(i_idx[active])):
            ii = i_idx[active][k]
            jj = j_idx[active][k]
            grad[ii] += factor[k] * d_active[k]
            grad[jj] -= factor[k] * d_active[k]
    
    return grad

def _grow_radii(centers, radii, max_iter=200, tol=1e-6):
    """
    Given fixed centers, try to grow radii subject to non-overlap and boundary.
    Uses a simple iterative approach: repeatedly try to increase each radius
    by a small amount if it doesn't violate constraints, or use optimization.
    """
    radii = radii.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Max possible radius for circle i given others
            max_r = min(centers[i][0], centers[i][1], 1 - centers[i][0], 1 - centers[i][1])
            for j in range(N):
                if i == j:
                    continue
                d = np.linalg.norm(centers[i] - centers[j])
                max_r = min(max_r, (d - radii[j]) / 2.0)
            if max_r > radii[i] + tol:
                radii[i] = max_r
                improved = True
        if not improved:
            break
    return radii

def _slsqp_polish(x, radii, time_limit):
    """
    Final polish using SLSQP with explicit constraints.
    """
    start = time.time()
    
    def obj(x):
        return -np.sum(radii)  # placeholder, we optimize positions with fixed radii? 
        # Actually, for polish, let's optimize both positions and radii
        # But SLSQP with many constraints is slow. Let's just do position optimization
        # with current radii, then grow radii again.
        pass
    
    # For simplicity, do a few L-BFGS-B steps with penalty, then grow radii
    result = minimize(
        _penalty_loss, x, jac=_penalty_grad, args=(radii,),
        method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    return result.x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: grid-based
    side = int(np.ceil(np.sqrt(N)))
    centers = []
    radii_init = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            centers.append([cx, cy])
            radii_init.append(0.5 / side - 1e-3)
    centers = np.array(centers[:N])
    radii = np.array(radii_init[:N])
    
    x0 = centers.flatten()
    
    best_x = x0.copy()
    best_radii = radii.copy()
    best_score = np.sum(radii)
    
    # Phase 1: L-BFGS-B with penalty to find valid positions
    # Use multiple restarts with different initial conditions
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if restart == 0:
            x_curr = x0.copy()
        else:
            # Perturb initial positions
            x_curr = x0 + rng.randn(2 * N) * 0.05
            # Clamp to valid range
            for i in range(N):
                x_curr[2*i] = np.clip(x_curr[2*i], radii[i], 1 - radii[i])
                x_curr[2*i+1] = np.clip(x_curr[2*i+1], radii[i], 1 - radii[i])
        
        result = minimize(
            _penalty_loss, x_curr, jac=_penalty_grad, args=(radii,),
            method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        
        if result.success or result.fun < _penalty_loss(best_x, radii):
            best_x = result.x
            best_score = np.sum(radii)  # radii haven't changed yet
    
    # Phase 2: Grow radii
    centers_curr = best_x.reshape(N, 2)
    radii_grown = _grow_radii(centers_curr, radii)
    
    # Phase 3: Polish with L-BFGS-B using grown radii
    if time.time() < deadline - 10:
        result = minimize(
            _penalty_loss, best_x, jac=_penalty_grad, args=(radii_grown,),
            method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        best_x = result.x
        centers_curr = best_x.reshape(N, 2)
        radii_grown = _grow_radii(centers_curr, radii_grown)
    
    # Final validation and output
    centers_final = best_x.reshape(N, 2)
    radii_final = radii_grown
    
    # Ensure validity: clamp and check
    for i in range(N):
        radii_final[i] = max(0, min(radii_final[i], 
                                     centers_final[i][0], centers_final[i][1],
                                     1 - centers_final[i][0], 1 - centers_final[i][1]))
    
    # Check for overlaps and shrink if necessary
    for _ in range(10):
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers_final[i] - centers_final[j])
                if d < radii_final[i] + radii_final[j] - 1e-9:
                    valid = False
                    # Shrink both
                    scale = d / (radii_final[i] + radii_final[j]) * 0.99
                    radii_final[i] *= scale
                    radii_final[j] *= scale
        if valid:
            break
    
    out = np.column_stack([centers_final, radii_final])
    return out
# EVOLVE-BLOCK-END
