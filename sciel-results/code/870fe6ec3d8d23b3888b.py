"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 21
DEADLINE = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_radii(centers, w):
    """Compute maximum radii for given centers and width."""
    h = 2.0 - w
    n = len(centers)
    r = np.minimum(np.minimum(centers[:, 0], w - centers[:, 0]), 
                   np.minimum(centers[:, 1], h - centers[:, 1]))
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < r[i] + r[j]:
                    deficit = r[i] + r[j] - d
                    if deficit > 1e-12:
                        reduce_i = deficit / 2.0
                        reduce_j = deficit / 2.0
                        if r[i] < reduce_i:
                            reduce_j += (reduce_i - r[i])
                            reduce_i = r[i]
                        if r[j] < reduce_j:
                            reduce_i += (reduce_j - r[j])
                            reduce_j = r[j]
                        r[i] -= reduce_i
                        r[j] -= reduce_j
                        changed = True
        if not changed:
            break
    return r

def _smooth_objective(x, w, temperature=0.01):
    """
    Smooth objective: minimize -sum(radii) using log-sum-exp approximation.
    x: (2N,) array of centers [x1, y1, ..., xN, yN]
    Returns (value, gradient)
    """
    n = N
    h = 2.0 - w
    centers = x.reshape(n, 2)
    
    # Compute distances to boundaries
    d_left = centers[:, 0]
    d_right = w - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = h - centers[:, 1]
    
    # Compute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = np.linalg.norm(centers[i] - centers[j])
    
    # For each circle, compute the "soft minimum" of all constraints
    # Using log-sum-exp: soft_min(a, b, c, ...) = -log(exp(-a/T) + exp(-b/T) + ...) * T
    # Actually: soft_min = -T * log(sum(exp(-constraint/T)))
    
    # Build constraint arrays for each circle
    # Constraint: distance to boundary (4 per circle) + distance to other circles (n-1 per circle)
    
    # We'll compute a smooth approximation of min(all_constraints)
    # r_i ≈ -T * log(sum_j exp(-c_ij / T)) where c_ij are the constraints for circle i
    
    radii_smooth = np.zeros(n)
    grad = np.zeros(2 * n)
    
    for i in range(n):
        # Constraints for circle i
        constraints = []
        # Boundary constraints
        constraints.extend([d_left[i], d_right[i], d_bottom[i], d_top[i]])
        # Circle-circle constraints
        for j in range(n):
            if i != j:
                constraints.append(dists[i, j])
        
        constraints = np.array(constraints)
        
        # Log-sum-exp for soft minimum
        # soft_min = -T * log(sum(exp(-c/T)))
        exp_terms = np.exp(-constraints / temperature)
        log_sum = np.log(np.sum(exp_terms))
        r_i_smooth = -temperature * log_sum
        
        radii_smooth[i] = r_i_smooth
        
        # Gradient with respect to center i
        # d(r_i_smooth)/d(x_i) = -T * d/dx_i [log(sum_j exp(-c_ij/T))]
        # = -T * [sum_j exp(-c_ij/T) * (-1/T) * dc_ij/dx_i] / sum_j exp(-c_ij/T)
        # = sum_j exp(-c_ij/T) * dc_ij/dx_i / sum_j exp(-c_ij/T)
        # = sum_j w_ij * dc_ij/dx_i where w_ij = exp(-c_ij/T) / sum_k exp(-c_ik/T)
        
        weights = exp_terms / np.sum(exp_terms)
        
        # Compute gradient contributions
        gx = 0.0
        gy = 0.0
        
        # Boundary gradients
        # d_left = x_i, d/dx_i = 1
        gx += weights[0] * 1.0
        # d_right = w - x_i, d/dx_i = -1
        gx += weights[1] * (-1.0)
        # d_bottom = y_i, d/dy_i = 1
        gy += weights[2] * 1.0
        # d_top = h - y_i, d/dy_i = -1
        gy += weights[3] * (-1.0)
        
        # Circle-circle gradients
        idx = 4
        for j in range(n):
            if i != j:
                # c_ij = dist(i, j) = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
                # dc_ij/dx_i = (x_i - x_j) / dist(i,j)
                if dists[i, j] > 1e-12:
                    gx += weights[idx] * (centers[i, 0] - centers[j, 0]) / dists[i, j]
                    gy += weights[idx] * (centers[i, 1] - centers[j, 1]) / dists[i, j]
                idx += 1
        
        grad[2*i] = gx
        grad[2*i+1] = gy
    
    # Objective: maximize sum(radii_smooth) => minimize -sum(radii_smooth)
    value = -np.sum(radii_smooth)
    grad = -grad  # Negate because we're minimizing
    
    return value, grad

def _optimize_positions(w, initial_centers, max_iter=1000, temperature=0.01):
    """Optimize positions for fixed w using L-BFGS-B."""
    h = 2.0 - w
    n = N
    
    # Initial point
    x0 = initial_centers.flatten()
    
    # Bounds: centers must be in [0, w] x [0, h]
    bounds = []
    for i in range(n):
        bounds.append((1e-6, w - 1e-6))
        bounds.append((1e-6, h - 1e-6))
    
    # Optimize
    result = minimize(
        lambda x: _smooth_objective(x, w, temperature)[0],
        x0,
        jac=lambda x: _smooth_objective(x, w, temperature)[1],
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    return result.x.reshape(n, 2)

def _grow_radii(centers, w, max_iter=50, growth_rate=0.02):
    """
    Iteratively grow radii and re-optimize positions.
    """
    current_centers = centers.copy()
    current_radii = _compute_radii(current_centers, w)
    
    for iteration in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        
        # Try to grow all radii uniformly
        growth = growth_rate
        new_radii = current_radii + growth
        
        # Check if this is feasible by optimizing positions
        # We'll use the smooth objective with a target radius constraint
        
        # For now, just re-optimize positions and see if radii can be maintained
        new_centers = _optimize_positions(w, current_centers, max_iter=200, temperature=0.01)
        new_computed_radii = _compute_radii(new_centers, w)
        
        # If we can maintain or improve, accept
        if np.sum(new_computed_radii) >= np.sum(current_radii) - 1e-6:
            current_centers = new_centers
            current_radii = new_computed_radii
        else:
            # Reduce growth and try again
            growth *= 0.5
            if growth < 1e-4:
                break
    
    return current_centers, current_radii

def _is_valid(centers, radii, w, tol=1e-7):
    """Check if packing is valid."""
    h = 2.0 - w
    n = len(centers)
    
    # Check boundaries
    for i in range(n):
        if centers[i, 0] < -tol or centers[i, 0] > w + tol:
            return False
        if centers[i, 1] < -tol or centers[i, 1] > h + tol:
            return False
        if radii[i] < -tol:
            return False
    
    # Check circle-circle
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j] - tol:
                return False
    
    return True

def construct():
    start_time = time.time()
    best_score = -1
    best_centers = None
    best_radii = None
    best_w = 1.0
    
    # Try multiple aspect ratios
    w_values = np.linspace(0.6, 1.4, 9)
    
    for w in w_values:
        if time.time() > DEADLINE - 10:
            break
        
        h = 2.0 - w
        
        # Initialize with grid
        side = int(np.ceil(np.sqrt(N)))
        dx = w / side
        dy = h / side
        
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) >= N:
                    break
                x = (i + 0.5) * dx
                y = (j + 0.5) * dy
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        
        # Optimize positions
        optimized_centers = _optimize_positions(w, centers, max_iter=500, temperature=0.01)
        
        # Compute radii
        radii = _compute_radii(optimized_centers, w)
        score = np.sum(radii)
        
        # Grow radii iteratively
        if time.time() < DEADLINE - 20:
            grown_centers, grown_radii = _grow_radii(optimized_centers, w, max_iter=30, growth_rate=0.01)
            grown_score = np.sum(grown_radii)
            if grown_score > score:
                optimized_centers = grown_centers
                radii = grown_radii
                score = grown_score
        
        # Validate
        if _is_valid(optimized_centers, radii, w):
            if score > best_score:
                best_score = score
                best_centers = optimized_centers.copy()
                best_radii = radii.copy()
                best_w = w
    
    # Fallback if no valid solution found
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final polish: one more optimization round
    if time.time() < DEADLINE - 15:
        final_centers = _optimize_positions(best_w, best_centers, max_iter=300, temperature=0.005)
        final_radii = _compute_radii(final_centers, best_w)
        if _is_valid(final_centers, final_radii, best_w) and np.sum(final_radii) > best_score:
            best_centers = final_centers
            best_radii = final_radii
            best_score = np.sum(final_radii)
    
    # Format output: (x, y, r)
    result = np.column_stack((best_centers, best_radii))
    return result
# EVOLVE-BLOCK-END
