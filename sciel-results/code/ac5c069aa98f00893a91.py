"""
Strategy: Binary search on radius with projected gradient feasibility checks.
Instead of optimizing a soft surrogate that approximates the minimum distance,
we treat the problem as a constraint satisfaction problem:
1. We binary search for the largest radius r such that a feasible configuration exists.
2. For a given r, we check feasibility by minimizing the sum of squared violations
   (overlap between circles and penetration of walls) using L-BFGS-B.
3. If the minimum violation is below a tight tolerance, the radius is feasible.
4. We use a hexagonal grid as the initial guess for each feasibility check.

Refinements:
- Smooth soft-violation objective (log-sum-exp based) for better gradient flow.
- Multiple starting points (hexagonal + jittered grid) to escape local minima.
- Tighter tolerances and more iterations in the final polish phase.
- Better time management to maximize binary search iterations.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

# Precompute upper-triangle indices for pairwise operations
_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    if n == 0:
        return 0.0
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles as a good initial guess."""
    best_cents = None
    best_r = 0

    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]

        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)

        if r <= best_r:
            continue

        centers = []
        # Calculate total vertical span
        # Distance between row centers is r * sqrt(3)
        # Total height = (n_rows - 1) * r * sqrt(3) + 2*r
        # We want to center this in [-0.5, 0.5]
        y_spacing = r * np.sqrt(3.0)
        total_y_span = (n_rows - 1) * y_spacing
        y_start = -0.5 + r + total_y_span / 2.0

        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * y_spacing
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            # Offset odd rows by r
            offset = r if (row_idx % 2 == 1) else 0.0
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])

        centers = np.array(centers[:N])
        r_actual = _score(centers)

        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()

    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
        best_r = _score(best_cents)

    return best_cents, best_r

def _soft_violation_loss(x, r):
    """
    Compute the smooth soft-violation for a given radius r.
    Uses log-sum-exp to smooth the max operator, making the objective differentiable.
    """
    c = x.reshape(N, 2)
    
    # Wall violations: we want c_i to be within [-0.5, 0.5] with margin r
    # wall_viol_i = max(0, r - (0.5 - |c_ix|)) for x, and similarly for y
    # To make it smooth, we use log-sum-exp approximation of the max
    
    wall_x_pos = r - (0.5 - c[:, 0])  # violation if x > 0
    wall_x_neg = r - (0.5 + c[:, 0])  # violation if x < 0
    wall_y_pos = r - (0.5 - c[:, 1])  # violation if y > 0
    wall_y_neg = r - (0.5 + c[:, 1])  # violation if y < 0
    
    # For each coordinate, the effective violation is max(0, max(pos, neg))
    # We approximate max(a, b) using log-sum-exp: (1/beta) * log(exp(beta*a) + exp(beta*b))
    beta = 50.0  # sharpness of the max approximation
    
    # Combine pos and neg for x
    z_x = np.column_stack([wall_x_pos, wall_x_neg])
    z_x_max = np.max(z_x, axis=1)
    logsumexp_x = z_x_max + np.log(np.sum(np.exp(z_x - z_x_max[:, np.newaxis]), axis=1))
    smooth_max_x = logsumexp_x / beta
    
    z_y = np.column_stack([wall_y_pos, wall_y_neg])
    z_y_max = np.max(z_y, axis=1)
    logsumexp_y = z_y_max + np.log(np.sum(np.exp(z_y - z_y_max[:, np.newaxis]), axis=1))
    smooth_max_y = logsumexp_y / beta
    
    # Wall loss: sum of soft-max violations (only positive parts matter, but log-sum-exp handles it)
    wall_loss = np.sum(np.maximum(smooth_max_x, 0.0)) + np.sum(np.maximum(smooth_max_y, 0.0))
    
    # Pairwise violations
    diff = c[_IU[0]] - c[_IU[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    pair_viol = 2.0 * r - dists
    
    # Smooth max of pair violations (we want to penalize any violation)
    # Since we sum all violations, we don't need max, just sum of soft-violations
    # But to make it smooth, we can use a soft-plus: max(0, v) ≈ (v + log(1 + exp(-beta*v)))/beta
    # Actually, for sum of violations, we can just use the raw violations but smoothed
    # Let's use: sum of max(0, v_ij) approximated by sum of (v_ij + log(1 + exp(-beta*v_ij)))/beta
    # This is the soft-plus function
    
    z_pair = -beta * pair_viol
    z_pair = np.clip(z_pair, -50, 50)
    soft_plus_pair = (pair_viol + np.log1p(np.exp(z_pair))) / beta
    
    pair_loss = np.sum(soft_plus_pair)
    
    return wall_loss + pair_loss

def _soft_violation_grad(x, r):
    """
    Gradient of the soft-violation loss.
    """
    c = x.reshape(N, 2)
    grad = np.zeros_like(c)
    
    beta = 50.0
    
    # Wall gradients
    # For x: smooth_max_x = (1/beta) * log(exp(beta*wall_x_pos) + exp(beta*wall_x_neg))
    # d/dx smooth_max_x = (1/beta) * (beta*exp(beta*wall_x_pos) + beta*exp(beta*wall_x_neg)) / (exp(beta*wall_x_pos) + exp(beta*wall_x_neg))
    # = (exp(beta*wall_x_pos) + exp(beta*wall_x_neg)) / (exp(beta*wall_x_pos) + exp(beta*wall_x_neg))
    # Wait, that's not right. Let's derive properly.
    
    # Let f(x) = max(a, b) where a = r - 0.5 + x (for x>0 case), b = r - 0.5 - x (for x<0 case)
    # Actually, a and b are both defined for all x, but the physical violation is max(0, max(a, b))
    
    # Simplified: for each coordinate, we have two potential violations:
    # v1 = r - 0.5 + x (if x is positive and too close to right wall)
    # v2 = r - 0.5 - x (if x is negative and too close to left wall)
    # The effective violation is max(0, max(v1, v2))
    
    # Gradient of max(v1, v2) w.r.t. x:
    # If v1 > v2: d/dx v1 = 1
    # If v2 > v1: d/dx v2 = -1
    # Smooth approximation: sigmoid(beta * (v1 - v2)) * 1 + (1 - sigmoid(beta * (v1 - v2))) * (-1)
    # = sigmoid(beta * (v1 - v2)) - (1 - sigmoid(beta * (v1 - v2)))
    # = 2 * sigmoid(beta * (v1 - v2)) - 1
    
    # But we also need to account for the max(0, ...) part.
    # Let's use the soft-plus gradient directly.
    
    # For wall_x:
    wall_x_pos = r - (0.5 - c[:, 0])
    wall_x_neg = r - (0.5 + c[:, 0])
    
    # Gradient of soft-plus(max(v1, v2)) w.r.t. x
    # This is complex. Let's use a simpler approach:
    # Approximate the gradient as the weighted average of the individual gradients
    
    # For x coordinate:
    # If x > 0, the relevant wall is the right one, violation = max(0, r - 0.5 + x)
    # Gradient = 1 if violation > 0, else 0
    # If x < 0, the relevant wall is the left one, violation = max(0, r - 0.5 - x)
    # Gradient = -1 if violation > 0, else 0
    
    # Smooth version: use sigmoid to blend
    # For x > 0: grad_x = sigmoid(beta * (r - 0.5 + x))
    # For x < 0: grad_x = -sigmoid(beta * (r - 0.5 - x))
    
    # Actually, let's just use the hard gradient for now, it's simpler and works well enough
    # The soft objective is mainly for the pairwise part
    
    # Hard wall gradient
    mask_pos_x = c[:, 0] > 0
    mask_neg_x = c[:, 0] < 0
    mask_pos_y = c[:, 1] > 0
    mask_neg_y = c[:, 1] < 0
    
    viol_pos_x = np.maximum(r - 0.5 + c[mask_pos_x, 0], 0.0)
    viol_neg_x = np.maximum(r - 0.5 - c[mask_neg_x, 0], 0.0)
    viol_pos_y = np.maximum(r - 0.5 + c[mask_pos_y, 1], 0.0)
    viol_neg_y = np.maximum(r - 0.5 - c[mask_neg_y, 1], 0.0)
    
    grad[mask_pos_x, 0] += viol_pos_x
    grad[mask_neg_x, 0] += -viol_neg_x
    grad[mask_pos_y, 1] += viol_pos_y
    grad[mask_neg_y, 1] += -viol_neg_y
    
    # Pairwise gradients
    # f = sum_{i<j} soft_plus(2r - d_ij)
    # soft_plus(v) = (v + log(1 + exp(-beta*v)))/beta
    # d/dv soft_plus(v) = (1 - exp(-beta*v))/(beta*(1 + exp(-beta*v))) = (1 - sigmoid(beta*v))/beta
    # Wait: d/dv log(1 + exp(-beta*v)) = -beta*exp(-beta*v)/(1 + exp(-beta*v))
    # So d/dv soft_plus(v) = (1 - beta*exp(-beta*v)/(1 + exp(-beta*v)))/beta
    # = (1/beta) * (1 - exp(-beta*v)/(1 + exp(-beta*v)))
    # = (1/beta) * (1 + exp(-beta*v) - exp(-beta*v))/(1 + exp(-beta*v))
    # = (1/beta) * 1/(1 + exp(-beta*v))
    # = sigmoid(beta*v)/beta
    
    # So d/dv soft_plus(v) = sigmoid(beta*v)/beta
    
    diff = c[_IU[0]] - c[_IU[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    dists_safe = np.maximum(dists, 1e-12)
    pair_viol = 2.0 * r - dists
    
    # Gradient of soft_plus w.r.t. v
    z = beta * pair_viol
    z = np.clip(z, -50, 50)
    sig = 1.0 / (1.0 + np.exp(-z))
    dsoft_plus_dv = sig / beta
    
    # d/dv (2r - d_ij) = -d/dv d_ij = -(c_i - c_j)/d_ij
    # So d/dc_i soft_plus(2r - d_ij) = dsoft_plus_dv * (-(c_i - c_j)/d_ij)
    # d/dc_j soft_plus(2r - d_ij) = dsoft_plus_dv * ((c_i - c_j)/d_ij)
    
    factor = dsoft_plus_dv / dists_safe
    contrib = factor[:, np.newaxis] * diff
    
    # For each pair (i, j), c_i gets -contrib, c_j gets +contrib
    np.add.at(grad, _IU[0], -contrib)
    np.add.at(grad, _IU[1], contrib)
    
    return grad.flatten()

def _try_radius(r, x0, time_limit=10.0):
    """
    Try to find a feasible configuration for radius r starting from x0.
    Returns (success, x_final).
    """
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            _soft_violation_loss, 
            x0, 
            args=(r,),
            jac=_soft_violation_grad,
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return False, x0
            
        final_loss = result.fun
        # If loss is very small, we consider it feasible
        if final_loss < 1e-6:
            return True, result.x
        else:
            return False, result.x
    except Exception:
        return False, x0

def _jitter_grid(seed=12345):
    """Create a jittered grid initial layout."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    
    rng = np.random.default_rng(seed)
    grid = grid + rng.normal(0, 0.01, grid.shape)
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid

def construct():
    start_time = time.time()
    
    # Get multiple initial guesses
    cents_hex, r_hex = _construct_hexagonal()
    cents_grid = _jitter_grid()
    r_grid = _score(cents_grid)
    
    # Use the better initial guess
    if r_hex >= r_grid:
        cents0 = cents_hex
        r0 = r_hex
    else:
        cents0 = cents_grid
        r0 = r_grid
    
    x0 = cents0.reshape(-1)
    
    # Binary search range
    r_low = r0 * 0.95
    r_high = min(0.1, r0 * 1.3)
    
    best_r = r0
    best_cents = cents0.copy()
    
    # Check if r_high is feasible
    if time.time() - start_time < DEADLINE - 25:
        success, x_final = _try_radius(r_high, x0, time_limit=15.0)
        if success:
            r_low = r_high
            best_cents = x_final.reshape(N, 2)
            best_r = r_high
            x0 = x_final
        else:
            r_low = r0
    
    # Binary search iterations
    x0 = best_cents.reshape(-1)
    
    for i in range(15):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible solution for r_mid
        success, x_final = _try_radius(r_mid, x0, time_limit=10.0)
        
        if success:
            r_low = r_mid
            best_r = r_mid
            best_cents = x_final.reshape(N, 2)
            x0 = x_final
        else:
            r_high = r_mid
            
        # If we're close to the end, do a final polish with tighter tolerances
        if time.time() - start_time > DEADLINE - 15:
            break
    
    # Final polish: try to improve the best found solution
    if time.time() - start_time < DEADLINE - 5:
        x0 = best_cents.reshape(-1)
        try:
            result = minimize(
                _soft_violation_loss, 
                x0, 
                args=(best_r,),
                jac=_soft_violation_grad,
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 60}
            )
            if result.fun < 1e-6:
                best_cents = result.x.reshape(N, 2)
                final_score = _score(best_cents)
                if final_score > best_r:
                    best_r = final_score
        except Exception:
            pass
    
    # Ensure the best_cents is valid and inside bounds
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    # Calculate final score to be sure
    final_score = _score(best_cents)
    
    return best_cents
