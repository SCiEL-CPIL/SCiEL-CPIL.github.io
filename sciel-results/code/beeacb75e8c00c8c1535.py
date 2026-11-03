import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    # Try a few different spacings and rotations
    for attempt in range(14):
        # Spacing estimation
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.015 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        # Use a fixed rotation per attempt to break symmetry
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Apply rotation
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]

        # Add noise for attempts > 0
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.04, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]
            
        # Ensure exactly N points
        if len(selected) != n_points:
            continue

        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
        
    return best_pts

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    score = min(min_pair / 2.0, min_bound)
    return score

def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum

def soft_min_objective_and_grad(x_flat, T):
    """
    Objective: minimize -softmin(constraints)
    x_flat: flattened points (N, 2)
    T: temperature for soft-min
    """
    pts = x_flat.reshape(N, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Pairwise distances (only upper triangle to save memory/time)
    # For N=300, full matrix is 90k elements, manageable.
    d = cdist(pts, pts)
    iu = np.triu_indices(n, k=1)
    pair_half = d[iu] / 2.0
    
    all_vals = np.concatenate([pair_half, boundary])
    
    obj = -_soft_min_logsumexp(all_vals, T)
    
    # Gradient
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights
        
    grad_pts = np.zeros_like(pts)
    
    # Boundary gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    # grad of boundary (1 - ||x||) is -x/||x||
    # We want grad of -softmin. 
    # d(-softmin)/dx = sum( w_i * d(val_i)/dx )
    # w_i is negative (from grad_all).
    # Let's stick to the logic in inspiration:
    # grad_all corresponds to d(softmin)/d(val).
    # We are minimizing -softmin. So d(obj)/d(val) = -grad_all.
    # Wait, _soft_min_logsumexp returns the value.
    # Let's re-derive carefully.
    # S = -T log sum exp(-v_i/T)
    # dS/dv_i = exp(-v_i/T) / sum exp(-v_j/T) = w_i (positive weights summing to 1)
    # We want to maximize S. So we minimize -S.
    # d(-S)/dv_i = -w_i
    # d(-S)/dx = sum_i (-w_i) * (dv_i/dx)
    
    # My previous `grad_all` in inspiration was:
    # weights = exp... (positive)
    # grad_all = -weights
    # Then it used grad_all in the sum.
    # Let's ensure signs are correct.
    # We want to MINIMIZE the function f(x) = -S(x).
    # grad f = - grad S.
    # grad S = sum w_i grad v_i.
    # So grad f = - sum w_i grad v_i.
    
    # Boundary term v_i = 1 - ||x_i||. grad v_i = -x_i / ||x_i||.
    # Contribution to grad f: - w_i * (-x_i / ||x_i||) = w_i * x_i / ||x_i||.
    # Wait, if w_i is positive, this pushes x_i OUTWARDS?
    # If a point is close to boundary, v_i is small, w_i is large.
    # We want to increase v_i (move in).
    # grad v_i points inwards (opposite to x).
    # To maximize v_i, we move in direction grad v_i.
    # To minimize -v_i, we move in direction -grad v_i (outwards).
    # But we are minimizing -S. If S is dominated by boundary, we want to increase S (increase v_i).
    # So we should move in direction grad v_i.
    # Gradient of objective f=-S is -grad S.
    # If we use gradient descent: x_new = x - lr * grad f = x + lr * grad S.
    # So x moves in direction grad S.
    # grad S = sum w_i grad v_i.
    # For boundary, grad v_i = -x_i/||x_i|| (inwards).
    # So x moves inwards. Correct.
    
    # So:
    # grad_obj = - sum_i w_i * grad_val_i
    
    # Let's compute grad_obj directly.
    grad_obj = np.zeros_like(pts)
    
    # Boundary part
    # w_boundary = weights corresponding to boundary vals
    w_bound = weights[len(pair_half):]
    # grad_val_bound = -pts / norms
    # grad_obj += - w_bound * (-pts / norms) = w_bound * pts / norms
    # This seems to push out?
    # Let's check:
    # We want to minimize f.
    # If point is at boundary, v is small. w is large.
    # We want to increase v.
    # v = 1 - r. dv/dr = -1. dr/dx = x/r. dv/dx = -x/r.
    # To increase v, we need dx such that dv > 0.
    # dv = (-x/r) . dx > 0 => x . dx < 0. dx should be opposite to x.
    # Gradient of f = -S.
    # d(-S)/dx = - sum w_i dv_i/dx.
    # d(-S)/dx = - w_i (-x_i/r_i) = w_i x_i / r_i.
    # Gradient descent: x = x - lr * grad = x - lr * (w_i x_i / r_i).
    # This moves x inwards (opposite to x). Correct.
    
    grad_obj += (w_bound / safe_norms[:, np.newaxis]) * pts
    
    # Pairwise part
    # v_ij = ||x_i - x_j|| / 2.
    # grad v_ij wrt x_i = (x_i - x_j) / (2 * ||x_i - x_j||)
    # grad v_ij wrt x_j = (x_j - x_i) / (2 * ||x_i - x_j||)
    
    # We need to sum contributions from all pairs.
    # This is expensive to do with full matrix if not careful, but N=300 is ok.
    # w_pair = weights[:len(pair_half)]
    
    # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # dists = np.linalg.norm(diff, axis=2)
    # np.fill_diagonal(dists, np.inf)
    # safe_dists = np.where(dists > 1e-10, dists, 1.0)
    # unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # W = np.zeros((n, n))
    # W[iu] = w_pair
    # W[iu[1], iu[0]] = w_pair
    
    # grad_val_ij_wrt_i = 0.5 * unit_vecs
    # grad_val_ij_wrt_j = -0.5 * unit_vecs
    
    # grad_obj_i = - sum_j w_ij * grad_val_ij_wrt_i
    # grad_obj_j = - sum_i w_ji * grad_val_ij_wrt_j
    
    # Let's vectorize:
    # grad_obj += - sum_j (w_ij * 0.5 * (x_i - x_j)/d_ij)
    # Note: w_ij = w_ji.
    
    # Using the full matrix approach for clarity, N=300 -> 90k ops, trivial.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    safe_dists = np.where(dists > 1e-10, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # Construct weight matrix
    W = np.zeros((n, n))
    W[iu] = w_pair
    W[iu[1], iu[0]] = w_pair
    
    # grad_val_i = 0.5 * unit_vecs (summed over j with weights)
    # weighted_unit = W[:, :, np.newaxis] * unit_vecs
    # grad_obj += -0.5 * np.sum(weighted_unit, axis=1)
    
    grad_obj -= 0.5 * np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
    
    return obj, grad_obj.ravel()

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Initialization
    points = _hex_init(N)
    best_points = points.copy()
    best_score = compute_score(points)
    
    # 2. Soft-Min Optimization Phase
    # We use L-BFGS-B to minimize -soft_min
    T_start = 0.01
    T_end = 0.001
    
    # Run a few iterations with decreasing T
    temps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for T in temps:
        if time.time() - start_time > time_limit - 20.0:
            break
            
        try:
            res = minimize(
                soft_min_objective_and_grad,
                best_points.flatten(),
                args=(T,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            new_points = res.x.reshape(N, 2)
            
            # Validate and project if necessary
            rad = np.linalg.norm(new_points, axis=1)
            if np.any(rad > 1.0):
                new_points[rad > 1.0] *= (1.0 / rad[rad > 1.0])[:, np.newaxis]
            
            new_score = compute_score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                
        except Exception as e:
            # If optimization fails, keep previous best
            pass
            
    # 3. SLSQP Final Polish
    # Use active set of constraints for high precision
    if time.time() - start_time < time_limit - 10.0:
        # Identify active constraints
        pts = best_points
        r_current = best_score
        
        # Pairs
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        pair_dists = d / 2.0
        
        # Boundaries
        norms = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - norms
        
        # Threshold for "active"
        # We want constraints that are close to the minimum
        threshold_pair = r_current * 1.01
        threshold_bound = r_current * 1.01
        
        active_pairs = []
        iu = np.triu_indices(N, k=1)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            if pair_dists[i, j] < threshold_pair:
                active_pairs.append((i, j))
                
        active_bounds = []
        for i in range(N):
            if bound_dists[i] < threshold_bound:
                active_bounds.append(i)
                
        # If no active constraints found (shouldn't happen), add the min ones
        if not active_pairs:
            idx_min = np.unravel_index(np.argmin(pair_dists), pair_dists.shape)
            if idx_min[0] > idx_min[1]:
                idx_min = (idx_min[1], idx_min[0])
            active_pairs.append(tuple(idx_min))
        if not active_bounds:
            idx_min = np.argmin(bound_dists)
            active_bounds.append(idx_min)
        
        # Define SLSQP problem
        # Minimize -r (where r is variable)
        # Variables: x_1...x_N (2N vars) + r (1 var) -> Total 2N+1
        # Constraints:
        # 1. ||c_i - c_j||^2 - 4r^2 >= 0
        # 2. 1 - ||c_i||^2 - r^2 >= 0
        
        def slsqp_obj(x_flat):
            r = x_flat[-1]
            return -r
            
        def slsqp_grad(x_flat):
            grad = np.zeros_like(x_flat)
            grad[-1] = -1.0
            return grad
            
        constraints = []
        
        # Pair constraints
        for i, j in active_pairs:
            def con_pair(x_flat, i=i, j=j):
                pts = x_flat[:-1].reshape(N, 2)
                r = x_flat[-1]
                dist_sq = np.sum((pts[i] - pts[j])**2)
                return dist_sq - 4.0 * r**2
                
            def con_pair_grad(x_flat, i=i, j=j):
                pts = x_flat[:-1].reshape(N, 2)
                r = x_flat[-1]
                grad = np.zeros_like(x_flat)
                diff = pts[i] - pts[j]
                # d(dist_sq)/dx_i = 2 * diff
                # d(dist_sq)/dx_j = -2 * diff
                # d(-4r^2)/dr = -8r
                grad[2*i] = 2.0 * diff[0]
                grad[2*i+1] = 2.0 * diff[1]
                grad[2*j] = -2.0 * diff[0]
                grad[2*j+1] = -2.0 * diff[1]
                grad[-1] = -8.0 * r
                return grad
                
            constraints.append({'type': 'ineq', 'fun': con_pair, 'jac': con_pair_grad})
            
        # Boundary constraints
        for i in active_bounds:
            def con_bound(x_flat, i=i):
                pts = x_flat[:-1].reshape(N, 2)
                r = x_flat[-1]
                norm_sq = np.sum(pts[i]**2)
                return 1.0 - norm_sq - r**2
                
            def con_bound_grad(x_flat, i=i):
                pts = x_flat[:-1].reshape(N, 2)
                r = x_flat[-1]
                grad = np.zeros_like(x_flat)
                # d(1 - ||x||^2 - r^2)/dx = -2*x
                # d(-r^2)/dr = -2r
                grad[2*i] = -2.0 * pts[i][0]
                grad[2*i+1] = -2.0 * pts[i][1]
                grad[-1] = -2.0 * r
                return grad
                
            constraints.append({'type': 'ineq', 'fun': con_bound, 'jac': con_bound_grad})
        
        # Initial guess
        x0 = np.zeros(2*N + 1)
        x0[:-1] = best_points.flatten()
        x0[-1] = best_score
        
        try:
            res = minimize(
                slsqp_obj,
                x0,
                jac=slsqp_grad,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success:
                new_points = res.x[:-1].reshape(N, 2)
                new_r = res.x[-1]
                
                # Verify
                rad = np.linalg.norm(new_points, axis=1)
                if np.any(rad > 1.0 - 1e-9) or not np.all(np.isfinite(new_points)):
                    pass # Invalid
                else:
                    new_score = compute_score(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        
        except Exception as e:
            pass

    # Final validation
    rad_final = np.linalg.norm(best_points, axis=1)
    if np.any(rad_final > 1.0) or not np.all(np.isfinite(best_points)):
        return np.zeros((N, 2))
        
    return best_points

if __name__ == "__main__":
    construct()
