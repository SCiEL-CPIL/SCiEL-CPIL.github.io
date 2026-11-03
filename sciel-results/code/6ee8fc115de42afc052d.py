"""Circle packing in unit square – 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26


def _init_positions():
    """Create an initial grid of N centers."""
    side = int(np.ceil(np.sqrt(N)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= N:
                break
            pts.append([(i + 0.5) / side, (j + 0.5) / side])
    return np.array(pts)


def _radius_and_obj(pts):
    """
    Given centers pts (N x 2), compute the radius of each circle as
    the minimum distance to the boundary and to the nearest other center
    (Voronoi-based: radius_i = min(dist_to_boundary_i, 0.5 * min_j dist_ij)).
    Return (radii, sum_of_radii).
    """
    n = len(pts)
    # Distance to boundary for each point
    dist_boundary = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]),
                               np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    min_dist = np.min(d, axis=1)
    radii = np.minimum(dist_boundary, 0.5 * min_dist)
    return radii, np.sum(radii)


def _neg_obj_flat(x):
    """Objective for minimizer: negative sum of radii."""
    pts = x.reshape(-1, 2)
    _, obj = _radius_and_obj(pts)
    return -obj


def _grad_flat(x):
    """
    Gradient of negative sum of radii w.r.t. centers.
    Uses finite differences on the non-smooth min, but we use a soft version
    for better optimization. Actually, let's use a smooth surrogate:
    replace min with a smooth min via log-sum-exp.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    eps = 1e-4
    # We'll use a smooth approximation: for each i,
    # r_i = min( dist_boundary_i, 0.5 * min_j d_ij )
    # Smooth min via: softmin(a, b) = -log(exp(-a/t) + exp(-b/t)) * t  (as t->0 gives min)
    # But for gradient, let's just do finite differences on the whole objective.
    # Given time constraints, use a simple finite-difference gradient.
    grad = np.zeros_like(x)
    for k in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[k] += eps
        x_minus[k] -= eps
        grad[k] = (_neg_obj_flat(x_plus) - _neg_obj_flat(x_minus)) / (2 * eps)
    return grad


def _smooth_neg_obj(x, temp=0.02):
    """
    Smooth surrogate: replace hard min with log-sum-exp based soft min.
    r_i ≈ softmin(dist_boundary_i, 0.5*min_j d_ij)
    We approximate min_j d_ij with a soft-min over j.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    dist_boundary = np.minimum(np.minimum(pts[:, 0], 1.0 - pts[:, 0]),
                               np.minimum(pts[:, 1], 1.0 - pts[:, 1]))
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(d, np.inf)
    # soft min over j for each i: softmin_j(d_ij) ≈ -t * log(sum_j exp(-d_ij/t))
    # but we need 0.5 * softmin
    t = temp
    # For numerical stability
    log_sum = np.zeros(n)
    for i in range(n):
        vals = d[i] / t
        max_val = np.max(vals[np.isfinite(vals)])
        log_sum[i] = max_val + np.log(np.sum(np.exp(-vals[np.isfinite(vals)])))
    soft_min_d = t * log_sum  # this is actually t * (max + log(sum(exp(-x/t)))) ... let me redo
    # Actually: softmin(x) = -t * log(sum(exp(-x_i / t)))
    # = -t * ( -max_i(x_i/t) + log(sum(exp(-(x_i - max_i)/t))) )  ... let's just compute directly
    soft_min_d = np.zeros(n)
    for i in range(n):
        vals = d[i]
        finite = np.isfinite(vals)
        if not np.any(finite):
            soft_min_d[i] = 1.0
            continue
        v = vals[finite]
        m = np.min(v)
        # softmin ≈ m - t*log(sum(exp(-(v-m)/t)))... no
        # softmin(v) = -t*log(sum(exp(-v/t)))
        # = -t * ( -m/t + log(sum(exp(-(v-m)/t))) )  ... 
        # = m - t*log(sum(exp(-(v-m)/t)))
        # Wait: -t*log(sum(exp(-v/t))) = -t*log(exp(-m/t)*sum(exp(-(v-m)/t)))
        # = -t*(-m/t + log(sum(exp(-(v-m)/t)))) = m - t*log(sum(exp(-(v-m)/t)))
        exp_sum = np.sum(np.exp(-(v - m) / t))
        soft_min_d[i] = m - t * np.log(exp_sum)
    
    radii = np.minimum(dist_boundary, 0.5 * soft_min_d)
    # Now smooth the min between boundary and neighbor
    # softmin(a, b) = -t*log(exp(-a/t)+exp(-b/t))
    a = dist_boundary
    b = 0.5 * soft_min_d
    exp_sum = np.exp(-a / t) + np.exp(-b / t)
    soft_r = -t * np.log(exp_sum)
    return -np.sum(soft_r)


def _smooth_grad(x, temp=0.02):
    """Finite-difference gradient of the smooth objective."""
    n = len(x)
    eps = 1e-6
    grad = np.zeros(n)
    for k in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[k] += eps
        xm[k] -= eps
        grad[k] = (_smooth_neg_obj(xp, temp) - _smooth_neg_obj(xm, temp)) / (2 * eps)
    return grad


def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    best_pts = _init_positions()
    best_val, _ = _radius_and_obj(best_pts)
    
    # Try a few different initializations
    inits = [best_pts]
    # Random init
    np.random.seed(42)
    for _ in range(3):
        pts_rand = np.random.rand(N, 2)
        inits.append(pts_rand)
    # Jittered grid
    base = _init_positions()
    for _ in range(3):
        jitter = np.random.randn(N, 2) * 0.05
        pts_jit = np.clip(base + jitter, 0.01, 0.99)
        inits.append(pts_jit)
    
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline:
            break
        x0 = init.flatten()
        
        # Phase 1: coarse optimization with larger temp
        res = minimize(_smooth_neg_obj, x0, method='L-BFGS-B',
                       args=(0.05,), options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
        if time.time() > deadline:
            break
        x1 = res.x
        _, val1 = _radius_and_obj(x1.reshape(-1, 2))
        
        # Phase 2: refine with smaller temp
        res2 = minimize(_smooth_neg_obj, x1, method='L-BFGS-B',
                        args=(0.01,), options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() > deadline:
            break
        x2 = res2.x
        _, val2 = _radius_and_obj(x2.reshape(-1, 2))
        
        # Phase 3: final polish with very small temp
        res3 = minimize(_smooth_neg_obj, x2, method='L-BFGS-B',
                        args=(0.005,), options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
        x3 = res3.x
        _, val3 = _radius_and_obj(x3.reshape(-1, 2))
        
        if val3 > best_val:
            best_val = val3
            best_pts = x3.reshape(-1, 2)
    
    # Final: compute radii from best centers
    radii, total = _radius_and_obj(best_pts)
    
    # Build output array
    result = np.column_stack([best_pts, radii])
    
    # Validate: ensure all circles are inside [0,1]^2 and disjoint
    for i in range(N):
        x, y, r = result[i]
        if x - r < -1e-9 or y - r < -1e-9 or x + r > 1 + 1e-9 or y + r > 1 + 1e-9:
            # Clamp
            r = min(x, 1 - x, y, 1 - y)
            result[i, 2] = r
    # Check pairwise
    for i in range(N):
        for j in range(i + 1, N):
            dx = result[i, 0] - result[j, 0]
            dy = result[i, 1] - result[j, 1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < result[i, 2] + result[j, 2] - 1e-9:
                # Adjust radii to make them just touching
                avg = dist / 2
                result[i, 2] = min(result[i, 2], avg)
                result[j, 2] = min(result[j, 2], avg)
    
    return result


# Quick self-test
if __name__ == '__main__':
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Sum of radii: {np.sum(res[:,2]):.6f}")
    print(f"Shape: {res.shape}")
    # Verify validity
    valid = True
    for i in range(N):
        x, y, r = res[i]
        if x - r < -1e-8 or y - r < -1e-8 or x + r > 1 + 1e-8 or y + r > 1 + 1e-8:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((res[i,0]-res[j,0])**2 + (res[i,1]-res[j,1])**2)
                if d < res[i,2] + res[j,2] - 1e-8:
                    valid = False
                    break
            if not valid:
                break
    print(f"Valid: {valid}")
