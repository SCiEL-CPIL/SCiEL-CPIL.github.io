import numpy as np
import time
from scipy.optimize import minimize

N = 200

def compute_radius(points):
    """Compute the maximum radius for given circle centers."""
    # Calculate min pairwise distance
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair_dist_sq = np.min(dist_sq)
    
    # Calculate min wall distance
    x = points[:, 0]
    y = points[:, 1]
    dist_x = 0.5 - np.abs(x)
    dist_y = 0.5 - np.abs(y)
    min_wall_dist = np.min(np.minimum(dist_x, dist_y))
    
    r = np.min([np.sqrt(min_pair_dist_sq)/2, min_wall_dist])
    return r

def _softmin_objective_and_grad(flat, lam):
    """
    Compute the soft-min objective and its gradient.
    We want to maximize the minimum of:
      - pairwise distances / 2
      - wall distances
    
    Equivalent to minimizing:
      F = logsumexp(-lam * d_ij^2 / 4) over all pairs i<j
        + logsumexp(-lam * w_k^2) over all wall distances w_k
    
    where d_ij is the distance between circles i and j,
    and w_k are the distances from each circle to the 4 walls.
    
    Minimizing F pushes all distances up.
    """
    pts = flat.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 0)
    
    # Use upper triangle only to avoid double counting
    iu = np.triu_indices(N, k=1)
    d2 = dist_sq[iu]
    
    # Wall distances (squared)
    x = pts[:, 0]
    y = pts[:, 1]
    wx = np.minimum(x + 0.5, 0.5 - x)  # min distance to left/right walls
    wy = np.minimum(y + 0.5, 0.5 - y)  # min distance to bottom/top walls
    w2 = np.minimum(wx**2, wy**2)
    
    # Combine all squared "distances" (pairwise/4 for radius, and wall)
    # For pairwise: radius = d/2, so we use d^2/4
    # For walls: radius = w, so we use w^2
    all_d2 = np.concatenate([d2 / 4.0, w2])
    
    # Soft-min via logsumexp: F = (1/lam) * log(sum(exp(-lam * all_d2)))
    # Gradient: dF/d(pts) = -sum_j exp(-lam*d2_j) * 2*lam*d2_j * (d/d(pts) d2_j) / sum_k exp(-lam*d2_k)
    
    # For numerical stability, subtract max
    max_val = np.max(all_d2)
    exp_vals = np.exp(-lam * (all_d2 - max_val))
    sum_exp = np.sum(exp_vals)
    
    # F = (1/lam) * log(sum_exp) + max_val
    F = max_val + np.log(sum_exp) / lam
    
    # Gradient computation
    # dF/d(pts_i) from pairwise terms:
    # For pair (i,j): d2_ij = |p_i - p_j|^2 / 4
    # d(d2_ij)/d(p_i) = (p_i - p_j) / 2
    # dF/d(p_i) += -exp(-lam*d2_ij) * 2*lam*d2_ij * (p_i - p_j)/2 / sum_exp
    #            = -lam * exp(-lam*d2_ij) * d2_ij * (p_i - p_j) / sum_exp
    
    grad = np.zeros((N, 2))
    
    # Pairwise gradient
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        w = exp_vals[idx] / sum_exp
        d2_ij = d2[idx]
        pij = pts[i] - pts[j]
        grad[i] += -lam * w * d2_ij * pij
        grad[j] += lam * w * d2_ij * pij
    
    # Wall gradient
    # For wall distance w = min(x+0.5, 0.5-x) for x-coordinate
    # If x < 0: w = x+0.5, dw/dx = 1
    # If x > 0: w = 0.5-x, dw/dx = -1
    # w2 = w^2, d(w2)/dx = 2*w*dw/dx
    
    # We need to track which wall is the minimum for each circle
    # For x: if x <= 0, left wall is min; else right wall
    # For y: if y <= 0, bottom wall is min; else top wall
    
    for i in range(N):
        xi = pts[i, 0]
        yi = pts[i, 1]
        
        # Determine min wall for x
        wx_l = xi + 0.5
        wx_r = 0.5 - xi
        if wx_l <= wx_r:
            # Left wall is min
            wx = wx_l
            dwdx = 1.0
        else:
            wx = wx_r
            dwdx = -1.0
        
        # Determine min wall for y
        wy_b = yi + 0.5
        wy_t = 0.5 - yi
        if wy_b <= wy_t:
            # Bottom wall is min
            wy = wy_b
            dwdy = 1.0
        else:
            wy = wy_t
            dwdy = -1.0
        
        # Which is the overall min?
        if wx**2 <= wy**2:
            w = wx
            dwdp = np.array([dwdx, 0.0])
        else:
            w = wy
            dwdp = np.array([0.0, dwdy])
        
        w2_i = w**2
        w_wall_idx = len(d2) + i  # index in all_d2
        w_wall = exp_vals[w_wall_idx] / sum_exp
        
        # dF/d(p_i) from wall = -lam * w_wall * w2_i * 2 * w * dwdp
        # Simplified: -2*lam*w_wall*w2_i*(w*dwdp) but w2_i = w^2, so:
        # -2*lam*w_wall*w^2*(w*dwdp) ... wait, let me redo
        # d(w2)/d(p) = 2*w*dwdp
        # dF/d(p_i) += -lam * w_wall * 2 * w * dwdp * w  (since d2_i = w2_i = w^2)
        # Actually: dF/d(p) = -lam * w_k * d2_k * d(d2_k)/d(p) / sum_exp
        # where d2_k = w^2, d(d2_k)/d(p) = 2*w*dwdp
        # So: dF/d(p_i) += -lam * w_wall * w^2 * 2*w*dwdp
        #     = -2*lam*w_wall*w^3*dwdp
        
        # Hmm, that seems off. Let me reconsider.
        # F = (1/lam)*log(sum(exp(-lam*d2_k)))
        # dF/d(p_i) = (1/lam) * (1/sum_exp) * sum_k exp(-lam*d2_k) * (-lam) * d(d2_k)/d(p_i)
        #           = -sum_k exp(-lam*d2_k) * d(d2_k)/d(p_i) / sum_exp
        #           = -sum_k w_k * d(d2_k)/d(p_i)
        
        # So for wall: dF/d(p_i) += -w_wall * d(w2_i)/d(p_i) = -w_wall * 2*w*dwdp
        
        grad[i] += -w_wall * 2.0 * w * dwdp
    
    return F, grad.flatten()

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    best_r = 0
    best_points = None
    
    # Initializations
    inits = []
    
    # 1. Random initializations
    np.random.seed(42)
    for _ in range(30):
        pts = np.random.rand(N, 2) - 0.5
        inits.append(pts)
    
    # 2. Hexagonal grid initializations
    for n_rows in range(10, 21):
        c_base = N // n_rows
        rem = N % n_rows
        counts = [c_base] * n_rows
        for i in range(rem):
            counts[i] += 1
            
        pts = []
        y = 0
        for i, c in enumerate(counts):
            xs = np.arange(c) - (c - 1) / 2.0
            if i % 2 == 1:
                xs += 0.5
            for x in xs:
                pts.append([x, y])
            y += np.sqrt(3) / 2.0
            
        pts = np.array(pts)
        max_x = np.max(np.abs(pts[:, 0]))
        max_y = np.max(np.abs(pts[:, 1]))
        scale = 0.45 / max(max_x, max_y)
        pts *= scale
        inits.append(pts)
    
    # 3. Square grid initialization
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid_pts = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid_pts[:N])
    
    # Optimization with L-BFGS-B
    # Use a moderate lambda for the soft-min
    lam = 50.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        flat_init = init.flatten()
        
        # Define objective and gradient for scipy
        def obj_grad(flat):
            F, g = _softmin_objective_and_grad(flat, lam)
            return F, g
        
        # Run L-BFGS-B
        try:
            result = minimize(
                obj_grad,
                flat_init,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < 1e10:
                pts_opt = result.x.reshape(N, 2)
                # Clip to ensure valid
                pts_opt = np.clip(pts_opt, -0.4999, 0.4999)
                r = compute_radius(pts_opt)
                if r > best_r:
                    best_r = r
                    best_points = pts_opt.copy()
        except Exception:
            pass
        
        # Also try a higher lambda for polishing if we have time
        if time.time() < deadline - 10:
            lam_high = 200.0
            
            def obj_grad_high(flat):
                F, g = _softmin_objective_and_grad(flat, lam_high)
                return F, g
            
            try:
                result2 = minimize(
                    obj_grad_high,
                    best_points.flatten() if best_points is not None else flat_init,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 100, 'maxfun': 300, 'gtol': 1e-8}
                )
                
                if result2.success or result2.fun < 1e10:
                    pts_opt2 = result2.x.reshape(N, 2)
                    pts_opt2 = np.clip(pts_opt2, -0.4999, 0.4999)
                    r2 = compute_radius(pts_opt2)
                    if r2 > best_r:
                        best_r = r2
                        best_points = pts_opt2.copy()
            except Exception:
                pass
    
    # Final check
    if best_points is not None:
        r = compute_radius(best_points)
        if r > best_r:
            best_r = r
    
    if best_points is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points
