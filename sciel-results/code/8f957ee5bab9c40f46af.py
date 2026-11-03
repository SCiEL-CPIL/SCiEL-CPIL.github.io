"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21


def _compute_radius(circles, W, H):
    """Compute max feasible radius given positions and rectangle.
    circles: (n, 2) array of centers.
    Returns the max r such that all circles fit and are disjoint.
    """
    n = len(circles)
    # Constraint 1: each circle must fit in rectangle
    r_boundary = np.min(np.minimum(np.minimum(circles[:, 0], W - circles[:, 0]),
                                   np.minimum(circles[:, 1], H - circles[:, 1])))
    # Constraint 2: pairwise distances
    r_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(circles[i] - circles[j])
            r_pair = min(r_pair, d / 2.0)
    return max(0.0, min(r_boundary, r_pair))


def _validate(circles, W, H, r):
    """Check if circles with radius r fit in W x H rectangle without overlap."""
    if r <= 0:
        return False
    if W <= 0 or H <= 0:
        return False
    # Boundary check
    if np.any(circles[:, 0] < r) or np.any(circles[:, 0] > W - r):
        return False
    if np.any(circles[:, 1] < r) or np.any(circles[:, 1] > H - r):
        return False
    # Pairwise distance check
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            d = np.linalg.norm(circles[i] - circles[j])
            if d < 2 * r - 1e-9:
                return False
    return True


def _objective(params, W, H):
    """Objective: negative sum of radii (to minimize). params = centers (n*2)."""
    centers = params.reshape(-1, 2)
    r = _compute_radius(centers, W, H)
    if r <= 0:
        return 1e10  # infeasible
    return -r


def _nelder_mead_minimize(func, x0, max_iter=2000, tol=1e-8, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [x0.astype(float)]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 if x0[i] != 0 else 0.005
        xi[i] += step
        simplex.append(xi)
    simplex = np.array(simplex)
    fvals = np.array([func(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        if np.abs(fvals[-1] - fvals[0]) < tol:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
    
    return simplex[np.argmin(fvals)], np.min(fvals)


def _pack_for_aspect(W, H, deadline):
    """Pack circles in W x H rectangle. Returns (circles, W, H, sum_radii)."""
    # Initial grid placement
    cols = max(1, int(np.round(np.sqrt(N * W / H))))
    rows = max(1, int(np.ceil(N / cols)))
    
    # Scale to fill rectangle
    cell_w = W / cols
    cell_h = H / rows
    
    centers = np.zeros((N, 2))
    for i in range(N):
        c = i % cols
        r = i // cols
        centers[i, 0] = (c + 0.5) * cell_w
        centers[i, 1] = (r + 0.5) * cell_h
    
    # Clamp to valid positions
    r_est = min(cell_w, cell_h) / 2.0
    centers[:, 0] = np.clip(centers[:, 0], r_est, W - r_est)
    centers[:, 1] = np.clip(centers[:, 1], r_est, H - r_est)
    
    best_centers = centers.copy()
    best_r = _compute_radius(centers, W, H)
    best_sum = best_r * N
    
    # Iterative radius growth with collision resolution
    r = best_r
    for iteration in range(200):
        if time.time() > deadline:
            break
        r *= 1.02  # grow radius by 2%
        
        # Check if current configuration is feasible at new r
        if _validate(centers, W, H, r):
            best_r = r
            best_sum = r * N
            best_centers = centers.copy()
            continue
        
        # Resolve collisions: push overlapping circles apart
        converged = False
        for inner in range(50):
            if time.time() > deadline:
                break
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    min_d = 2 * r
                    if d < min_d:
                        if d < 1e-10:
                            centers[i, 0] += 0.01
                            centers[j, 0] -= 0.01
                        else:
                            direction = (centers[i] - centers[j]) / d
                            push = (min_d - d) / 2.0 + 1e-6
                            centers[i] += direction * push
                            centers[j] -= direction * push
                        moved = True
            
            # Clamp to boundaries
            centers[:, 0] = np.clip(centers[:, 0], r, W - r)
            centers[:, 1] = np.clip(centers[:, 1], r, H - r)
            
            if not moved:
                converged = True
                break
        
        if converged and _validate(centers, W, H, r):
            best_r = r
            best_sum = r * N
            best_centers = centers.copy()
        else:
            # Failed to resolve, revert and try smaller growth
            centers = best_centers.copy()
            r = best_r
            break
    
    # Local optimization with Nelder-Mead
    if time.time() < deadline - 5:
        def local_obj(params):
            c = params.reshape(-1, 2)
            rad = _compute_radius(c, W, H)
            if rad <= 0:
                return 1e10
            # Penalize if any circle out of bounds
            out = np.any(c[:, 0] < rad) or np.any(c[:, 0] > W - rad) or \
                  np.any(c[:, 1] < rad) or np.any(c[:, 1] > H - rad)
            if out:
                return 1e10
            return -rad
        
        x0 = best_centers.flatten()
        x_opt, f_opt = _nelder_mead_minimize(local_obj, x0, max_iter=500, tol=1e-7)
        if f_opt < -best_r:
            best_r = -f_opt
            best_centers = x_opt.reshape(-1, 2)
            best_sum = best_r * N
    
    return best_centers, W, H, best_sum


def construct():
    """Construct 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_sum = 0.0
    best_result = None
    
    # Try different aspect ratios
    # Perimeter = 2(W + H) = 4, so W + H = 2
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5, 2.0, 0.4]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        # W * H with W + H = 2
        # W = ar * H, so H = 2/(1+ar), W = 2*ar/(1+ar)
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        centers, W, H, s = _pack_for_aspect(W, H, deadline)
        
        if s > best_sum:
            best_sum = s
            best_result = (centers, W, H)
    
    if best_result is None:
        # Fallback: square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    centers, W, H = best_result
    r = _compute_radius(centers, W, H)
    
    # Build output: (x, y, r) for each circle
    result = np.zeros((N, 3))
    result[:, 0] = centers[:, 0]
    result[:, 1] = centers[:, 1]
    result[:, 2] = r
    
    return result
