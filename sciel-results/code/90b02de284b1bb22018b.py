import numpy as np
import time
import scipy.optimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_score = -np.inf
    best_points = None
    
    # Helper to compute the true score (minimum radius)
    def get_score(points):
        # points: (N, 2)
        # Check boundary
        d_bound = 1.0 - np.linalg.norm(points, axis=1)
        
        # Check pairwise
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dists_sq)
        dists[~mask] = np.inf
        min_pair = np.min(dists[mask])
        
        return min(np.min(d_bound), min_pair) / 2.0

    # Smoothed objective for optimization with analytic gradient
    def objective_and_grad(x, alpha):
        points = x.reshape(N, 2)
        
        # Boundary distances
        norms = np.linalg.norm(points, axis=1)
        d_bound = 1.0 - norms
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(np.maximum(dists_sq, 0))
        dists[~mask] = np.inf
        
        # Combine all distances
        all_dists = np.concatenate([d_bound, dists[mask]])
        
        # Optimization trick: log-sum-exp stability
        min_d = np.min(all_dists)
        
        # Calculate sum of exps
        shifted = -alpha * (all_dists - min_d)
        # Clip for numerical stability
        shifted = np.clip(shifted, -700, 700)
        sum_exp = np.sum(np.exp(shifted))
        
        # soft_min = min_d - 1/alpha * log( sum( exp(-alpha * (d_k - min_d)) ) )
        soft_min = min_d - (1.0 / alpha) * np.log(sum_exp)
        obj_val = -soft_min
        
        # Compute gradient
        # d(soft_min)/d(x_i) for each coordinate
        # soft_min = min_d - (1/alpha) * log(sum exp(-alpha*(d_k - min_d)))
        # For simplicity, use the weighted average gradient
        # w_k = exp(-alpha*(d_k - min_d)) / sum_exp
        weights = np.exp(shifted) / sum_exp
        
        grad = np.zeros_like(points)
        
        # Boundary contributions
        for i in range(N):
            if norms[i] > 1e-10:
                # d(d_bound_i)/d(points_i) = -points_i / |points_i|
                w = weights[i]
                grad[i] += w * (points[i] / norms[i])
        
        # Pairwise contributions
        # For each pair (i,j), contribution to grad[i] and grad[j]
        pair_idx = 0
        for i in range(N):
            for j in range(i+1, N):
                w = weights[N + pair_idx]
                d_ij = dists[i, j]
                if d_ij > 1e-10:
                    unit_vec = (points[i] - points[j]) / d_ij
                    grad[i] += w * unit_vec
                    grad[j] -= w * unit_vec
                pair_idx += 1
        
        # The objective is -soft_min, so gradient of objective is -gradient of soft_min
        # But we want to minimize -soft_min, so we return gradient of -soft_min
        # d(-soft_min)/dx = -d(soft_min)/dx
        # d(soft_min)/dx = sum w_k * d(d_k)/dx
        # For boundary: d(d_bound_i)/d(x_i) = -x_i/|x_i|, so d(soft_min)/dx_i += w_i * (-x_i/|x_i|)
        # Wait, let me re-derive.
        
        # soft_min = min_d - (1/alpha)*log(sum exp(-alpha*(d_k - min_d)))
        # Let S = sum exp(-alpha*(d_k - min_d))
        # soft_min = min_d - (1/alpha)*log(S)
        # d(soft_min)/dx = d(min_d)/dx - (1/alpha) * (1/S) * dS/dx
        # dS/dx = sum exp(-alpha*(d_k-min_d)) * (-alpha) * d(d_k)/dx
        # = -alpha * sum w_k * d(d_k)/dx
        # So d(soft_min)/dx = d(min_d)/dx + sum w_k * d(d_k)/dx
        # But min_d is not differentiable. The log-sum-exp trick makes it smooth.
        # Actually, the standard result is:
        # d(soft_min)/dx = sum w_k * d(d_k)/dx
        # where w_k = exp(-alpha*(d_k-min_d))/S
        
        # So for boundary i: d(d_bound_i)/d(x_i) = -x_i/|x_i|
        # For pair (i,j): d(d_ij)/d(x_i) = (x_i - x_j)/|x_i - x_j|
        
        # Let me recompute gradient properly
        grad = np.zeros_like(points)
        
        # Boundary: d_bound_i = 1 - |x_i|
        # d(d_bound_i)/d(x_i) = -x_i / |x_i|
        for i in range(N):
            if norms[i] > 1e-10:
                w = weights[i]
                grad[i] -= w * (points[i] / norms[i])
        
        # Pairwise: d_ij = |x_i - x_j|
        # d(d_ij)/d(x_i) = (x_i - x_j)/d_ij
        # d(d_ij)/d(x_j) = (x_j - x_i)/d_ij = -(x_i - x_j)/d_ij
        pair_idx = 0
        for i in range(N):
            for j in range(i+1, N):
                w = weights[N + pair_idx]
                d_ij = dists[i, j]
                if d_ij > 1e-10:
                    unit_vec = (points[i] - points[j]) / d_ij
                    grad[i] += w * unit_vec
                    grad[j] -= w * unit_vec
                pair_idx += 1
        
        # Objective is -soft_min, so grad_objective = -grad_soft_min
        # But wait, we computed grad as sum w_k * d(d_k)/dx which is d(soft_min)/dx
        # So we need to negate it for the objective -soft_min
        grad = -grad
        
        return obj_val, grad.flatten()

    # Initializations
    def hex_lattice():
        points = []
        spacing = 0.2
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if np.sqrt(x*x + y*y) < 0.9:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            idx = np.random.choice(len(points), N, replace=False)
            return points[idx]
        return points

    def concentric_rings():
        points = []
        points.append([0,0])
        for i in range(6):
            theta = i * np.pi / 3
            points.append([0.35 * np.cos(theta), 0.35 * np.sin(theta)])
        for i in range(12):
            theta = i * np.pi / 6
            points.append([0.65 * np.cos(theta), 0.65 * np.sin(theta)])
        while len(points) < N:
            r = np.random.uniform(0.7, 0.95)
            theta = np.random.uniform(0, 2*np.pi)
            points.append([r * np.cos(theta), r * np.sin(theta)])
        return np.array(points[:N])

    def random_init():
        while True:
            pts = np.random.uniform(-1, 1, (N, 2))
            norms = np.linalg.norm(pts, axis=1)
            if np.all(norms < 1.0):
                return pts

    def ring_based_init():
        """Create initial configuration based on ring structure similar to known packings"""
        points = []
        # Center
        points.append([0, 0])
        # Ring 1: 6 points
        for i in range(6):
            theta = i * 2 * np.pi / 6
            points.append([0.25 * np.cos(theta), 0.25 * np.sin(theta)])
        # Ring 2: 12 points
        for i in range(12):
            theta = i * 2 * np.pi / 12 + np.pi / 12
            points.append([0.5 * np.cos(theta), 0.5 * np.sin(theta)])
        # Ring 3: 18 points
        for i in range(18):
            theta = i * 2 * np.pi / 18
            points.append([0.75 * np.cos(theta), 0.75 * np.sin(theta)])
        # Ring 4: remaining points
        remaining = N - len(points)
        for i in range(remaining):
            theta = i * 2 * np.pi / remaining + np.pi / remaining
            points.append([0.9 * np.cos(theta), 0.9 * np.sin(theta)])
        return np.array(points[:N])

    def better_ring_init():
        """Improved ring initialization based on known packing structure"""
        points = []
        # Center
        points.append([0, 0])
        
        # Ring 1: 6 points at radius ~0.25
        r1 = 0.25
        for i in range(6):
            theta = i * 2 * np.pi / 6
            points.append([r1 * np.cos(theta), r1 * np.sin(theta)])
        
        # Ring 2: 12 points at radius ~0.5
        r2 = 0.5
        for i in range(12):
            theta = i * 2 * np.pi / 12 + np.pi / 12
            points.append([r2 * np.cos(theta), r2 * np.sin(theta)])
        
        # Ring 3: 18 points at radius ~0.75
        r3 = 0.75
        for i in range(18):
            theta = i * 2 * np.pi / 18
            points.append([r3 * np.cos(theta), r3 * np.sin(theta)])
        
        # Ring 4: remaining 14 points at radius ~0.92
        remaining = N - len(points)
        r4 = 0.92
        for i in range(remaining):
            theta = i * 2 * np.pi / remaining + np.pi / remaining
            points.append([r4 * np.cos(theta), r4 * np.sin(theta)])
        
        return np.array(points[:N])

    def hexagonal_init():
        """Hexagonal lattice initialization similar to inspiration 1"""
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        for _ in range(20):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
        return random_init()

    inits = [hex_lattice(), concentric_rings(), random_init(), hex_lattice() * 0.9, 
             random_init(), ring_based_init(), better_ring_init(), hexagonal_init()]
    
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init_points.flatten()
        
        # Multi-stage alpha annealing with analytic gradient
        alphas = [10, 50, 200, 1000, 3000]
        
        current_x = x0
        for alpha in alphas:
            if time.time() > deadline - 2:
                break
            
            try:
                res = scipy.optimize.minimize(
                    lambda x: objective_and_grad(x, alpha),
                    current_x, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 300, 'gtol': 1e-6, 'maxls': 50}
                )
                
                if res.x is not None:
                    current_x = res.x
            except Exception:
                pass
        
        # Evaluate final result
        curr_points = current_x.reshape(N, 2)
        curr_score = get_score(curr_points)
        
        if curr_score > best_score:
            best_score = curr_score
            best_points = curr_points.copy()

    # If we still have time, do some local polishing on the best point with very high alpha
    if best_points is not None and time.time() < deadline - 5:
        for _ in range(8):
            if time.time() > deadline - 2:
                break
            try:
                res = scipy.optimize.minimize(
                    lambda x: objective_and_grad(x, 5000),
                    best_points.flatten(), 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 200, 'gtol': 1e-9}
                )
                if res.x is not None:
                    curr_points = res.x.reshape(N, 2)
                    curr_score = get_score(curr_points)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_points = curr_points.copy()
            except Exception:
                pass

    # Final high-precision polish with even higher alpha
    if best_points is not None and time.time() < deadline - 1:
        try:
            res = scipy.optimize.minimize(
                lambda x: objective_and_grad(x, 10000),
                best_points.flatten(), 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-10}
            )
            if res.x is not None:
                curr_points = res.x.reshape(N, 2)
                curr_score = get_score(curr_points)
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = curr_points.copy()
        except Exception:
            pass

    # If no valid result found, return a safe default
    if best_points is None:
        return better_ring_init()

    return best_points
