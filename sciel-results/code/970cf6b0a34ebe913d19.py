import numpy as np
import time
import scipy.optimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_score = -np.inf
    best_points = None
    
    def get_score(points):
        """Compute the true score (minimum radius)."""
        if points.shape != (N, 2):
            return -np.inf
        if not np.all(np.isfinite(points)):
            return -np.inf
        
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms >= 1.0):
            return -np.inf
            
        d_bound = 1.0 - norms
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dists_sq)
        dists[~mask] = np.inf
        min_pair = np.min(dists[mask])
        
        return min(np.min(d_bound), min_pair) / 2.0

    def objective_and_grad(x, alpha):
        """
        Smoothed objective: -soft_min(distances)
        Returns (value, gradient)
        """
        points = x.reshape(N, 2)
        
        # Boundary distances
        norms = np.linalg.norm(points, axis=1) + 1e-15
        d_bound = 1.0 - norms
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dists_sq) + 1e-15
        dists[~mask] = np.inf
        
        all_dists = np.concatenate([d_bound, dists[mask]])
        
        # Log-sum-exp stability
        min_d = np.min(all_dists)
        shifted = -alpha * (all_dists - min_d)
        sum_exp = np.sum(np.exp(shifted))
        
        # soft_min = min_d - 1/alpha * log(sum_exp)
        # We minimize -soft_min
        obj_val = -(min_d - (1.0 / alpha) * np.log(sum_exp))
        
        # Gradient computation
        # d/dx (-soft_min) = d/dx (1/alpha * log(sum exp(-alpha d_k)))
        # = 1/alpha * sum(exp(-alpha d_k) / sum_exp) * (-alpha * d(d_k)/dx)
        # = - sum(weights_k * d(d_k)/dx) where weights_k = exp(-alpha d_k)/sum_exp
        
        weights = np.exp(shifted) / sum_exp
        
        grad = np.zeros_like(points)
        
        # Boundary gradient: d(d_bound)/dx = -x/|x|
        w_bound = weights[:N]
        grad += -w_bound[:, np.newaxis] * (points / norms[:, np.newaxis])
        
        # Pairwise gradient: d(dist_ij)/dx_i = (x_i - x_j)/dist_ij
        # We need to accumulate for each pair (i, j)
        iu = np.triu_indices(N, 1)
        i_idx, j_idx = iu
        w_pair = weights[N:]
        
        diff_pairs = points[i_idx] - points[j_idx]
        dist_pairs = dists[i_idx, j_idx]
        dir_vecs = diff_pairs / dist_pairs[:, np.newaxis]
        
        # grad[i] += -w * dir_vec
        # grad[j] += +w * dir_vec
        np.add.at(grad, i_idx, -w_pair[:, np.newaxis] * dir_vecs)
        np.add.at(grad, j_idx, w_pair[:, np.newaxis] * dir_vecs)
        
        return obj_val, grad.flatten()

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
            # Sort by distance to center to pick the most central ones
            norms = np.linalg.norm(points, axis=1)
            idx = np.argsort(norms)[:N]
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

    def ring_config():
        # Known good configuration: 1 center, 6, 12, 18, 13 in rings
        centers = []
        centers.append([0.0, 0.0])
        for i in range(6):
            theta = i * np.pi / 3
            centers.append([0.25 * np.cos(theta), 0.25 * np.sin(theta)])
        for i in range(12):
            theta = i * np.pi / 6
            centers.append([0.5 * np.cos(theta), 0.5 * np.sin(theta)])
        for i in range(18):
            theta = i * 2 * np.pi / 18
            centers.append([0.75 * np.cos(theta), 0.75 * np.sin(theta)])
        for i in range(13):
            theta = i * 2 * np.pi / 13
            centers.append([0.95 * np.cos(theta), 0.95 * np.sin(theta)])
        return np.array(centers[:N])

    inits = [
        hex_lattice(), 
        concentric_rings(), 
        random_init(), 
        hex_lattice() * 0.9, 
        random_init(),
        ring_config()
    ]
    
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init_points.flatten()
        
        # Multi-stage alpha annealing
        # Start with low alpha (smooth, global) and increase to high alpha (sharp, local)
        alphas = [10, 50, 200, 1000]
        
        current_x = x0
        for alpha in alphas:
            if time.time() > deadline - 2:
                break
            
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad, 
                    current_x, 
                    args=(alpha,),
                    jac=True,
                    method='L-BFGS-B', 
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
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad, 
                    best_points.flatten(), 
                    args=(5000,),
                    jac=True,
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'gtol': 1e-8}
                )
                if res.x is not None:
                    curr_points = res.x.reshape(N, 2)
                    curr_score = get_score(curr_points)
                    if curr_score > best_score:
                        best_score = curr_score
                        best_points = curr_points.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback
        best_points = hex_lattice()
        
    return best_points
