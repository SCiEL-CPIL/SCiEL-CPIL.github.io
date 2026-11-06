import numpy as np
import time
import scipy.optimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
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

    # Smoothed objective for optimization
    # We want to maximize the minimum of distances.
    # Let d_k be the list of all relevant distances (pairwise and boundary).
    # We minimize -soft_min(d_k).
    # soft_min(d, alpha) = -1/alpha * log( sum( exp(-alpha * d_k) ) )
    # As alpha -> inf, soft_min -> min.
    
    def objective_smooth(x, alpha):
        points = x.reshape(N, 2)
        
        # Boundary distances
        d_bound = 1.0 - np.linalg.norm(points, axis=1)
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        mask = ~np.eye(N, dtype=bool)
        dists = np.sqrt(dists_sq)
        dists[~mask] = np.inf
        
        # Combine all distances
        all_dists = np.concatenate([d_bound, dists[mask]])
        
        # Optimization trick: log-sum-exp stability
        min_d = np.min(all_dists)
        
        # Calculate sum of exps
        # exp(-alpha * (d - min_d))
        shifted = -alpha * (all_dists - min_d)
        # Clip to avoid underflow to 0 which is fine, but overflow?
        # shifted is <= 0. So exp(shifted) <= 1. No overflow.
        sum_exp = np.sum(np.exp(shifted))
        
        # soft_min = min_d - 1/alpha * log( sum( exp(-alpha * (d_k - min_d)) ) )
        # We want to maximize this. So minimize negative of this.
        obj_val = -(min_d - (1.0 / alpha) * np.log(sum_exp))
        
        return obj_val

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

    inits = [hex_lattice(), concentric_rings(), random_init(), hex_lattice() * 0.9, random_init(), ring_based_init()]
    
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init_points.flatten()
        
        # Multi-stage alpha annealing
        # Start with low alpha (smooth, global) and increase to high alpha (sharp, local)
        alphas = [10, 30, 100, 300, 1000]
        
        current_x = x0
        for alpha in alphas:
            if time.time() > deadline - 2:
                break
            
            try:
                # Use L-BFGS-B which is efficient for smooth functions
                res = scipy.optimize.minimize(
                    objective_smooth, 
                    current_x, 
                    args=(alpha,),
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
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            try:
                # Very sharp alpha for fine tuning
                res = scipy.optimize.minimize(
                    objective_smooth, 
                    best_points.flatten(), 
                    args=(5000,),
                    method='L-BFGS-B', 
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

    return best_points
