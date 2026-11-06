import numpy as np
import time
import scipy.optimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

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
        dists = np.sqrt(np.maximum(dists_sq, 0))
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
        obj_val = -(min_d - (1.0 / alpha) * np.log(sum_exp))

        # Compute gradient
        # grad = -d(obj)/dx = d(soft_min)/dx
        # soft_min = min_d - (1/alpha) * log(sum_exp)
        # We need d(soft_min)/dx
        # = d(min_d)/dx - (1/alpha) * d(log(sum_exp))/dx
        # = d(min_d)/dx - (1/alpha) * (1/sum_exp) * sum(d(exp(shifted_i))/dx)
        # = d(min_d)/dx - (1/alpha) * sum( (exp(shifted_i)/sum_exp) * d(shifted_i)/dx )
        # = d(min_d)/dx - sum( w_i * d(shifted_i)/dx )  where w_i = exp(shifted_i)/sum_exp
        # Note: shifted_i = -alpha * (d_i - min_d), so d(shifted_i)/dx = -alpha * d(d_i)/dx
        # So: = d(min_d)/dx + alpha * sum( w_i * d(d_i)/dx )
        # But this is only valid if min_d is one of the d_i, which it is.
        # Actually, the soft_min is a smooth approximation of min.
        # Let's use the weighted average of gradients of the distances.
        # grad_soft_min = sum( w_i * grad(d_i) )  where w_i = exp(-alpha*(d_i - min_d)) / sum_exp
        # This is a standard result for the log-sum-exp trick applied to min.

        weights = np.exp(shifted) / sum_exp

        # Gradient of boundary distances
        # d_bound_i = 1 - norm_i, grad = -points_i / norm_i
        grad_points = np.zeros_like(points)
        for i in range(N):
            if norms[i] > 1e-12:
                grad_points[i] = -points[i] / norms[i]
            else:
                grad_points[i] = 0.0

        # Gradient of pairwise distances
        # dists_ij = ||p_i - p_j||, grad w.r.t p_i = (p_i - p_j)/dists_ij, w.r.t p_j = (p_j - p_i)/dists_ij
        # We need to accumulate weighted gradients.
        # This is expensive to do in a loop for 50*49/2 pairs.
        # Let's vectorize where possible or use a loop if N is small enough.
        # N=50, so 1225 pairs. A loop might be acceptable if optimized.

        # Alternative: compute gradient of the objective directly using the chain rule
        # The objective is -soft_min.
        # grad_obj = -grad_soft_min
        # grad_soft_min = sum_{k} w_k * grad(d_k)
        # where k runs over all distances (boundary + pairwise).

        # Let's compute grad_soft_min as a vector of shape (N, 2)
        grad_soft_min = np.zeros_like(points)

        # Add boundary contributions
        # w_bound_i = weights[i]
        # grad(d_bound_i) = -p_i / ||p_i||
        for i in range(N):
            if norms[i] > 1e-12:
                grad_soft_min[i] += weights[i] * (-points[i] / norms[i])

        # Add pairwise contributions
        # We need to iterate over pairs (i, j) with i < j
        # d_ij = ||p_i - p_j||
        # grad w.r.t p_i: (p_i - p_j) / d_ij
        # grad w.r.t p_j: (p_j - p_i) / d_ij
        # weight = weights[N + idx] where idx is the index in the masked dists array

        # Let's construct the index mapping
        # The masked dists array has shape (N, N) with diagonal masked out.
        # We flatten it, but we need to know which (i,j) pair corresponds to which index.
        # Easier: loop over i < j.

        pair_idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                d_ij = dists[i, j]
                if d_ij > 1e-12:
                    w = weights[N + pair_idx]
                    grad_ij = (points[i] - points[j]) / d_ij
                    grad_soft_min[i] += w * grad_ij
                    grad_soft_min[j] += w * (-grad_ij)
                pair_idx += 1

        grad_obj = -grad_soft_min
        grad_flat = grad_obj.flatten()

        return obj_val, grad_flat

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

    inits = [hex_lattice(), concentric_rings(), random_init(), hex_lattice() * 0.9, 
             random_init(), ring_based_init(), better_ring_init()]
    
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline:
            break
            
        x0 = init_points.flatten()
        
        # Multi-stage alpha annealing
        alphas = [10, 50, 200, 1000, 3000]
        
        current_x = x0
        for alpha in alphas:
            if time.time() > deadline - 2:
                break
            
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad, 
                    current_x, 
                    args=(alpha,),
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
                    objective_and_grad, 
                    best_points.flatten(), 
                    args=(5000,),
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
                objective_and_grad, 
                best_points.flatten(), 
                args=(10000,),
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

    return best_points
