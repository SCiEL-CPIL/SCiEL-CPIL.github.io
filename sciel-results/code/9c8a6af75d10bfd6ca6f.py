import numpy as np
import time
import scipy.optimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
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

    # Physics-based relaxation: minimize energy where circles repel and wall attracts
    def physics_energy_and_grad(x):
        points = x.reshape(N, 2)
        energy = 0.0
        grad = np.zeros_like(x)
        
        # Pairwise repulsion: strong inverse power
        # We want to maximize min distance, so we penalize close pairs strongly
        # Use a soft potential: for each pair, if d < d0, add penalty
        # But for smooth optimization, use: sum over pairs of max(0, d0 - d)^2
        # Actually, let's use: sum over pairs of 1/d^2 for all pairs (repulsion)
        # plus wall potential: sum over points of max(0, |p| - 1)^2
        
        # Better approach: minimize sum of exp(-alpha * d) for all constraints
        # But let's use the physics approach:
        # Energy = sum_{i<j} k_rep / d_ij^p + sum_i k_wall * max(0, |p_i| - 1)^2
        
        # For smoothness, use:
        # Pair potential: k_rep * exp(-d_ij / sigma) or k_rep / (d_ij^2 + eps)^p
        # Let's use: for each pair, if d < 2*r_target, penalize
        # But we don't know r_target. Instead, use repulsion that pushes apart.
        
        # Use: E_pair = sum_{i<j} 1 / (d_ij^2 + 1e-8)
        # This creates strong repulsion when close.
        
        # Wall potential: E_wall = sum_i (1 - |p_i|)^2 if |p_i| < 1, else (|p_i| - 1)^2
        # Actually, we want points inside, so: E_wall = sum_i max(0, |p_i| - 1)^2
        
        # But this doesn't maximize the minimum radius directly.
        # Let's combine: minimize E = sum_{i<j} f(d_ij) + sum_i g(|p_i|)
        # where f(d) is decreasing in d (repulsion) and g(r) penalizes being outside.
        
        # A better approach: use the log-sum-exp of the constraints.
        # Let's stick with the smooth min approach but with gradient.
        
        # For now, use a simple physics model:
        # Repulsion: for each pair i,j, force = k * (r0 - d_ij) if d_ij < r0, else 0
        # But this is non-smooth. Use: force = k * exp(-(d_ij - r0)^2 / width)
        
        # Actually, let's use the objective_smooth with analytic gradient.
        
        # Compute all distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Boundary distances
        norms = np.linalg.norm(points, axis=1)
        d_bound = 1.0 - norms
        
        # All constraints: pairwise distances and boundary distances
        # We want to maximize min of these.
        # Soft min: -1/alpha * log(sum(exp(-alpha * d_k)))
        
        alpha = 100.0  # Will be annealed
        
        # Collect all distances
        all_dists = np.concatenate([d_bound, dists[~np.eye(N, dtype=bool)]])
        
        # Soft min computation
        min_d = np.min(all_dists)
        shifted = -alpha * (all_dists - min_d)
        sum_exp = np.sum(np.exp(shifted))
        soft_min = min_d - (1.0 / alpha) * np.log(sum_exp)
        
        energy = -soft_min  # We minimize this
        
        # Gradient
        # d(soft_min)/dx = ...
        # This is complex. Let's use finite differences or a simpler approach.
        
        # Alternative: use scipy.optimize with the smooth objective
        # But we want to use physics-based relaxation with gradient descent.
        
        # Let's compute gradient numerically for now (expensive but correct)
        # Actually, let's use a different approach: direct gradient descent on a different energy.
        
        # Physics energy: E = sum_{i<j} 1/d_ij^2 + sum_i (1 - |p_i|)^2 * w
        # where w is large to enforce boundary.
        
        # Recompute with this energy:
        energy = 0.0
        grad = np.zeros_like(x)
        
        # Pairwise: E_pair = sum_{i<j} 1/d_ij^2
        # dE_pair/dp_i = -2 * sum_{j != i} (p_i - p_j) / d_ij^4
        for i in range(N):
            for j in range(i+1, N):
                d_vec = points[i] - points[j]
                d_sq = np.dot(d_vec, d_vec)
                if d_sq < 1e-10:
                    d_sq = 1e-10
                # d(1/d_sq)/dp_i = -2 * d_vec / d_sq^2
                grad[2*i] += -2 * d_vec[0] / (d_sq * d_sq)
                grad[2*i+1] += -2 * d_vec[1] / (d_sq * d_sq)
                grad[2*j] += 2 * d_vec[0] / (d_sq * d_sq)
                grad[2*j+1] += 2 * d_vec[1] / (d_sq * d_sq)
                energy += 1.0 / d_sq
        
        # Wall: E_wall = sum_i max(0, |p_i| - 1)^2
        # But this only penalizes outside. We want to maximize the minimum,
        # so we also want to push points towards the boundary if they're inside.
        # Let's use: E_wall = sum_i (1 - |p_i|)^2 * small_weight + max(0, |p_i| - 1)^2 * large_weight
        # Actually, for circle packing, the optimal has most points touching the boundary.
        # So we want to maximize |p_i| for each point, but also maximize pairwise distances.
        
        # Let's use: E_wall = sum_i (1 - |p_i|)^2 with a positive weight
        # This encourages points to be near the boundary.
        for i in range(N):
            r = norms[i]
            if r < 1e-10:
                # At origin, gradient is undefined, use small random
                grad[2*i] += 0.0
                grad[2*i+1] += 0.0
            else:
                # d(1-r)^2/dr = -2(1-r)
                # dr/dp = p/r
                # So dE/dp = -2(1-r) * p/r
                grad[2*i] += -2 * (1 - r) * points[i, 0] / r
                grad[2*i+1] += -2 * (1 - r) * points[i, 1] / r
            energy += (1 - r) ** 2
        
        return energy, grad

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
        
        # Physics-based gradient descent with annealing
        # Start with large step size, decrease over time
        current_x = x0.copy()
        
        # Annealing schedule: start with high temperature (large steps), cool down
        num_steps = 500
        for step in range(num_steps):
            if time.time() > deadline - 2:
                break
            
            # Learning rate schedule: start large, decay
            lr = 0.01 * (1.0 - step / num_steps) + 0.001
            
            # Compute energy and gradient
            try:
                energy, grad = physics_energy_and_grad(current_x)
                
                # Gradient descent step
                current_x = current_x - lr * grad
                
                # Project back inside the unit circle (soft constraint)
                points = current_x.reshape(N, 2)
                norms = np.linalg.norm(points, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    points[mask] = points[mask] / norms[mask, np.newaxis] * 0.999
                    current_x = points.flatten()
                    
            except Exception:
                break
        
        # Evaluate final result
        curr_points = current_x.reshape(N, 2)
        curr_score = get_score(curr_points)
        
        if curr_score > best_score:
            best_score = curr_score
            best_points = curr_points.copy()

    # If we still have time, do some local polishing on the best point
    if best_points is not None and time.time() < deadline - 5:
        # Use scipy.optimize with the smooth objective for fine tuning
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
            shifted = -alpha * (all_dists - min_d)
            sum_exp = np.sum(np.exp(shifted))
            
            # soft_min = min_d - 1/alpha * log( sum( exp(-alpha * (d_k - min_d)) ) )
            # We want to maximize this. So minimize negative of this.
            obj_val = -(min_d - (1.0 / alpha) * np.log(sum_exp))
            
            return obj_val
        
        for _ in range(5):
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
