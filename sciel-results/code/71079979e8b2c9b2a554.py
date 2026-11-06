import numpy as np
import time
import scipy.optimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
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

    # Smooth objective with analytic gradient for L-BFGS-B
    # We minimize -soft_min of all constraint distances
    # soft_min(d, alpha) = -1/alpha * log(sum(exp(-alpha * d_k)))
    # As alpha -> inf, soft_min -> min
    # We want to maximize min, so minimize -soft_min
    
    def objective_and_grad(x, alpha):
        points = x.reshape(N, 2)
        
        # Boundary distances: 1 - |p_i|
        norms = np.linalg.norm(points, axis=1)
        d_bound = 1.0 - norms
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        
        # Collect all distances
        all_dists = np.concatenate([d_bound, dists[~np.eye(N, dtype=bool)]])
        
        # For numerical stability, find min and shift
        min_d = np.min(all_dists)
        shifted = -alpha * (all_dists - min_d)
        
        # Clip to avoid overflow
        shifted = np.clip(shifted, -700, 700)
        sum_exp = np.sum(np.exp(shifted))
        
        soft_min = min_d - (1.0 / alpha) * np.log(sum_exp)
        
        # We minimize -soft_min
        obj_val = -soft_min
        
        # Gradient computation
        # d(soft_min)/dx_i = (1/sum_exp) * sum_k exp(-alpha*(d_k - min_d)) * (-alpha) * d(d_k)/dx_i
        # But d_k depends on x_i only for constraints involving point i
        # 
        # For boundary constraint k=i: d_k = 1 - |p_i|, d(d_k)/dx_i = -p_i/|p_i|
        # For pairwise constraint k=(i,j): d_k = |p_i - p_j|, d(d_k)/dx_i = (p_i - p_j)/|p_i - p_j|
        #
        # Let's compute weights for each constraint
        # weights_k = exp(-alpha*(d_k - min_d)) / sum_exp
        
        weights = np.exp(shifted) / sum_exp
        
        grad = np.zeros_like(x)
        
        # Boundary contributions
        for i in range(N):
            r = norms[i]
            if r > 1e-10:
                w = weights[i]
                # d(d_bound_i)/dx_i = -p_i / r
                # d(-soft_min)/dx_i = -d(soft_min)/dx_i
                # d(soft_min)/dx_i includes contribution from boundary: 
                #   (1/alpha) * (-alpha) * w_i * d(d_bound_i)/dx_i = -w_i * d(d_bound_i)/dx_i
                # So d(soft_min)/dx_i from boundary = -w_i * (-p_i/r) = w_i * p_i/r
                # d(-soft_min)/dx_i = -w_i * p_i/r
                grad[2*i] += -w * points[i, 0] / r
                grad[2*i+1] += -w * points[i, 1] / r
        
        # Pairwise contributions
        # For each pair (i,j), the constraint distance is d_ij
        # Its weight is weights[N + idx] where idx is the index in the flattened pairwise array
        # Let's compute more carefully
        
        # The pairwise distances are stored in dists with diagonal set to inf
        # The order in all_dists is: [d_bound_0, ..., d_bound_{N-1}, d_01, d_02, ..., d_0_{N-1}, d_12, ...]
        # Let's build a mapping
        
        # Actually, let's recompute pairwise weights directly
        # For pair (i, j), the weight is exp(-alpha*(d_ij - min_d)) / sum_exp
        # We can compute this from the dists matrix
        
        # Create weight matrix for pairwise
        pair_weights = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d_ij = dists[i, j]
                if d_ij < np.inf:
                    w_ij = np.exp(-alpha * (d_ij - min_d)) / sum_exp
                    pair_weights[i, j] = w_ij
                    pair_weights[j, i] = w_ij
        
        # Now compute gradient from pairwise
        for i in range(N):
            for j in range(i+1, N):
                d_ij = dists[i, j]
                if d_ij > 1e-10:
                    w_ij = pair_weights[i, j]
                    # d(d_ij)/dx_i = (p_i - p_j) / d_ij
                    # d(soft_min)/dx_i from pair (i,j) = -w_ij * d(d_ij)/dx_i = -w_ij * (p_i - p_j)/d_ij
                    # d(-soft_min)/dx_i = w_ij * (p_i - p_j)/d_ij
                    diff_vec = points[i] - points[j]
                    grad[2*i] += w_ij * diff_vec[0] / d_ij
                    grad[2*i+1] += w_ij * diff_vec[1] / d_ij
                    grad[2*j] += -w_ij * diff_vec[0] / d_ij
                    grad[2*j+1] += -w_ij * diff_vec[1] / d_ij
        
        return obj_val, grad

    # Physics-based energy for gradient descent (faster per step)
    def physics_energy_and_grad(x):
        points = x.reshape(N, 2)
        energy = 0.0
        grad = np.zeros_like(x)
        
        # Pairwise repulsion: sum 1/d_ij^2
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
        
        # Wall potential: sum (1 - |p_i|)^2 to encourage boundary contact
        norms = np.linalg.norm(points, axis=1)
        for i in range(N):
            r = norms[i]
            if r < 1e-10:
                grad[2*i] += 0.0
                grad[2*i+1] += 0.0
            else:
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

    def random_init():
        while True:
            pts = np.random.uniform(-1, 1, (N, 2))
            norms = np.linalg.norm(pts, axis=1)
            if np.all(norms < 1.0):
                return pts

    # Generate multiple initial configurations
    inits = []
    inits.append(hex_lattice())
    inits.append(ring_based_init())
    inits.append(random_init())
    inits.append(hex_lattice() * 0.9)
    inits.append(random_init())
    
    # Add perturbed versions of good inits
    base_inits = [hex_lattice(), ring_based_init()]
    for base in base_inits:
        for _ in range(2):
            perturbed = base + np.random.normal(0, 0.05, (N, 2))
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 0.95
            if np.any(mask):
                perturbed[mask] *= 0.95 / norms[mask, np.newaxis]
            inits.append(perturbed)

    # Phase 1: Physics-based gradient descent for fast convergence
    for init_idx, init_points in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        current_x = init_points.flatten().copy()
        
        # Gradient descent with adaptive learning rate
        num_steps = 300
        for step in range(num_steps):
            if time.time() > deadline - 10:
                break
            
            # Learning rate schedule: start large, decay
            lr = 0.005 * (1.0 - step / num_steps) + 0.0005
            
            try:
                energy, grad = physics_energy_and_grad(current_x)
                
                # Gradient descent step
                current_x = current_x - lr * grad
                
                # Project back inside the unit circle
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

    # Phase 2: Fine-tuning with L-BFGS-B using smooth objective
    if best_points is not None and time.time() < deadline - 5:
        # Use multiple alpha values for annealing
        alphas = [100.0, 500.0, 2000.0, 10000.0]
        
        current_best = best_points.copy()
        current_score = best_score
        
        for alpha in alphas:
            if time.time() > deadline - 2:
                break
            
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad, 
                    current_best.flatten(), 
                    args=(alpha,),
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                if res.x is not None:
                    curr_points = res.x.reshape(N, 2)
                    # Check validity
                    norms = np.linalg.norm(curr_points, axis=1)
                    if np.all(norms < 1.0) and np.all(np.isfinite(curr_points)):
                        curr_score = get_score(curr_points)
                        if curr_score > current_score:
                            current_score = curr_score
                            current_best = curr_points.copy()
            except Exception:
                pass
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_best.copy()

    # Phase 3: Multi-start with L-BFGS-B if time remains
    if time.time() < deadline - 5:
        # Try a few more random restarts
        for restart in range(3):
            if time.time() > deadline - 2:
                break
            
            c0 = random_init()
            
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad, 
                    c0.flatten(), 
                    args=(500.0,),
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                if res.x is not None:
                    curr_points = res.x.reshape(N, 2)
                    norms = np.linalg.norm(curr_points, axis=1)
                    if np.all(norms < 1.0) and np.all(np.isfinite(curr_points)):
                        curr_score = get_score(curr_points)
                        if curr_score > best_score:
                            best_score = curr_score
                            best_points = curr_points.copy()
            except Exception:
                pass

    # Final validation
    if best_points is None:
        # Fallback to a simple valid configuration
        best_points = np.zeros((N, 2))
        # Place points in a grid inside the unit circle
        spacing = 0.15
        points = []
        for i in range(-10, 11):
            for j in range(-10, 11):
                x = i * spacing
                y = j * spacing
                if np.sqrt(x*x + y*y) < 0.9:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            best_points = points[:N]
        elif len(points) < N:
            # Add random points
            while len(points) < N:
                p = np.random.uniform(-0.9, 0.9, 2)
                if np.linalg.norm(p) < 0.9:
                    points = np.vstack([points, p])
            best_points = points[:N]
        else:
            best_points = points
    
    # Ensure all points are strictly inside the unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_points[mask] *= 0.999 / norms[mask, np.newaxis]
    
    return best_points
