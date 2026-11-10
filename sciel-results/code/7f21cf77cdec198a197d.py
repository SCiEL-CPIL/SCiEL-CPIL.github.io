import numpy as np
import time
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 30  # 570 seconds to be safe

    def get_distances(points):
        """Calculate pairwise distances for a (N, 2) array using pdist."""
        return scipy.spatial.distance.pdist(points)

    def objective(x, temp=1.0):
        """
        Objective function to minimize.
        We want to maximize R = (d_min / d_max)^2.
        So we minimize -R.
        x is a flat array of size 32.
        temp is used for softmin/softmax to smooth the objective.
        """
        points = x.reshape(n, d)
        dists = get_distances(points)
        
        # Avoid division by zero or log(0)
        dists = np.maximum(dists, 1e-12)
        
        # Use softmin and softmax to create a smooth approximation of min and max
        # As temp -> 0, softmin -> min, softmax -> max
        # We want to maximize (d_min / d_max).
        # log(d_min) - log(d_max) = log(d_min/d_max)
        # softmin(dists) ~ -1/temp * log(sum(exp(-dists/temp)))
        # softmax(dists) ~ 1/temp * log(sum(exp(dists/temp)))
        
        # To avoid overflow in exp, we shift by max/min
        d_max_raw = np.max(dists)
        d_min_raw = np.min(dists)
        
        # Stable log-sum-exp for softmin
        # softmin = -1/temp * log(sum(exp(-dists/temp)))
        # = -1/temp * log(sum(exp(-(dists - d_min)/temp))) + d_min
        # Wait, standard softmin:
        # m = min(dists)
        # softmin = m - 1/temp * log(sum(exp(-(dists-m)/temp)))
        
        # Standard softmax:
        # M = max(dists)
        # softmax = M + 1/temp * log(sum(exp((dists-M)/temp)))
        
        m = d_min_raw
        M = d_max_raw
        
        # If m is very small, ratio is 0.
        if m < 1e-12:
             return 0.0

        # Calculate softmin and softmax
        # Note: for large temp, this approximates mean. For small temp, approximates min/max.
        
        # We want to maximize (softmin / softmax)
        # log(softmin/softmax) = log(softmin) - log(softmax)
        
        # However, simply using log ratio might be unstable if softmin is close to 0.
        # Let's stick to the ratio of the smoothed values, but compute carefully.
        
        # Actually, a common trick is to maximize the sum of logs of the smallest k distances
        # and minimize the sum of logs of the largest k distances.
        # But here we have just min and max.
        
        # Let's use the smoothed values directly.
        # softmin_val = m - (1/temp) * np.log(np.sum(np.exp(-(dists - m) / temp)))
        # softmax_val = M + (1/temp) * np.log(np.sum(np.exp((dists - M) / temp)))
        
        # To avoid underflow/overflow in exp:
        # For softmin: exp(-(dists-m)/temp) is between exp(-(M-m)/temp) and 1.
        # For softmax: exp((dists-M)/temp) is between exp(-(M-m)/temp) and 1.
        
        # If M-m is large and temp is small, exp(-(M-m)/temp) underflows to 0. This is fine.
        
        softmin_val = m - (1.0/temp) * np.log(np.sum(np.exp(-(dists - m) / temp)))
        softmax_val = M + (1.0/temp) * np.log(np.sum(np.exp((dists - M) / temp)))
        
        if softmax_val == 0:
            return 0.0
            
        ratio = softmin_val / softmax_val
        return -ratio

    # 1. Generate diverse initial candidates
    # Candidate 1: Hexagonal Lattice (often optimal for packing)
    # 16 points in a hex lattice
    # We can generate a larger hex grid and take the 16 closest to center
    # Or just construct a specific hex-like arrangement.
    # Let's use a grid of hex points.
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            # Hex grid coordinates to cartesian
            x = i + 0.5 * (j if j % 2 == 0 else 0) # Simplified hex mapping
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Center it
    hex_pts -= np.mean(hex_pts, axis=0)
    # Sort by distance to center and take 16
    dists_to_center = np.sum(hex_pts**2, axis=1)
    indices = np.argsort(dists_to_center)[:16]
    hex_init = hex_pts[indices]
    
    # Candidate 2: Multi-ring
    # 1 center, 6 inner ring, 9 outer ring? 1+6+9=16.
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
    ring1 = np.column_stack((np.cos(angles_inner), np.sin(angles_inner))) * 1.0
    angles_outer = np.linspace(0, 2*np.pi, 9, endpoint=False)
    ring2 = np.column_stack((np.cos(angles_outer), np.sin(angles_outer))) * 2.0
    multi_ring_init = np.vstack([center, ring1, ring2])
    
    # Candidate 3: Random
    np.random.seed(42)
    rand_init = np.random.randn(n, d) * 2
    
    candidates = [hex_init, multi_ring_init, rand_init]
    
    best_points = None
    best_score = -np.inf
    
    # 2. Simulated Annealing for each candidate
    # Parameters
    T_start = 1.0
    T_end = 1e-6
    alpha = 0.999 # Cooling rate
    step_size = 0.5
    
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline:
            break
            
        current_pts = init_pts.copy()
        current_score = -objective(current_pts.flatten(), T_start)
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
            
        T = T_start
        step = step_size
        
        # Run SA
        # We need to balance iterations and time.
        # Each objective call takes ~1-2 ms for pdist(16).
        # We can do ~10000-50000 iterations in a few seconds.
        
        while T > T_end:
            if time.time() > deadline:
                break
                
            # Generate new point
            # Perturb a random subset of points
            num_perturb = np.random.randint(1, 5)
            indices = np.random.choice(n, num_perturb, replace=False)
            
            new_pts = current_pts.copy()
            # Gaussian perturbation
            noise = np.random.randn(num_perturb, d) * step
            new_pts[indices] += noise
            
            new_score = -objective(new_pts.flatten(), T)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0:
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_pts.copy()
            else:
                # Accept with probability exp(delta / T)
                # delta is negative
                if np.random.rand() < np.exp(delta / T):
                    current_pts = new_pts
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_pts.copy()
                        
            # Cool down
            T *= alpha
            # Reduce step size slightly to refine
            step *= 0.9995
            
    # 3. Local Refinement with Nelder-Mead on the best found points
    # Use the raw min/max objective (temp=0 effectively, or very small)
    def final_objective(x):
        points = x.reshape(n, d)
        dists = get_distances(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max)
        
    if best_points is not None:
        try:
            res_local = scipy.optimize.minimize(
                final_objective, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if -res_local.fun > best_score:
                best_score = -res_local.fun
                best_points = res_local.x.reshape(n, d)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = hex_init

    return best_points
