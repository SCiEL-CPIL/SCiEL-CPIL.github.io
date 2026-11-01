import numpy as np
import time
from scipy.spatial.distance import cdist

N = 300

def construct():
    # Time management
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer
    
    # 1. Initialization: Hexagonal Lattice
    # Try to fit as many points as possible in a hex lattice inside unit circle
    # Then select N points closest to center or use a specific N-point lattice
    
    # Alternative Init: Concentric Rings
    # For N=300, rings are often good.
    # Let's try a hybrid: Start with a hex lattice, then anneal.
    
    def hex_init():
        # Estimate spacing for N points in unit circle
        # Area of unit circle = pi. Area per point = pi/N.
        # Hex packing density is pi/sqrt(12) ~ 0.9069
        # Effective area per point in hex lattice ~ (2*r)^2 * sqrt(3)/2
        # Let's just generate a dense grid and pick N.
        
        # Spacing s such that ~N points fit
        # Number of points ~ pi/s^2 * (2/s+1)^2 roughly
        # Let's try s = sqrt(pi/N) * 1.1
        s = np.sqrt(np.pi / N) * 1.05
        
        # Generate grid
        k = int(2.0 / s) + 2
        x, y = np.meshgrid(np.arange(-k, k+1)*s, np.arange(-k, k+1)*s)
        # Offset odd rows
        for j in range(x.shape[0]):
            if j % 2 == 1:
                x[j, :] += s / 2.0
        
        points = np.column_stack((x.ravel(), y.ravel()))
        r = np.hypot(points[:, 0], points[:, 1])
        
        # Keep points inside circle with margin
        margin = s * 0.5
        points = points[r <= 1.0 - margin]
        
        if len(points) < N:
            # If not enough, reduce spacing
            while len(points) < N:
                s *= 0.95
                k = int(2.0 / s) + 2
                x, y = np.meshgrid(np.arange(-k, k+1)*s, np.arange(-k, k+1)*s)
                for j in range(x.shape[0]):
                    if j % 2 == 1:
                        x[j, :] += s / 2.0
                points = np.column_stack((x.ravel(), y.ravel()))
                r = np.hypot(points[:, 0], points[:, 1])
                points = points[r <= 1.0 - s * 0.5]
        
        # Select N points closest to center
        idx = np.argsort(r)[:N]
        return points[idx]
    
    points = hex_init()
    
    # 2. Simulated Annealing
    # Objective: Maximize min_dist = min( min_{i<j} |p_i - p_j|/2, min_i (1 - |p_i|) )
    
    def get_min_dist(pts):
        # Compute all pairwise distances
        # For N=300, cdist is fast
        d = cdist(pts, pts)
        # Mask diagonal
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        # Distance to boundary
        dist_to_boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
        min_boundary_dist = np.min(dist_to_boundary)
        
        return min(min_pair_dist / 2.0, min_boundary_dist)
    
    current_r = get_min_dist(points)
    best_points = points.copy()
    best_r = current_r
    
    T = 0.01  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.999
    
    # We can also use a force-based method, but SA is robust.
    # To speed up, we can move multiple points or use larger steps.
    
    # Optimization: Use a simple repulsion force to guide moves?
    # No, standard SA is safer.
    
    # To improve speed, we can pre-compute indices of close pairs?
    # For now, full cdist is O(N^2) ~ 90k ops. In Python/NumPy, this is ~1ms.
    # 140s / 1ms = 140,000 iterations. This should be enough for significant improvement.
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random point
        i = np.random.randint(N)
        # Move it
        step = T * 0.5
        dx = np.random.normal(0, step)
        dy = np.random.normal(0, step)
        
        old_x, old_y = points[i, 0], points[i, 1]
        new_x, new_y = old_x + dx, old_y + dy
        
        # Check if new position is way outside (quick reject)
        if new_x**2 + new_y**2 > 4.0: # Way outside
            T *= cooling_rate
            continue
            
        points[i, 0] = new_x
        points[i, 1] = new_y
        
        new_r = get_min_dist(points)
        
        # Accept if better or with probability exp((new_r - current_r)/T)
        if new_r >= current_r:
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_points = points.copy()
        else:
            delta = new_r - current_r
            if np.random.rand() < np.exp(delta / T):
                current_r = new_r
            else:
                # Reject
                points[i, 0] = old_x
                points[i, 1] = old_y
                
        T *= cooling_rate
        
    # 3. Final Polish with Local Search
    # Use scipy minimize on the negative min distance
    # Since min distance is non-smooth, we can use a soft-min approximation or just coordinate descent.
    
    # Simple coordinate descent polish
    def polish(pts):
        pts = pts.copy()
        for _ in range(50): # 50 passes
            improved = False
            for i in range(N):
                # Try to move point i to maximize local min distance
                # This is complex. Instead, let's use a simple gradient step on a soft-min.
                pass
        return pts
        
    # Better Polish: Use scipy.optimize.minimize with a smooth surrogate
    # Surrogate: -log-sum-exp(-k * distances)
    # This is smooth and differentiable.
    
    def soft_min_dist(pts, k=100.0):
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        # Soft min of pairwise distances / 2
        # log-sum-exp trick for min: -1/k * log(sum(exp(-k*x)))
        # We want min(d/2, 1-r)
        # Let's just maximize the soft-min of all constraints.
        
        # Pairwise constraints
        d_half = d / 2.0
        # Mask diagonal with large value to ignore in sum
        d_half[~np.eye(N, dtype=bool)] = d_half[~np.eye(N, dtype=bool)]
        # Actually, fill diagonal with inf before log-sum-exp is tricky if inf is there.
        # Let's set diagonal to a large number.
        d_half[np.diag_indices(N)] = 100.0
        
        # Soft min of d_half
        max_d = np.max(d_half)
        log_sum_exp = max_d + np.log(np.sum(np.exp(d_half - max_d)))
        soft_min_pair = -1.0 / k * log_sum_exp
        
        # Boundary constraints
        r = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r
        max_b = np.max(boundary)
        log_sum_exp_b = max_b + np.log(np.sum(np.exp(boundary - max_b)))
        soft_min_boundary = -1.0 / k * log_sum_exp_b
        
        return min(soft_min_pair, soft_min_boundary)
        
    def neg_soft_min(pts_flat, k=100.0):
        pts = pts_flat.reshape(N, 2)
        return -soft_min_dist(pts, k)
        
    # Run a few iterations of L-BFGS-B
    if time.time() - start_time < time_limit - 5:
        try:
            from scipy.optimize import minimize
            x0 = best_points.ravel()
            # Use a few steps
            res = minimize(neg_soft_min, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.success or res.fun < neg_soft_min(x0):
                candidate = res.x.reshape(N, 2)
                # Verify validity and score
                cand_r = get_min_dist(candidate)
                if cand_r > best_r:
                    best_r = cand_r
                    best_points = candidate
        except:
            pass
            
    return best_points

