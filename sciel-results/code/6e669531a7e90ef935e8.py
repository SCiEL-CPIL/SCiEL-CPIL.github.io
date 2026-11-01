import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

def _compute_radius(centers):
    """Compute the radius for a given set of centers."""
    # Distance to walls
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    
    # Distance between centers
    tree = cKDTree(centers)
    pairs, dists = tree.query_pairs(2.0, output_type='ndist')
    if len(dists) == 0:
        min_dist = float('inf')
    else:
        min_dist = np.min(dists)
    
    return min(min_dist / 2.0, min_wall)

def _violation(centers, r):
    """
    Compute violation: how much the circles exceed radius r or go out of bounds.
    We want to minimize this.
    violation = max(0, r - min_dist/2, r - min_wall_dist)
    Actually, let's define it as:
    v = max(0, max_i (r - (0.5 - |c_i|_inf)), max_{i<j} (2r - |c_i - c_j|))
    """
    # Wall violation
    wall_viol = np.max(np.abs(centers), axis=1) + r - 0.5
    wall_viol = np.max(wall_viol)
    
    # Pair violation
    tree = cKDTree(centers)
    pairs, dists = tree.query_pairs(2*r + 0.1, output_type='ndist')
    if len(dists) == 0:
        pair_viol = -np.inf
    else:
        pair_viol = np.max(2*r - dists)
    
    return max(wall_viol, pair_viol)

def _grad_violation(centers, r, eps=1e-6):
    """
    Compute gradient of the violation function.
    This is a bit tricky because of the max operations.
    We can use subgradients or just numerical gradient.
    Given time constraints, numerical gradient might be safer but slower.
    Let's try analytical subgradients.
    
    The violation is V(c) = max( V_wall(c), V_pair(c) )
    V_wall(c) = max_i ( |c_ix| + |c_iy| + r - 0.5 )
    V_pair(c) = max_{i<j} ( 2r - ||c_i - c_j|| )
    
    If V_wall > V_pair, we only need gradient of V_wall.
    If V_pair > V_wall, we only need gradient of V_pair.
    If equal, sum of gradients.
    
    Gradient of V_wall:
    For each i, let m_i = max(|c_ix|, |c_iy|).
    dV_wall/dc_ix = sign(c_ix) if |c_ix| > |c_iy|, 0 otherwise (subgradient)
    dV_wall/dc_iy = sign(c_iy) if |c_iy| > |c_ix|, 0 otherwise
    
    Actually, for the max over i, we only take the gradient for the argmax i.
    
    Gradient of V_pair:
    For pair (i,j), let d = ||c_i - c_j||.
    dV_pair/dc_i = -(c_i - c_j) / d
    dV_pair/dc_j = (c_i - c_j) / d
    
    For the max over pairs, we take the gradient for the argmax pair.
    """
    grad = np.zeros_like(centers)
    
    # Calculate components
    wall_viol_arr = np.abs(centers).max(axis=1) + r - 0.5
    max_wall_val = np.max(wall_viol_arr)
    wall_argmax_idx = np.argmax(wall_viol_arr)
    
    tree = cKDTree(centers)
    pairs, dists = tree.query_pairs(2*r + 0.1, output_type='ndist')
    
    if len(dists) == 0:
        pair_viol_val = -np.inf
        pair_argmax_idx = -1
    else:
        pair_viol_arr = 2*r - dists
        max_pair_val = np.max(pair_viol_arr)
        pair_argmax_idx = np.argmax(pair_viol_arr)
    
    # Determine which term is active
    if max_wall_val > max_pair_val:
        # Only wall term active
        i = wall_argmax_idx
        cx, cy = centers[i]
        if abs(cx) > abs(cy):
            grad[i, 0] = np.sign(cx)
        elif abs(cy) > abs(cx):
            grad[i, 1] = np.sign(cy)
        else:
            # On the diagonal, subgradient is ambiguous. Choose (1,1)/sqrt(2) or similar
            # Actually, for L-infinity norm, subgradient is any unit vector in L1 ball.
            # Let's just pick (sign(cx), 0) or (0, sign(cy)) arbitrarily
            grad[i, 0] = np.sign(cx) if cx != 0 else 1.0
            grad[i, 1] = np.sign(cy) if cy != 0 else 1.0
            # Normalize? No, subgradient of max is just the subgradient of the active term.
            # The subgradient of ||x||_inf is:
            # if |x1| > |x2|: (sign(x1), 0)
            # if |x2| > |x1|: (0, sign(x2))
            # if |x1| == |x2|: (sign(x1)*alpha, sign(x2)*(1-alpha)) for alpha in [0,1]
            # Let's use (0.5*sign(cx), 0.5*sign(cy)) normalized? No.
            # Let's just use (sign(cx), sign(cy)) / 2 to be safe?
            # Actually, any vector in the convex hull of the extreme subgradients works.
            # Extreme subgradients at (1,1) are (1,0) and (0,1).
            # So (0.5, 0.5) is a valid subgradient.
            grad[i, 0] = 0.5 * np.sign(cx) if cx != 0 else 0.5
            grad[i, 1] = 0.5 * np.sign(cy) if cy != 0 else 0.5
            
    elif max_pair_val > max_wall_val:
        # Only pair term active
        i, j = pairs[pair_argmax_idx]
        d = dists[pair_argmax_idx]
        if d > 1e-12:
            diff = centers[i] - centers[j]
            unit = diff / d
            grad[i] -= unit
            grad[j] += unit
        else:
            # Coincident points, arbitrary direction
            grad[i] = np.random.normal(0, 1, 2)
            grad[j] = -grad[i]
    else:
        # Both active
        # Wall part
        i = wall_argmax_idx
        cx, cy = centers[i]
        if abs(cx) > abs(cy):
            grad[i, 0] += np.sign(cx)
        elif abs(cy) > abs(cx):
            grad[i, 1] += np.sign(cy)
        else:
            grad[i, 0] += 0.5 * np.sign(cx) if cx != 0 else 0.5
            grad[i, 1] += 0.5 * np.sign(cy) if cy != 0 else 0.5
            
        # Pair part
        i, j = pairs[pair_argmax_idx]
        d = dists[pair_argmax_idx]
        if d > 1e-12:
            diff = centers[i] - centers[j]
            unit = diff / d
            grad[i] -= unit
            grad[j] += unit
        else:
            grad[i] += np.random.normal(0, 1, 2)
            grad[j] -= grad[i]
            
    return grad

def _project_to_square(centers):
    """Project centers back into the square [-0.5, 0.5]^2."""
    centers = np.clip(centers, -0.5, 0.5)
    return centers

def _hex_seed():
    """Generate a hexagonal lattice seed."""
    r_est = 0.07
    spacing = 2 * r_est
    v1 = np.array([spacing, 0])
    v2 = np.array([spacing/2, spacing * np.sqrt(3)/2])
    
    points = []
    max_idx = 20
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            p = i * v1 + j * v2
            if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5:
                points.append(p)
                
    points = np.array(points)
    
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        centers = points[idx]
    else:
        centers = np.vstack([points, np.random.uniform(-0.5, 0.5, size=(N - len(points), 2))])
        
    return centers

def _grid_seed():
    """Generate a grid seed."""
    # 7x8 grid = 56 points, pick 50
    # Or 5x10 = 50
    # Let's try 5x10
    xs = np.linspace(-0.45, 0.45, 5)
    ys = np.linspace(-0.45, 0.45, 10)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    if len(points) > N:
        idx = np.random.choice(len(points), N, replace=False)
        centers = points[idx]
    else:
        centers = np.vstack([points, np.random.uniform(-0.5, 0.5, size=(N - len(points), 2))])
        
    return centers

def _sa_search(initial_centers, deadline, start_T=0.01, end_T=1e-6, alpha=0.999):
    """Simulated Annealing search."""
    current_centers = initial_centers.copy()
    current_r = _compute_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r
    
    T = start_T
    step_size = 0.02
    
    while time.time() < deadline:
        # Pick random circle
        i = np.random.randint(N)
        
        # Generate move
        delta = np.random.normal(0, step_size, 2)
        new_centers = current_centers.copy()
        new_centers[i] += delta
        new_centers = _project_to_square(new_centers)
        
        new_r = _compute_radius(new_centers)
        
        # Acceptance criterion
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            prob = np.exp((new_r - current_r) / T)
            if np.random.random() < prob:
                current_centers = new_centers
                current_r = new_r
        
        T *= alpha
        if T < end_T:
            T = end_T
            
        # Adaptive step size? Maybe reduce slowly
        # step_size *= 0.99999
        
    return best_centers, best_r

def _gradient_polish(centers, deadline, lr=0.01):
    """Polish the solution using gradient descent on the violation function."""
    current_centers = centers.copy()
    best_centers = centers.copy()
    best_r = _compute_radius(current_centers)
    
    # Binary search for the maximum feasible radius
    # Start with current r
    r_low = best_r
    r_high = 0.1 # Upper bound
    
    # First, find a feasible r
    # We can try to increase r slightly and see if we can fix violations
    # Or just use the current r and try to improve it
    
    # Let's do a few iterations of gradient descent for a fixed r, then increase r
    # Actually, let's just minimize the violation for a target r
    
    # Target r slightly above current best
    target_r = best_r * 1.01
    
    # Gradient descent
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        v = _violation(current_centers, target_r)
        
        if v < 1e-8:
            # Feasible for target_r
            # Try higher target_r
            r_low = target_r
            target_r *= 1.01
            continue
            
        grad = _grad_violation(current_centers, target_r)
        
        # Gradient descent step
        current_centers -= lr * grad
        current_centers = _project_to_square(current_centers)
        
        # Check if we found a better feasible solution
        current_r = _compute_radius(current_centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = current_centers.copy()
            
    return best_centers, best_r

def construct():
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin for overhead/return
    
    # Try multiple seeds
    best_centers = None
    best_r = -np.inf
    
    seeds = [_hex_seed(), _grid_seed(), np.random.uniform(-0.45, 0.45, size=(N, 2))]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
            
        # Run SA
        centers, r = _sa_search(seed, deadline - 20)
        
        if r > best_r:
            best_r = r
            best_centers = centers
            
    # Polish the best result
    if best_centers is not None:
        best_centers, best_r = _gradient_polish(best_centers, deadline)
        
    return best_centers
