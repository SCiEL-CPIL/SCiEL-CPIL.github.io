"""Pack 50 equal circles in a unit square using Hard-Disk Expansion + SLSQP Epigraph."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall radius
    wall_r = np.min(0.5 - np.abs(centers))
    if wall_r < 0:
        return 0.0
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _init_hexagonal():
    """
    Generate a high-quality hexagonal packing initialization.
    We want to place N points in a hex lattice that fits well in the square.
    """
    # Try different grid sizes to find one that yields at least N points
    # and is compact.
    # A hex grid with spacing dx, dy=sqrt(3)/2*dx.
    # Area per point ~ dx * sqrt(3)/2 * dx = dx^2 * sqrt(3)/2.
    # For 50 points in area 1, avg area 0.02.
    # dx^2 * 0.866 = 0.02 => dx^2 = 0.023 => dx = 0.15.
    
    best_points = None
    best_score = -1
    
    for trial in range(20):
        # Randomly perturb the grid spacing and offset to explore different packings
        dx = 0.14 + np.random.uniform(-0.02, 0.02)
        dy = dx * np.sqrt(3) / 2.0
        
        # Generate points in a range slightly larger than the square
        x_min, x_max = -0.6, 0.6
        y_min, y_max = -0.6, 0.6
        
        points = []
        row = 0
        y = y_min
        while y <= y_max:
            x_start = x_min
            if row % 2 == 1:
                x_start += dx / 2.0
            
            x = x_start
            while x <= x_max:
                points.append([x, y])
                x += dx
            y += dy
            row += 1
            
        points = np.array(points)
        
        # Keep only points strictly inside the square with a small margin to ensure validity
        # Actually, we can keep points on the boundary, but for init, let's keep them inside.
        mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & \
               (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        points = points[mask]
        
        if len(points) < N:
            continue
            
        # If we have more than N points, we need to select N.
        # Strategy: Select the N points that maximize the minimum distance to other points?
        # Or simply select a central block?
        # A central block might be better for expansion.
        # Let's sort by distance from center and take the first N?
        # This creates a circular blob, which might be good.
        
        # Alternative: Randomly drop points from the periphery until N remain.
        # This preserves the local hex structure better.
        
        if len(points) > N:
            # Calculate distance from center for each point
            dists = np.linalg.norm(points, axis=1)
            # Sort indices by distance
            idx = np.argsort(dists)
            # Take the first N
            points = points[idx[:N]]
            
        # Evaluate the quality of this init (current radius)
        r = _radius(points)
        if r > best_score:
            best_score = r
            best_points = points.copy()
            
    if best_points is None:
        # Fallback: simple grid
        ticks = (np.arange(7) + 0.5) / 7 - 0.5
        points = np.array([(x, y) for y in ticks for x in ticks])[:N]
        return points
        
    return best_points

def _expand_radius(centers, time_limit=5.0):
    """
    Incrementally increase the radius, resolving collisions by pushing points apart.
    This is a "hard disk" simulation where we try to fit larger disks.
    """
    centers = centers.copy()
    r = _radius(centers)
    start_time = time.time()
    
    # We will try to increase r by small steps.
    # If a collision occurs, we push the colliding pair apart.
    # We repeat this until we can't increase r anymore or time runs out.
    
    # Initial step size
    step = 0.0005
    min_step = 1e-6
    
    # To speed up, we can use a larger step initially if the current r is small
    # But let's stick to a robust approach.
    
    # We need to handle the "wall" constraints too.
    # The effective radius is limited by both walls and other circles.
    
    # We'll simulate a "pressure" increase.
    # Instead of a fixed step, we can try to find the next collision.
    # But for simplicity, we use a fixed small step and check.
    
    # Optimization: Only check pairs that are close.
    # For N=50, O(N^2) is 2500 ops, which is fast.
    
    while time.time() - start_time < time_limit:
        # Try to increase r
        new_r = r + step
        
        # Check if new_r is feasible.
        # If not, we need to move points.
        
        # 1. Check walls
        # If any point is too close to wall, move it in.
        # 2. Check pairs
        # If any pair is closer than 2*new_r, push them apart.
        
        # We will perform a few iterations of "relaxation" at this new_r
        # before accepting or rejecting.
        
        feasible = True
        # We do a few relaxation steps
        for _ in range(10):
            # Move points away from walls
            # x < -0.5 + new_r -> x = -0.5 + new_r
            # x > 0.5 - new_r -> x = 0.5 - new_r
            # y < -0.5 + new_r -> y = -0.5 + new_r
            # y > 0.5 - new_r -> y = 0.5 - new_r
            
            # This is a hard constraint projection.
            # Let's use soft forces for stability.
            
            forces = np.zeros_like(centers)
            
            # Wall forces
            # Left
            mask_l = centers[:, 0] < -0.5 + new_r
            if np.any(mask_l):
                pen = (-0.5 + new_r) - centers[mask_l, 0]
                forces[mask_l, 0] += pen * 10.0  # Stiffness
            
            # Right
            mask_r = centers[:, 0] > 0.5 - new_r
            if np.any(mask_r):
                pen = centers[mask_r, 0] - (0.5 - new_r)
                forces[mask_r, 0] -= pen * 10.0
            
            # Bottom
            mask_b = centers[:, 1] < -0.5 + new_r
            if np.any(mask_b):
                pen = (-0.5 + new_r) - centers[mask_b, 1]
                forces[mask_b, 1] += pen * 10.0
            
            # Top
            mask_t = centers[:, 1] > 0.5 - new_r
            if np.any(mask_t):
                pen = centers[mask_t, 1] - (0.5 - new_r)
                forces[mask_t, 1] -= pen * 10.0
            
            # Pair forces
            for i in range(N):
                for j in range(i + 1, N):
                    diff = centers[i] - centers[j]
                    dist_sq = np.dot(diff, diff)
                    min_dist = 2.0 * new_r
                    if dist_sq < min_dist * min_dist:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-8:
                            # Random nudge
                            forces[i] += np.random.normal(0, 0.01, 2)
                            forces[j] -= np.random.normal(0, 0.01, 2)
                        else:
                            # Push apart
                            # Force magnitude proportional to penetration
                            pen = min_dist - dist
                            force_mag = pen * 10.0
                            force_vec = force_mag * (diff / dist)
                            forces[i] += force_vec
                            forces[j] -= force_vec
            
            # Update positions
            # Use a simple gradient step with damping
            centers += forces * 0.1
            # Damping? Maybe not necessary if forces are small.
            # Let's add a small damping to prevent oscillation
            # Actually, this is a projection-like step.
            
            # Clip to square to be safe
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
            
        # After relaxation, check if the configuration is valid for new_r
        # Calculate the actual max radius
        actual_r = _radius(centers)
        
        if actual_r >= new_r - 1e-9:
            # Success! Increase r
            r = actual_r
            # Keep step size or increase slightly?
            # If we succeed often, we can increase step.
            # If we fail, decrease step.
            # Let's keep it constant for simplicity, or adapt.
            # step = step * 1.1
        else:
            # Failure. Decrease step and try again?
            # Or just stop?
            # Let's decrease step and try to recover.
            step *= 0.9
            if step < min_step:
                break
            # Revert to previous state? We modified centers.
            # We should keep a backup.
            # This is getting complex. Let's simplify.
            
            # Instead of complex backtracking, let's just stop if we can't improve
            # with the current step after a few tries.
            # For now, if we fail, we just break out and rely on SLSQP.
            # The expansion is just to get close to the optimum.
            break
            
    return centers

def _slsqp_polish(centers, time_limit=10.0):
    """
    Use SLSQP to maximize the minimum distance (radius).
    This is an epigraph formulation.
    Maximize r
    Subject to:
    d_ij >= 2r for all i<j
    0.5 - x_i >= r
    x_i + 0.5 >= r
    0.5 - y_i >= r
    y_i + 0.5 >= r
    """
    x0 = centers.ravel()
    start_time = time.time()
    
    # Variables: x_0...x_{N-1}, y_0...y_{N-1}, r
    # Total 2N + 1 variables.
    
    def objective(x):
        # We want to maximize r, so minimize -r
        return -x[-1]
        
    def gradient(x):
        # Gradient of -r is 0 for all position vars, -1 for r
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad
        
    # Constraints
    # SLSQP expects g(x) >= 0
    # 1. Pairwise distances: d_ij - 2r >= 0
    # 2. Wall distances: 0.5 - |x_i| - r >= 0 (actually 4 constraints per point)
    
    # We define a single constraint function that returns an array of all constraints.
    
    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        cons = []
        
        # Wall constraints
        # x_i >= -0.5 + r  => x_i - r + 0.5 >= 0
        # x_i <= 0.5 - r   => -x_i - r + 0.5 >= 0
        # y_i >= -0.5 + r  => y_i - r + 0.5 >= 0
        # y_i <= 0.5 - r   => -y_i - r + 0.5 >= 0
        
        cons.append(c[:, 0] - r + 0.5)
        cons.append(-c[:, 0] - r + 0.5)
        cons.append(c[:, 1] - r + 0.5)
        cons.append(-c[:, 1] - r + 0.5)
        
        # Pairwise constraints
        # d_ij - 2r >= 0
        # We need to add these one by one or in a batch.
        # SLSQP accepts a list of dicts or a single dict with 'fun' returning array.
        # Let's return a flat array.
        
        pair_cons = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(c[i] - c[j])
                pair_cons.append(d - 2 * r)
                
        cons.append(np.array(pair_cons))
        
        # Flatten
        return np.concatenate(cons)
        
    # Bounds
    # x, y in [-0.5, 0.5]
    # r in [0, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0, 0.5)]
    
    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            constraints={'type': 'ineq', 'fun': constraints},
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-9, 'disp': False}
        )
        
        if res.success or res.x is not None:
            new_centers = res.x[:2*N].reshape(N, 2)
            # Ensure validity
            if _radius(new_centers) > 0:
                return new_centers
    except Exception as e:
        pass
        
    return centers

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    np.random.seed(42)
    
    # 1. Initialization
    centers = _init_hexagonal()
    
    # 2. Hard-Disk Expansion
    time_left = DEADLINE - 30 - (time.time() - start_time)
    if time_left > 5:
        centers = _expand_radius(centers, time_limit=min(15.0, time_left))
    
    r_current = _radius(centers)
    best_r = r_current
    best_centers = centers.copy()
    
    # 3. SLSQP Polish
    time_left = DEADLINE - 10 - (time.time() - start_time)
    if time_left > 5:
        polished = _slsqp_polish(centers, time_limit=min(20.0, time_left))
        r_pol = _radius(polished)
        
        if r_pol > best_r:
            best_r = r_pol
            best_centers = polished.copy()
            
    # 4. Final check and return
    if best_centers is None:
        best_centers = _init_hexagonal()
        
    # Ensure the returned centers are valid and within bounds
    # Clip to square
    best_centers[:, 0] = np.clip(best_centers[:, 0], -0.5, 0.5)
    best_centers[:, 1] = np.clip(best_centers[:, 1], -0.5, 0.5)
    
    return best_centers

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    r = _radius(res)
    print(f"Time: {t1-t0:.2f}s, Radius: {r}")
