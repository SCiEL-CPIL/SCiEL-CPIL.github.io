"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


def _radius(centers):
    """Compute the maximum common radius for given centers (vectorized)."""
    n = len(centers)
    if n < 2:
        wall = 0.5 - np.max(np.abs(centers), axis=1)
        return np.min(wall)
    # Pairwise distances (upper triangle)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    # Distance to walls
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)

def _negative_radius(centers):
    """Objective to minimize: negative radius."""
    return -_radius(centers)

def _bcc_init(n_points):
    """Generate initial points on a distorted BCC lattice."""
    # BCC lattice points in a unit cell: corners (0,0,0) etc and body center (0.5,0.5,0.5)
    # We want to place points in [-0.5, 0.5]^3.
    # Let's create a 3x3x3 grid of BCC cells.
    # Each cell has 2 points. Total 3*3*3*2 = 54 points. We need 50.
    
    # BCC basis vectors relative to cell origin (0,0,0)
    # Standard BCC: (0,0,0) and (0.5,0.5,0.5)
    
    # Let's just generate a 3x3x3 grid of cells, each with 2 points.
    # Scale to fit in [-0.5, 0.5].
    # A 3x3x3 grid of BCC cells has extent 3 in each dimension (in cell units).
    # If we map cell units to [-0.5, 0.5], the scale is 1/3.
    # So cell size is 1/3.
    
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Cell origin in cube coordinates
                ox = -0.5 + i * (1.0/3.0)
                oy = -0.5 + j * (1.0/3.0)
                oz = -0.5 + k * (1.0/3.0)
                
                # Two points in this cell
                # Corner: (ox, oy, oz)
                # Center: (ox + 1/6, oy + 1/6, oz + 1/6)
                
                points.append([ox, oy, oz])
                points.append([ox + 1.0/6.0, oy + 1.0/6.0, oz + 1.0/6.0])
                
    points = np.array(points)
    
    # We have 54 points, need 50.
    # Remove 4 points. Which ones?
    # The corner points at the very corners of the cube might be close to walls.
    # Let's remove the 4 points that are furthest from the center? Or just the last 4?
    # Actually, removing corner points from the BCC lattice might be bad if they are part of the densest packing.
    # Let's just take the first 50.
    return points[:n_points].copy()


def _sa_optimize(centers, time_budget):
    """Simulated Annealing optimization."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.005
    T_min = 1e-5
    step_size = 0.01
    decay = 0.9995
    
    while time.time() - start < time_budget:
        # Pick a random point
        i = np.random.randint(n)
        
        # Perturb
        delta = np.random.normal(0, step_size, 3)
        new_center = centers[i].copy()
        new_center += delta
        new_center = np.clip(new_center, -0.5, 0.5)
        
        # Calculate new radius
        new_centers = centers.copy()
        new_centers[i] = new_center
        new_r = _radius(new_centers)
        
        # Acceptance criterion
        if new_r >= current_r:
            centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            if np.random.rand() < np.exp((new_r - current_r) / T):
                centers = new_centers
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
        
        T *= decay
        step_size *= decay
        if T < T_min:
            T = T_min
            step_size = max(step_size, 0.001)
            
    return best_centers, best_r

def _local_optimize(centers, time_budget):
    """Local optimization using Nelder-Mead on the full configuration."""
    start = time.time()
    centers = centers.copy()
    
    # Flatten centers for optimization
    x0 = centers.flatten()
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # We want to maximize radius, so minimize -radius
    # Use L-BFGS-B or Nelder-Mead? L-BFGS-B is better for constrained, but Nelder-Mead is simpler.
    # However, Nelder-Mead doesn't handle bounds well. Let's use L-BFGS-B with penalty or just clip.
    # Actually, let's use a custom objective that penalizes going out of bounds.
    
    def objective(x):
        c = x.reshape(N, 3)
        # Penalty for going out of bounds
        penalty = 0.0
        for i in range(N):
            for j in range(3):
                if c[i, j] > 0.5:
                    penalty += (c[i, j] - 0.5) ** 2
                elif c[i, j] < -0.5:
                    penalty += (c[i, j] + 0.5) ** 2
        # Add a large weight to penalty to keep points inside
        return -_radius(c) + 100.0 * penalty
    
    # Use Nelder-Mead as it's robust for this kind of non-smooth objective (min of distances)
    # But it can be slow. Let's try it for a short time.
    result = opt.minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    if time.time() - start < time_budget:
        best_centers = result.x.reshape(N, 3)
        best_r = _radius(best_centers)
        return best_centers, best_r
    else:
        return centers, _radius(centers)


def _coordinate_descent(centers, time_budget):
    """Coordinate Descent polish."""
    start = time.time()
    centers = centers.copy()
    n = len(centers)
    current_r = _radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    while time.time() - start < time_budget:
        improved = False
        for i in range(n):
            for d in range(3):
                # Optimize coordinate d of point i
                val = centers[i, d]
                
                candidates = np.linspace(-0.5, 0.5, 20)
                best_val = val
                best_r_local = current_r
                
                for c_val in candidates:
                    new_centers = centers.copy()
                    new_centers[i, d] = c_val
                    r = _radius(new_centers)
                    if r > best_r_local:
                        best_r_local = r
                        best_val = c_val
                
                if best_val != val:
                    centers[i, d] = best_val
                    current_r = best_r_local
                    improved = True
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        if not improved:
            break
            
    return best_centers, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Initial configuration: BCC lattice
    centers = _bcc_init(N)
    
    best_centers = centers.copy()
    best_r = _radius(best_centers)

    # Phase 1: Simulated Annealing
    if time.time() - start_time < deadline - 20:
        budget = min(60.0, deadline - 20 - (time.time() - start_time))
        sa_centers, sa_r = _sa_optimize(centers, budget)
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

    # Phase 2: Local Optimization (Nelder-Mead)
    if time.time() - start_time < deadline - 10:
        budget = min(30.0, deadline - 10 - (time.time() - start_time))
        lo_centers, lo_r = _local_optimize(best_centers, budget)
        if lo_r > best_r:
            best_r = lo_r
            best_centers = lo_centers.copy()

    # Phase 3: Coordinate Descent
    # Polish the result
    if time.time() - start_time < deadline - 5:
        budget = min(10.0, deadline - 5 - (time.time() - start_time))
        cd_centers, cd_r = _coordinate_descent(best_centers, budget)
        if cd_r > best_r:
            best_r = cd_r
            best_centers = cd_centers.copy()

    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.5):
        # Fallback to grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].copy()

    return best_centers
# EVOLVE-BLOCK-END
