import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

N = 50

def compute_radius(centres):
    """Compute the radius for a given set of centres."""
    if centres.shape != (N, 2):
        return 0.0
    if np.any(~np.isfinite(centres)):
        return 0.0
    # Check boundaries: all coords must be in [-0.5, 0.5]
    if np.any(np.abs(centres) > 0.5 + 1e-9):
        return 0.0
    # Compute min distance between pairs
    dists = pdist(centres, metric='euclidean')
    min_pair_dist = np.min(dists)
    # Compute min distance to boundary
    dist_to_walls = np.array([
        0.5 - np.abs(centres[:, 0]),  # left/right
        0.5 - np.abs(centres[:, 1])   # top/bottom
    ])
    min_wall_dist = np.min(dist_to_walls)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return max(0.0, r)

def _force_directed_anneal(x0, max_iter=5000, initial_temp=1.0, cooling=0.999):
    """
    Force-directed repulsion annealing.
    We minimize the potential energy: E = sum_{i<j} 1/d_ij^2 + sum_i 1/d_iw^2
    This is a smooth, convex-like function (locally) that pushes points apart.
    We use gradient descent with adaptive step size.
    """
    c = x0.copy()
    temp = initial_temp
    
    # Precompute indices for vectorization if possible, but N=50 is small.
    # We will compute forces in a loop for clarity and correctness, 
    # but vectorize the distance calculations where possible.
    
    for it in range(max_iter):
        # Compute distances and forces
        forces = np.zeros_like(c)
        
        # Pairwise forces
        # d_ij = c_i - c_j
        # F_ij = -2 * (c_i - c_j) / |c_i - c_j|^4  (gradient of 1/d^2 w.r.t c_i)
        # Wait, potential V = 1/d^2. grad V wrt c_i = -2/d^4 * (c_i - c_j).
        # We want to move in direction of -grad E.
        # So force on i from j is +2/d^4 * (c_i - c_j).
        
        # Vectorized pairwise calculation
        # Create matrices of coordinates
        cx = c[:, 0].reshape(-1, 1)
        cy = c[:, 1].reshape(-1, 1)
        
        dx = cx - cx.T  # NxN matrix, dx[i,j] = x_i - x_j
        dy = cy - cy.T
        d2 = dx**2 + dy**2
        
        # Avoid division by zero (if points coincide)
        d2[diag_indices := np.diag_indices(N)] = 1.0
        d = np.sqrt(d2)
        
        # Force magnitude factor: 2 / d^4
        # F = factor * (dx, dy)
        # factor[i,j] = 2 / d[i,j]^4
        factor = 2.0 / (d2**2)
        factor[diag_indices] = 0.0
        
        fx = np.sum(factor * dx, axis=1)
        fy = np.sum(factor * dy, axis=1)
        
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        # Wall forces
        # Potential V_wall = sum 1/d_wall^2
        # d_left = 0.5 - x_i
        # d_right = 0.5 + x_i
        # d_bottom = 0.5 - y_i
        # d_top = 0.5 + y_i
        
        d_left = 0.5 - c[:, 0]
        d_right = 0.5 + c[:, 0]
        d_bottom = 0.5 - c[:, 1]
        d_top = 0.5 + c[:, 1]
        
        # Clamp distances to avoid singularity if point is outside
        d_left = np.maximum(d_left, 1e-6)
        d_right = np.maximum(d_right, 1e-6)
        d_bottom = np.maximum(d_bottom, 1e-6)
        d_top = np.maximum(d_top, 1e-6)
        
        # Gradient of 1/d^2 wrt x:
        # d/dx (1/(0.5-x)^2) = -2/(0.5-x)^3 * (-1) = 2/(0.5-x)^3
        # Force = -grad V = -2/(0.5-x)^3 for left wall? 
        # Let's check: V = 1/d^2. dV/dx = -2/d^3 * dd/dx.
        # For left wall, d = 0.5 - x, dd/dx = -1. dV/dx = 2/d^3.
        # Force = -dV/dx = -2/d^3. This pushes x to the right (positive force if x < 0.5). Correct.
        
        f_left = -2.0 / (d_left**3)
        f_right = -2.0 / (d_right**3)  # d = 0.5+x, dd/dx=1, dV/dx=-2/d^3, F=2/d^3? 
        # Wait. V = 1/(0.5+x)^2. dV/dx = -2/(0.5+x)^3. Force = -dV/dx = 2/(0.5+x)^3.
        # My formula above was -2/d^3. Let's re-verify.
        # Force = -grad V.
        # Left: V = (0.5-x)^-2. dV/dx = -2(0.5-x)^-3 * (-1) = 2(0.5-x)^-3. Force = -2(0.5-x)^-3.
        # Right: V = (0.5+x)^-2. dV/dx = -2(0.5+x)^-3 * (1) = -2(0.5+x)^-3. Force = 2(0.5+x)^-3.
        
        f_left = -2.0 / (d_left**3)
        f_right = 2.0 / (d_right**3)
        f_bottom = -2.0 / (d_bottom**3)
        f_top = 2.0 / (d_top**3)
        
        forces[:, 0] += f_left + f_right
        forces[:, 1] += f_bottom + f_top
        
        # Update positions
        # Adaptive step size based on temperature
        # We want to move in direction of forces (which is -grad E)
        # Step size should be small enough to avoid overshooting.
        # Let's use a fixed small step scaled by temp? Or just a fixed step?
        # With repulsion, forces can be huge if points are close.
        # Let's normalize forces or cap them.
        
        # Simple approach: step_size = temp
        # But temp cools. 
        # Better: step_size = 1.0 / (1.0 + np.max(np.abs(forces)))
        # This ensures steps are small when forces are large.
        
        max_force = np.max(np.abs(forces))
        if max_force > 0:
            step = 1.0 / max_force
            # Scale by a factor to control speed
            step *= 0.1 
        else:
            step = 0.01
            
        c += step * forces
        
        # Project back to square
        c = np.clip(c, -0.5, 0.5)
        
        # Cool down
        temp *= cooling
        
    return c

def _polish(centres, maxiter=2000):
    """
    High-precision polish using Nelder-Mead on the exact nonsmooth objective.
    """
    def objective(x):
        c = x.reshape(N, 2)
        r = compute_radius(c)
        return -r
    
    # Use a small initial simplex to avoid large jumps
    x0 = centres.flatten()
    # Nelder-Mead options
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
    
    new_c = res.x.reshape(N, 2)
    np.clip(new_c, -0.5, 0.5, out=new_c)
    return new_c

def _grid_init():
    """Generate a grid initialization."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _random_init():
    """Generate a random initialization within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

def _structured_init(strategy='spiral'):
    """Generate structured initializations."""
    if strategy == 'spiral':
        t = np.linspace(0, 4 * np.pi, N)
        r = 0.5 * t / (4 * np.pi)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y])
    elif strategy == 'fibonacci':
        n = N
        i = np.arange(n)
        golden = (1 + np.sqrt(5)) / 2
        x = (i / golden) % 1.0 - 0.5
        y = i / n - 0.5
        return np.column_stack([x, y])
    else:
        return _random_init()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Set seeds for reproducibility
    np.random.seed(42)
    
    # Phase 1: Force-directed annealing from multiple starts
    best_centres = _grid_init()
    best_r = compute_radius(best_centres)
    
    initializations = [
        _grid_init(),
        _structured_init('spiral'),
        _structured_init('fibonacci'),
        _random_init()
    ]
    
    for i, c0 in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Run force-directed annealing
        # We can run for a fixed number of iterations or until convergence
        # Let's run for 2000 iterations for each start
        c_fd = _force_directed_anneal(c0, max_iter=2000)
        r_fd = compute_radius(c_fd)
        
        if r_fd > best_r:
            best_r = r_fd
            best_centres = c_fd.copy()
            
        # Also try polishing the result immediately to see if it helps
        if time.time() > deadline - 10:
            continue
            
        c_polished = _polish(c_fd, maxiter=500)
        r_polished = compute_radius(c_polished)
        
        if r_polished > best_r:
            best_r = r_polished
            best_centres = c_polished.copy()

    # Phase 2: Final polish on the best result
    if time.time() < deadline - 5:
        c_final = _polish(best_centres, maxiter=1000)
        r_final = compute_radius(c_final)
        
        if r_final > best_r:
            best_r = r_final
            best_centres = c_final

    # Final validation
    best_centres = np.clip(best_centres, -0.5, 0.5)
    if np.any(~np.isfinite(best_centres)):
        best_centres = _grid_init()
    
    return best_centres
