"""Pack 200 equal circles in a unit square [-0.5, 0.5]^2 via smooth energy minimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 200
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _compute_radius(pts):
    """Compute the feasible radius for a set of points."""
    # Distance to boundaries
    boundary_dists = np.min(np.minimum(np.abs(pts[:, 0] + 0.5), np.abs(0.5 - pts[:, 0])),
                            np.minimum(np.abs(pts[:, 1] + 0.5), np.abs(0.5 - pts[:, 1])))
    min_boundary = np.min(boundary_dists)
    
    # Pairwise distances
    n = len(pts)
    if n <= 1:
        return min_boundary
    
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_boundary, min_pair / 2.0)

def _energy(pts, power=12):
    """Smooth energy: sum of inverse powers of distances + boundary penalty."""
    n = len(pts)
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Avoid zero division
    dists_sq = np.maximum(dists_sq, 1e-12)
    
    # Inverse power sum for pairs
    pair_energy = np.sum(dists_sq ** (-power / 2.0))
    
    # Boundary penalty: points should stay inside [-0.5, 0.5]
    # Use soft penalty for going outside
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Distance to each boundary (negative if outside)
    dx = np.minimum(0.5 - np.abs(x), 0.5)  # This is always 0.5, wrong approach
    # Better: penalty for being outside
    boundary_penalty = 0.0
    for coord in [x, y]:
        # Left boundary at -0.5
        outside_left = np.maximum(0, -0.5 - coord)
        boundary_penalty += np.sum(outside_left ** 2)
        # Right boundary at 0.5
        outside_right = np.maximum(0, coord - 0.5)
        boundary_penalty += np.sum(outside_right ** 2)
    
    return pair_energy + 100.0 * boundary_penalty

def _energy_grad(pts, power=12):
    """Gradient of the energy function."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e-12)
    
    # Gradient of sum_{i<j} d_ij^{-power}
    # d/dx_i d_ij^{-power} = -power * d_ij^{-power-2} * (x_i - x_j)
    dists_sq_clipped = np.maximum(dists_sq, 1e-12)
    factor = -power * dists_sq_clipped ** (-(power / 2.0) - 1.0)
    
    # Sum over j for each i
    grad_pairs = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    # Boundary gradient
    x = pts[:, 0]
    y = pts[:, 1]
    grad_boundary = np.zeros_like(pts)
    
    # For x
    outside_left_x = np.maximum(0, -0.5 - x)
    grad_boundary[:, 0] += 200.0 * (-1.0) * (x < -0.5).astype(float) * np.ones(n)
    # Actually, let's do it properly
    grad_boundary[:, 0] = 0
    grad_boundary[:, 1] = 0
    
    for i in range(n):
        if x[i] < -0.5:
            grad_boundary[i, 0] += 200.0 * (-1.0)
        elif x[i] > 0.5:
            grad_boundary[i, 0] += 200.0 * (1.0)
        if y[i] < -0.5:
            grad_boundary[i, 1] += 200.0 * (-1.0)
        elif y[i] > 0.5:
            grad_boundary[i, 1] += 200.0 * (1.0)
    
    return grad_pairs + grad_boundary

def _make_hexagonal_init():
    """Create a hexagonal close-packed initial configuration."""
    # For 200 circles, try to fit in a hexagonal pattern
    # Hexagonal packing: rows offset by half spacing
    # We want to fill a square as much as possible
    
    # Start with a grid and then perturb
    # Use a hexagonal lattice
    pts = []
    
    # Determine number of rows and columns
    # For hexagonal packing, row spacing = s * sqrt(3)/2, col spacing = s
    # where s is the side length
    
    # Try different numbers of rows
    best_pts = None
    best_score = 0
    
    for n_rows in range(13, 18):
        n_cols = int(np.ceil(N / n_rows))
        if n_rows * n_cols < N:
            continue
            
        # Spacing
        s = 1.0 / n_cols  # approximate
        row_spacing = s * np.sqrt(3) / 2
        
        # Center the pattern
        total_width = (n_cols - 1) * s
        total_height = (n_rows - 1) * row_spacing
        
        x_start = -total_width / 2
        y_start = -total_height / 2
        
        pts = []
        count = 0
        for row in range(n_rows):
            for col in range(n_cols):
                if count >= N:
                    break
                x = x_start + col * s + (row % 2) * s / 2
                y = y_start + row * row_spacing
                pts.append([x, y])
                count += 1
            if count >= N:
                break
        
        if len(pts) == N:
            pts = np.array(pts)
            # Clip to bounds
            pts = np.clip(pts, -0.49, 0.49)
            score = _compute_radius(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
    
    if best_pts is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return best_pts

def construct():
    start_time = time.time()
    
    # Initial configuration: hexagonal
    pts = _make_hexagonal_init()
    best_pts = pts.copy()
    best_radius = _compute_radius(pts)
    
    # Use L-BFGS-B to minimize the energy
    # We'll do multiple restarts with different power values
    
    for power in [8, 10, 12, 14, 16]:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Define objective and gradient for this power
        def obj(flat):
            p = flat.reshape(-1, 2)
            return _energy(p, power)
        
        def grad(flat):
            p = flat.reshape(-1, 2)
            return _energy_grad(p, power).reshape(-1)
        
        x0 = pts.reshape(-1)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B', jac=grad,
                options={'maxiter': 2000, 'maxfun': 50000, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < _energy(pts, power):
                new_pts = result.x.reshape(-1, 2)
                new_radius = _compute_radius(new_pts)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
        except Exception:
            pass
    
    # Final refinement: try to push points slightly toward optimal
    # Use a local search on the best configuration
    if time.time() - start_time < DEADLINE - 10:
        # Do a few more iterations with higher power for fine-tuning
        for power in [20, 24, 28]:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            def obj(flat):
                p = flat.reshape(-1, 2)
                return _energy(p, power)
            
            def grad(flat):
                p = flat.reshape(-1, 2)
                return _energy_grad(p, power).reshape(-1)
            
            x0 = best_pts.reshape(-1)
            
            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B', jac=grad,
                    options={'maxiter': 1000, 'maxfun': 30000, 'gtol': 1e-12}
                )
                
                if result.success or result.fun < _energy(best_pts, power):
                    new_pts = result.x.reshape(-1, 2)
                    new_radius = _compute_radius(new_pts)
                    if new_radius > best_radius:
                        best_radius = new_radius
                        best_pts = new_pts.copy()
            except Exception:
                pass
    
    # Ensure points are strictly inside the square
    best_pts = np.clip(best_pts, -0.499999, 0.499999)
    
    return best_pts
