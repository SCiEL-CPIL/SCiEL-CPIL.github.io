import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin for 120s limit

def _log_energy(pts):
    """Compute log energy for points on the unit sphere."""
    pts = pts.reshape(-1, 3)
    # Normalize to ensure they are on the sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    
    iu = np.triu_indices(N, 1)
    d = dist[iu]
    # Avoid log(0) just in case, though FPS and MD should prevent this
    d = np.maximum(d, 1e-12)
    energy = np.sum(np.log(1.0 / d))
    return energy

def _force(pts):
    """Compute the repulsive force on each point.
    F_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    This is the negative gradient of the log energy projected to the tangent space?
    Gradient of log(1/d) = -1/d * dd/dx.
    d/dx_i log(1/|x_i - x_j|) = -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j| = -(x_i - x_j)/|x_i - x_j|^2.
    So Force = -Gradient.
    """
    pts = pts.reshape(-1, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Force term: (x_i - x_j) / |x_i - x_j|^2
    # inv_dist_sq = 1 / dist_sq
    # forces = diff * inv_dist_sq[:, :, None]
    # grad = -sum(forces, axis=1)
    # Force = -grad = sum(forces, axis=1)
    
    inv_dist_sq = np.ones_like(dist_sq)
    mask = dist_sq > 1e-12
    inv_dist_sq[mask] = 1.0 / (dist_sq[mask])
    
    forces = diff * inv_dist_sq[:, :, None]
    f = np.sum(forces, axis=1)
    
    # Project force to tangent space to keep points on sphere
    # f_tangent = f - (f . x) x
    dot = np.sum(f * pts, axis=1, keepdims=True)
    f_tangent = f - dot * pts
    
    return f_tangent

def _fps_init():
    """Farthest Point Sampling initialization."""
    # Start with a random point
    pts = np.zeros((N, 3))
    pts[0] = np.random.randn(3)
    pts[0] /= np.linalg.norm(pts[0])
    
    # Distances to closest selected point
    d_min = np.full(N, np.inf)
    
    for i in range(1, N):
        # Compute distances from all unselected points to the last selected point
        # Actually, we need distance to the SET of selected points.
        # Efficient way: update d_min for all points
        # d_min[j] = min(d_min[j], ||x_j - x_last||)
        diff = pts[:i] # Not quite, we need to compare against all selected
        # Let's just compute pairwise distances from the new point to all others
        # and update the minimum distance for each unselected point.
        
        # This is O(N^2) per step, total O(N^3). N=150, so 150^3 = 3.3M ops. Fast enough.
        # But we can do better.
        # Let's just compute distance from the newly selected point to all unselected points.
        # And update the global min distance array.
        
        new_pt = pts[i-1] # The one just selected in previous iteration? No, we select i now.
        # Wait, the loop logic:
        # i=1: select pt 1.
        # We need to select the point farthest from the current set.
        
        # Let's restart the logic clearly.
        selected = [0]
        d_min = np.full(N, np.inf)
        
        for i in range(1, N):
            # Compute distance from all unselected points to the LAST selected point
            last_pt = pts[selected[-1]]
            diffs = pts - last_pt
            dists = np.linalg.norm(diffs, axis=1)
            # Update min distance for all points
            d_min = np.minimum(d_min, dists)
            
            # Select the point with max min-distance among unselected
            # Mask out selected
            mask = np.ones(N, dtype=bool)
            mask[selected] = False
            d_min_masked = d_min.copy()
            d_min_masked[selected] = -1
            next_idx = np.argmax(d_min_masked)
            
            pts[next_idx] = last_pt # Wait, pts is the container. 
            # Ah, I need a pool of candidate points.
            # Let's generate a large pool of random points first, then pick from it.
            
    # Better FPS implementation:
    # Generate a pool of 1000 random points.
    pool_size = 2000
    pool = np.random.randn(pool_size, 3)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    
    pts = np.zeros((N, 3))
    # Pick first random
    idx0 = np.random.randint(pool_size)
    pts[0] = pool[idx0]
    
    d_min = np.linalg.norm(pool - pts[0], axis=1)
    
    for i in range(1, N):
        # Pick the pool point with max min-distance
        idx_next = np.argmax(d_min)
        pts[i] = pool[idx_next]
        
        # Update d_min
        new_dists = np.linalg.norm(pool - pts[i], axis=1)
        d_min = np.minimum(d_min, new_dists)
        
    return pts

def _md_optimize(pts, time_budget):
    """Molecular Dynamics optimization with Velocity Verlet."""
    pts = pts.copy()
    velocities = np.zeros_like(pts)
    
    # Initial force
    f = _force(pts)
    
    dt = 0.01
    damping = 0.99 # Viscous damping
    max_iter = 5000
    start_time = time.time()
    
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    for step in range(max_iter):
        if time.time() - start_time > time_budget:
            break
            
        # Velocity Verlet
        # v_{n+1/2} = (v_n + 0.5 * dt * F_n) * damping
        v_half = (velocities + 0.5 * dt * f) * damping
        
        # x_{n+1} = x_n + dt * v_{n+1/2}
        pts_new = pts + dt * v_half
        
        # Project to sphere
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        # F_{n+1}
        f_new = _force(pts_new)
        
        # v_{n+1} = (v_{n+1/2} + 0.5 * dt * F_{n+1}) * damping
        velocities = (v_half + 0.5 * dt * f_new) * damping
        
        pts = pts_new
        f = f_new
        
        # Check energy
        if step % 10 == 0:
            e = _log_energy(pts)
            if e < best_energy:
                best_energy = e
                best_pts = pts.copy()
                
        # Adapt dt if energy increases too much (instability)
        # Simple heuristic: if energy jumps, reduce dt
        # For now, keep fixed dt with damping.
        
    return best_pts, best_energy

def _newton_polish(pts, time_budget):
    """Newton-CG polish using scipy minimize on the manifold."""
    pts = pts.copy()
    start_time = time.time()
    
    def objective_and_grad(x):
        p = x.reshape(-N, 3)
        # Normalize
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        
        # Energy
        diff = p[:, None, :] - p[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        iu = np.triu_indices(N, 1)
        d = dist[iu]
        d = np.maximum(d, 1e-12)
        energy = np.sum(np.log(1.0 / d))
        
        # Gradient
        inv_dist_sq = np.ones_like(dist)
        mask = dist > 1e-12
        inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
        forces = diff * inv_dist_sq[:, :, None]
        grad = -np.sum(forces, axis=1)
        
        # Project to tangent
        dot = np.sum(grad * p, axis=1, keepdims=True)
        grad_tangent = grad - dot * p
        
        return energy, grad_tangent.flatten()
    
    x0 = pts.flatten()
    
    # Run L-BFGS-B with tight tolerances
    try:
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
        pts = result.x.reshape(-N, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    except Exception:
        pass
        
    return pts

def construct():
    start_time = time.time()
    
    # 1. Initialize with FPS
    pts = _fps_init()
    
    # 2. MD Optimization
    # Give MD about 60 seconds
    md_time_budget = 60.0
    pts, energy = _md_optimize(pts, md_time_budget)
    
    # 3. Newton Polish
    # Give Newton about 40 seconds
    newton_time_budget = 40.0
    pts = _newton_polish(pts, newton_time_budget)
    
    # Final check
    final_energy = _log_energy(pts)
    
    # If time permits, do a second round of MD from the polished state with smaller dt?
    # Or just return.
    
    return pts

# Test locally if needed
if __name__ == "__main__":
    start = time.time()
    pts = construct()
    end = time.time()
    e = _log_energy(pts)
    print(f"Time: {end-start:.2f}s, Energy: {e}")
