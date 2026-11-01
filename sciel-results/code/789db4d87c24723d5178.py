# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n = 100
    # 1. Initialization: Jittered grid
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    x = np.linspace(-0.4, 0.4, nx)
    y = np.linspace(-0.4, 0.4, ny)
    z = np.linspace(-0.4, 0.4, nz)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))
    
    np.random.seed(42)
    pts += np.random.uniform(-0.02, 0.02, size=(n, 3))
    pts = np.clip(pts, -0.45, 0.45)
    
    # 2. Molecular Dynamics with Simulated Annealing
    # Goal: Minimize overlap energy to spread points
    num_steps = 10000
    dt = 0.005
    initial_temp = 0.05
    final_temp = 0.001
    cooling_rate = (final_temp / initial_temp) ** (1.0 / num_steps)
    
    d_target = 0.22 # Approx 2 * 0.11
    k_rep = 50.0
    k_bound = 100.0
    
    vel = np.zeros_like(pts)
    
    for step in range(num_steps):
        temp = initial_temp * (cooling_rate ** step)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    diff = np.random.normal(0, 0.01, size=3)
                    dist = np.linalg.norm(diff)
                
                if dist < d_target:
                    # Force proportional to overlap
                    f_mag = k_rep * (d_target - dist)
                    f_vec = f_mag * (diff / dist)
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        # Boundary forces
        # Push back if too close to boundary
        # Estimate current radius for boundary limit
        # To save time, use a fixed safe boundary for MD phase
        bound_limit = 0.42 # 0.5 - 0.08
        
        for i in range(n):
            for k in range(3):
                if pts[i, k] > bound_limit:
                    forces[i, k] -= k_bound * (pts[i, k] - bound_limit)
                elif pts[i, k] < -bound_limit:
                    forces[i, k] -= k_bound * (pts[i, k] + bound_limit)
        
        # Update
        vel = vel * 0.9 + forces * dt
        pts += vel * dt
        pts += np.random.normal(0, temp, size=(n, 3))
        pts = np.clip(pts, -0.499, 0.499)
    
    # 3. Subgradient Ascent for Max Min Distance
    # We want to maximize r = min( d_ij/2, boundary_dist )
    # Let f(p) = min( ... )
    # We perform gradient ascent on f.
    
    def compute_min_dist_and_indices(p):
        min_val = np.inf
        active_indices = [] # List of (i, j, type) or (i, k, type)
        
        # Pairwise
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(p[i] - p[j])
                val = d / 2.0
                if val < min_val:
                    min_val = val
                    active_indices = [(i, j, 'pair')]
                elif abs(val - min_val) < 1e-6:
                    active_indices.append((i, j, 'pair'))
        
        # Boundary
        for i in range(n):
            for k in range(3):
                b = 0.5 - np.abs(p[i, k])
                if b < min_val:
                    min_val = b
                    active_indices = [(i, k, 'bound')]
                elif abs(b - min_val) < 1e-6:
                    active_indices.append((i, k, 'bound'))
        
        return min_val, active_indices
    
    current_pts = pts.copy()
    current_r, _ = compute_min_dist_and_indices(current_pts)
    best_pts = current_pts.copy()
    best_r = current_r
    
    # Ascent steps
    step_size = 0.005
    for step in range(5000):
        r, active = compute_min_dist_and_indices(current_pts)
        
        if r > best_r:
            best_r = r
            best_pts = current_pts.copy()
        
        # Compute subgradient
        grad = np.zeros_like(current_pts)
        
        if not active:
            break
            
        # Average gradient of active constraints
        # For pair (i,j): dist/2. grad_i = (p_i - p_j) / (2 * dist) * 2? 
        # Let d = ||p_i - p_j||. f = d/2.
        # df/dp_i = (p_i - p_j) / d.
        # df/dp_j = (p_j - p_i) / d.
        # For bound (i,k): f = 0.5 - |x_ik|.
        # df/dx_ik = -sign(x_ik).
        
        num_active = len(active)
        
        for idx in active:
            if idx[2] == 'pair':
                i, j, _ = idx
                diff = current_pts[i] - current_pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    continue
                g = diff / dist
                grad[i] += g
                grad[j] -= g
            else: # bound
                i, k, _ = idx
                val = current_pts[i, k]
                if val > 0:
                    grad[i, k] -= 1.0
                else:
                    grad[i, k] += 1.0
        
        grad /= num_active
        
        # Update
        current_pts += step_size * grad
        current_pts = np.clip(current_pts, -0.4999, 0.4999)
        
        # Reduce step size
        if step % 100 == 0:
            step_size *= 0.95
        
        if step_size < 1e-6:
            break
            
    # Final check and return
    return best_pts
# EVOLVE-BLOCK-END
