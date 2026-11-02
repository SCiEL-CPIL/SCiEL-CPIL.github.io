import numpy as np
from scipy.spatial.distance import cdist

N = 50

def construct():
    def compute_forces(positions):
        c = positions
        forces = np.zeros_like(c)
        
        # Sphere-Sphere repulsion
        # Use Coulomb-like repulsion: F ~ 1/r^2, vector ~ r/r^3
        # Add softening to avoid singularity
        D = cdist(c, c)
        np.fill_diagonal(D, 1.0) # Avoid div by 0, will be zeroed later
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        
        # Force magnitude proportional to 1/D^2
        # Vector = diff / D^3
        # To soften, use (D + eps)
        D_safe = D + 0.01
        inv_d3 = 1.0 / (D_safe ** 3)
        inv_d3[np.diag_indices(N)] = 0.0
        
        F_ss = np.sum(diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        # Push spheres towards the center to keep them away from walls.
        # Force is proportional to displacement from center.
        # F_wall = -k * c
        # We want this to be strong enough to prevent wall violations but not dominate.
        F_wall = -0.1 * c
        
        return F_ss + F_wall

    def objective_r(c):
        D = cdist(c, c)
        np.fill_diagonal(D, np.inf)
        min_pair = np.min(D) / 2.0
        min_wall = np.min(0.5 - np.max(np.abs(c), axis=1))
        return min(min_pair, min_wall)

    best_r = 0.0
    best_c = None

    # Initialization: 5x5x2 grid
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.array([-0.2, 0.2])
    grid = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid.append([x, y, z])
    c_init = np.array(grid)
    
    # Multiple restarts with different noise levels
    for restart in range(10):
        c = c_init.copy()
        # Add noise to break symmetry
        noise_level = 0.001 + 0.001 * restart
        c += np.random.normal(0, noise_level, c.shape)
        c = np.clip(c, -0.49, 0.49)
        
        step_size = 0.005
        best_r_restart = objective_r(c)
        best_c_restart = c.copy()
        
        for it in range(3000):
            F = compute_forces(c)
            
            # Limit force magnitude to prevent explosion
            max_force = 1.0
            norm_F = np.linalg.norm(F, axis=1, keepdims=True)
            scale = np.where(norm_F > max_force, max_force / norm_F, 1.0)
            F = F * scale
            
            c_new = c + F * step_size
            c_new = np.clip(c_new, -0.49, 0.49)
            
            r_new = objective_r(c_new)
            
            if r_new > best_r_restart:
                best_r_restart = r_new
                best_c_restart = c_new.copy()
                c = c_new
            else:
                # If no improvement, reduce step size
                step_size *= 0.95
                # If step size is too small, break
                if step_size < 1e-5:
                    break
            
            # Slowly cool down step size
            step_size *= 0.999
            
        if best_r_restart > best_r:
            best_r = best_r_restart
            best_c = best_c_restart

    return best_c
