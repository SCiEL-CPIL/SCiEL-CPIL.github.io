import numpy as np
import time

N = 50
DEADLINE = 120.0

def _radius(c):
    """Compute the common radius for centres c (N,3) in [-0.5,0.5]^3."""
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    wall = 0.5 - np.abs(c)
    min_wall = np.min(wall)
    
    return min(min_pair, min_wall)

def _build_initializations():
    """Generate diverse initial configurations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. 5x5x2 Grid
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.4, 0.4, 2)
    grid1 = np.array([[x[i], y[j], z[k]] for k in range(2) for j in range(5) for i in range(5)])
    inits.append(grid1)
    
    # 2. 10x5x1 Grid
    x = np.linspace(-0.45, 0.45, 10)
    y = np.linspace(-0.45, 0.45, 5)
    grid2 = np.array([[x[i], y[j], 0.0] for j in range(5) for i in range(10)])
    inits.append(grid2)
    
    # 3. 4x4x4 Grid (64 points, remove 14)
    x = np.linspace(-0.35, 0.35, 4)
    y = np.linspace(-0.35, 0.35, 4)
    z = np.linspace(-0.35, 0.35, 4)
    grid3_full = np.array([[x[i], y[j], z[k]] for k in range(4) for j in range(4) for i in range(4)])
    idx = rng.choice(64, 50, replace=False)
    inits.append(grid3_full[idx])
    
    # 4. BCC-like
    x = np.linspace(-0.3, 0.3, 3)
    y = np.linspace(-0.3, 0.3, 3)
    z = np.linspace(-0.3, 0.3, 3)
    grid4 = np.array([[x[i], y[j], z[k]] for k in range(3) for j in range(3) for i in range(3)])
    extra = rng.uniform(-0.45, 0.45, (23, 3))
    inits.append(np.vstack([grid4, extra]))
    
    # 5. Random
    inits.append(rng.uniform(-0.48, 0.48, (50, 3)))
    
    return inits

def _optimize(c, deadline):
    """
    Optimize configuration c using a fast repulsive force model.
    """
    c = c.copy().astype(float)
    c = np.clip(c, -0.499, 0.499)
    best_c = c.copy()
    best_r = _radius(c)
    
    n_iter = 20000
    start_time = time.time()
    rng = np.random.default_rng(123)
    
    lr = 0.01
    stuck_count = 0
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute forces
        diff = c[:, None, :] - c[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Repulsion force: push apart
        # F = diff / dist^2
        inv_d2 = 1.0 / np.maximum(dists, 1e-6) ** 2
        F_pair = np.sum(diff * inv_d2[:, :, None], axis=1)
        
        # Wall force: push away from walls (towards center)
        wall_dist = 0.5 - np.abs(c)
        # Force is stronger if closer to wall
        F_wall = -np.sign(c) * (1.0 / np.maximum(wall_dist, 0.1))
        
        # Combine
        F = F_pair + 0.5 * F_wall
        
        # Normalize force to prevent huge steps
        F_norm = np.linalg.norm(F, axis=1, keepdims=True)
        F = F / np.maximum(F_norm, 1e-6)
        
        c_new = c + lr * F
        c_new = np.clip(c_new, -0.499, 0.499)
        
        r_new = _radius(c_new)
        
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            c = c_new
            stuck_count = 0
            # Slightly increase lr if improving
            lr = min(lr * 1.005, 0.05)
        else:
            c = c_new
            stuck_count += 1
            # Reduce lr
            lr *= 0.99
            # If stuck, try a random kick
            if stuck_count > 100:
                c = best_c + rng.normal(0, 0.05, (N, 3))
                c = np.clip(c, -0.499, 0.499)
                lr = 0.02
                stuck_count = 0
                best_r = _radius(c)
                best_c = c.copy()
                
    return best_c, best_r

def construct():
    start = time.time()
    deadline = start + DEADLINE - 10.0
    
    inits = _build_initializations()
    
    best_c = None
    best_r = -1.0
    
    for init in inits:
        if time.time() > deadline:
            break
        
        c_opt, r_opt = _optimize(init, deadline)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
    # Final polish
    if best_c is not None and time.time() < deadline:
        c_final, r_final = _optimize(best_c, deadline)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final
            
    return best_c
