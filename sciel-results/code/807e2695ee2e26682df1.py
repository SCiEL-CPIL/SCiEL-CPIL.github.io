"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 32

# EVOLVE-BLOCK-START

def _pack_powell(x0, max_iter=200, tol=1e-8):
    """Run Powell on a single packing configuration."""
    n = len(x0)
    
    def objective(z):
        # z = [x0,y0,r0, x1,y1,r1, ...]
        score = 0.0
        for i in range(n):
            xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
            # boundary penalty
            if xi < ri:
                score += 1000.0 * (ri - xi)**2
            if xi > 1.0 - ri:
                score += 1000.0 * (xi - (1.0 - ri))**2
            if yi < ri:
                score += 1000.0 * (ri - yi)**2
            if yi > 1.0 - ri:
                score += 1000.0 * (yi - (1.0 - ri))**2
            # overlap penalty
            for j in range(i+1, n):
                xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                min_dist = ri + rj
                if dist < min_dist:
                    score += 100.0 * (min_dist - dist)**2
        return score
    
    res = minimize(objective, x0, method='Powell', 
                   options={'maxiter': max_iter, 'xtol': tol, 'ftol': tol})
    return res.x

def _validate(z, n):
    """Check if configuration is valid (no overlaps, inside square). Return sum of radii."""
    total = 0.0
    for i in range(n):
        xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
        if ri <= 0:
            return -1
        if xi < ri - 1e-9 or xi > 1.0 - ri + 1e-9 or yi < ri - 1e-9 or yi > 1.0 - ri + 1e-9:
            return -1
        total += ri
        for j in range(i+1, n):
            xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-9:
                return -1
    return total

def _grid_init(side):
    """Create a grid-based initial configuration."""
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return cells[:N]

def _random_init(seed):
    """Create a random initial configuration."""
    rng = np.random.RandomState(seed)
    cells = []
    for _ in range(N):
        r = rng.uniform(0.02, 0.12)
        x = rng.uniform(r, 1.0 - r)
        y = rng.uniform(r, 1.0 - r)
        cells.append([x, y, r])
    return cells

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    best_score = 0.0
    best_z = None
    
    # Try multiple initializations
    inits = []
    
    # Grid inits with different densities
    for side in [4, 5, 6]:
        inits.append(_grid_init(side))
    
    # Random inits
    for seed in [42, 123, 789, 321, 555, 999, 111, 222, 333, 444]:
        inits.append(_random_init(seed))
    
    # Jittered grid inits
    for side in [4, 5]:
        for jitter_seed in [1, 2, 3]:
            rng = np.random.RandomState(jitter_seed)
            cells = []
            for i in range(side):
                for j in range(side):
                    x = (i + 0.5) / side + rng.uniform(-0.1/side, 0.1/side)
                    y = (j + 0.5) / side + rng.uniform(-0.1/side, 0.1/side)
                    r = 0.5 / side - 1e-3
                    x = np.clip(x, r, 1.0 - r)
                    y = np.clip(y, r, 1.0 - r)
                    cells.append([x, y, r])
            inits.append(cells[:N])
    
    for init_cells in inits:
        if time.time() > deadline:
            break
        
        z0 = np.array(init_cells).flatten()
        z = _pack_powell(z0, max_iter=150, tol=1e-9)
        
        score = _validate(z, N)
        if score > best_score:
            best_score = score
            best_z = z.copy()
    
    # Local perturbation search on the best solution
    if best_z is not None and time.time() < deadline:
        rng = np.random.RandomState(42)
        current_z = best_z.copy()
        current_score = best_score
        
        for step in range(50):
            if time.time() > deadline:
                break
            
            # Random perturbation
            perturb_scale = 0.01 / (1.0 + step * 0.1)
            z_pert = current_z + rng.randn(len(current_z)) * perturb_scale
            
            # Clamp to valid range
            for i in range(N):
                r_idx = 3*i + 2
                x_idx = 3*i
                y_idx = 3*i + 1
                r = max(0.01, z_pert[r_idx])
                z_pert[r_idx] = r
                z_pert[x_idx] = np.clip(z_pert[x_idx], r, 1.0 - r)
                z_pert[y_idx] = np.clip(z_pert[y_idx], r, 1.0 - r)
            
            z_new = _pack_powell(z_pert, max_iter=100, tol=1e-9)
            score_new = _validate(z_new, N)
            
            if score_new > current_score:
                current_score = score_new
                current_z = z_new.copy()
                if score_new > best_score:
                    best_score = score_new
                    best_z = z_new.copy()
    
    # Final polish with tighter tolerance
    if best_z is not None and time.time() < deadline:
        z_final = _pack_powell(best_z, max_iter=300, tol=1e-10)
        score_final = _validate(z_final, N)
        if score_final > best_score:
            best_score = score_final
            best_z = z_final.copy()
    
    # Convert to output format
    if best_z is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    result = []
    for i in range(N):
        x, y, r = best_z[3*i], best_z[3*i+1], best_z[3*i+2]
        # Ensure validity
        r = max(0.001, r)
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        result.append([x, y, r])
    
    return np.array(result)

# EVOLVE-BLOCK-END
