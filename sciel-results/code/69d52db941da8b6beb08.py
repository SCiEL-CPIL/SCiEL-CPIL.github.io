import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_radius(points):
    """
    Computes the radius r for a given set of points.
    r = min(min_{i<j} |c_i - c_j| / 2, min_i min_k (1/2 - |c_ik|))
    Returns 0.0 if invalid.
    """
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check bounds
    if np.any(points < -0.5) or np.any(points > 0.5):
        return 0.0

    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Compute distance to walls
    wall_dists = np.min(np.stack([0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1])], axis=1), axis=1)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min(values, beta):
    """
    Smooth approximation of min(values).
    """
    v_min = np.min(values)
    shifted = values - v_min
    exp_sum = np.sum(np.exp(-beta * shifted))
    return v_min - (1.0 / beta) * np.log(exp_sum)

def _objective_soft_min(points, beta):
    """
    Objective function for optimization: -soft_min(pair_dists/2, wall_dists)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dists = np.sqrt(dx**2 + dy**2)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    wall_x = 0.5 - np.abs(x)
    wall_y = 0.5 - np.abs(y)
    wall_dists = np.minimum(wall_x, wall_y)
    
    all_vals = np.concatenate([pair_dists, wall_dists])
    
    r_soft = _soft_min(all_vals, beta)
    return -r_soft

def _initial_hexagonal():
    """
    Generate an initial hexagonal packing configuration.
    """
    points = []
    # Hexagonal lattice
    # a = 1.0 (spacing)
    # rows of 7, 6, 7, 6, 7, 6, 5? 
    # Let's just generate a grid and then adjust.
    # Or generate a hex grid that fits in the square.
    
    # Approximate hex grid
    # Spacing s = 1.0
    # dx = 1.0
    # dy = sqrt(3)/2 * 1.0
    
    # Let's try a standard hex packing that fits 50 points.
    # 7 columns: 7, 6, 7, 6, 7, 6, 5 -> 44 points. Not enough.
    # 8 columns?
    
    # Simpler: Create a square grid of 7x7=49 and add 1 in center, then let MD sort it.
    # But a hex grid is better.
    
    # Let's create a hex grid with spacing 1.0
    # Center at 0,0
    # Points:
    # (0,0)
    # (1,0), (-1,0)
    # (0.5, sqrt(3)/2), (0.5, -sqrt(3)/2), (-0.5, sqrt(3)/2), (-0.5, -sqrt(3)/2)
    # etc.
    
    pts = []
    s = 1.0
    # Generate points in a hex lattice
    # Max radius to fit in square [-0.5, 0.5] is approx 0.5
    # So we can only fit a small cluster if s=1.0.
    # We need to scale down or just use a grid.
    
    # Let's use a jittered grid of 7x7 + 1 center.
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    p50 = np.array([0.0, 0.0])
    points = np.vstack([grid, p50])
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, -0.5, 0.5)
    
    return points

def _md_simulate(points, time_budget):
    """
    Run Molecular Dynamics simulation.
    """
    start_time = time.time()
    
    # State
    pos = points.copy()
    vel = np.zeros_like(pos)
    
    # Parameters
    dt = 0.005
    damping = 0.99
    repulsion_k = 100.0
    attraction_k = 1.0
    
    # Precompute indices for pairwise forces
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    best_pos = pos.copy()
    best_score = compute_radius(best_pos)
    
    # Adaptive dt and damping
    # We want to cool down the system
    
    while time.time() - start_time < time_budget:
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Pairwise repulsion
        diff = pos[i_idx] - pos[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        dists = np.maximum(dists, 1e-6)
        
        # Force magnitude: k / d^2
        # Direction: normalized diff
        # F = k * diff / d^3
        inv_d3 = repulsion_k / (dists**3)
        
        f_vecs = diff * inv_d3[:, np.newaxis]
        
        # Apply forces to i and j
        np.add.at(forces, i_idx, f_vecs)
        np.add.at(forces, j_idx, -f_vecs)
        
        # Attraction to center (weak) to keep them in the square
        # F_attr = -k_attr * pos
        forces -= attraction_k * pos
        
        # Wall forces (soft repulsion from walls)
        # If pos > 0.5 - margin, push back
        # margin = 0.01
        margin = 0.05
        # Left wall: x < -0.5 + margin
        # Force +x
        left_mask = pos[:, 0] < -0.5 + margin
        forces[left_mask, 0] += repulsion_k * (margin - (pos[left_mask, 0] + 0.5)) / margin
        # Right wall: x > 0.5 - margin
        right_mask = pos[:, 0] > 0.5 - margin
        forces[right_mask, 0] -= repulsion_k * (pos[right_mask, 0] - (0.5 - margin)) / margin
        # Bottom wall: y < -0.5 + margin
        bottom_mask = pos[:, 1] < -0.5 + margin
        forces[bottom_mask, 1] += repulsion_k * (margin - (pos[bottom_mask, 1] + 0.5)) / margin
        # Top wall: y > 0.5 - margin
        top_mask = pos[:, 1] > 0.5 - margin
        forces[top_mask, 1] -= repulsion_k * (pos[top_mask, 1] - (0.5 - margin)) / margin
        
        # Velocity Verlet Update
        # v_half = v + 0.5 * dt * F
        vel_half = vel + 0.5 * dt * forces
        
        # x_new = x + dt * v_half
        pos_new = pos + dt * vel_half
        
        # Compute new forces at x_new
        forces_new = np.zeros_like(pos_new)
        
        diff_new = pos_new[i_idx] - pos_new[j_idx]
        dists_new = np.linalg.norm(diff_new, axis=1)
        dists_new = np.maximum(dists_new, 1e-6)
        inv_d3_new = repulsion_k / (dists_new**3)
        f_vecs_new = diff_new * inv_d3_new[:, np.newaxis]
        np.add.at(forces_new, i_idx, f_vecs_new)
        np.add.at(forces_new, j_idx, -f_vecs_new)
        
        forces_new -= attraction_k * pos_new
        
        # Wall forces at new pos
        left_mask_n = pos_new[:, 0] < -0.5 + margin
        forces_new[left_mask_n, 0] += repulsion_k * (margin - (pos_new[left_mask_n, 0] + 0.5)) / margin
        right_mask_n = pos_new[:, 0] > 0.5 - margin
        forces_new[right_mask_n, 0] -= repulsion_k * (pos_new[right_mask_n, 0] - (0.5 - margin)) / margin
        bottom_mask_n = pos_new[:, 1] < -0.5 + margin
        forces_new[bottom_mask_n, 1] += repulsion_k * (margin - (pos_new[bottom_mask_n, 1] + 0.5)) / margin
        top_mask_n = pos_new[:, 1] > 0.5 - margin
        forces_new[top_mask_n, 1] -= repulsion_k * (pos_new[top_mask_n, 1] - (0.5 - margin)) / margin
        
        # v_new = v_half + 0.5 * dt * F_new
        vel_new = vel_half + 0.5 * dt * forces_new
        
        # Damping
        vel_new *= damping
        
        # Update state
        pos = pos_new
        vel = vel_new
        
        # Check score periodically
        if np.random.random() < 0.01: # Every ~100 steps
            curr_score = compute_radius(pos)
            if curr_score > best_score:
                best_score = curr_score
                best_pos = pos.copy()
        
        # Cool down damping slightly? Or keep it constant?
        # Damping helps settle.
        # If we want to explore, we might want to reduce damping (more chaotic) initially?
        # No, MD with repulsion naturally explores. Damping prevents divergence.
        
        # Adjust dt if energy is high?
        # For now, fixed dt.
        
    return best_pos, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Initialization
    points = _initial_hexagonal()
    
    best_points = points.copy()
    best_score = compute_radius(best_points)
    
    # 2. MD Simulation
    # Run MD for a significant portion of the time
    md_budget = 100.0
    md_start = time.time()
    
    # Multiple restarts?
    # Let's do one long MD run first.
    # If time permits, do more.
    
    # We can run MD in chunks to allow for checking time and restarting if stuck.
    # But a single long run is often good.
    
    # Let's try running MD for 50s, then check, then maybe restart if improved.
    
    # Actually, let's just run MD until deadline - 20s (leaving time for polish)
    
    # To allow for "restarts" if stuck, we can split MD into phases.
    # Phase 1: 40s
    # Phase 2: 40s (if improved)
    # Phase 3: Polish
    
    # Let's just run one big MD loop but check time internally.
    
    md_points, md_score = _md_simulate(points, md_budget)
    
    if md_score > best_score:
        best_score = md_score
        best_points = md_points
    
    # 3. Polish
    polish_start = time.time()
    remaining_time = deadline - (time.time() - start_time)
    
    if remaining_time > 5.0:
        # Use L-BFGS-B to polish the MD result
        def obj_func(pts):
            return _objective_soft_min(pts, beta=50.0)
        
        bounds = [(-0.5, 0.5)] * (N * 2)
        x0 = best_points.flatten()
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
            
            if res.fun < obj_func(x0):
                new_pts = res.x.reshape(N, 2)
                new_pts = np.clip(new_pts, -0.5, 0.5)
                new_score = compute_radius(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts
        except Exception:
            pass
        
        # Second polish with higher beta
        if time.time() - start_time < deadline - 2.0:
            def obj_func_high_beta(pts):
                return _objective_soft_min(pts, beta=200.0)
            
            try:
                res = minimize(obj_func_high_beta, best_points.flatten(), method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10})
                
                if res.fun < obj_func_high_beta(best_points.flatten()):
                    new_pts = res.x.reshape(N, 2)
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    new_score = compute_radius(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts
            except Exception:
                pass

    return best_points

if __name__ == "__main__":
    # Quick test
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time: {end - start:.2f}s")
    print(f"Radius: {compute_radius(pts)}")
