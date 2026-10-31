import numpy as np
from scipy.optimize import minimize
import time

N = 16
D = 2

def pairwise_distances_sq(pts):
    """Compute all pairwise squared distances (faster, no sqrt)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return dist_sq[iu]

def objective_exact(x):
    """(max_dist / min_dist)^2"""
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    if np.any(d_sq < 1e-24):
        return 1e10
    d_min = np.sqrt(d_sq.min())
    d_max = np.sqrt(d_sq.max())
    ratio = d_max / d_min
    return ratio**2

def objective_smooth(x, beta=100.0):
    """
    Smooth approximation using log-sum-exp on squared distances.
    We optimize the ratio of soft_max(d_sq) / soft_min(d_sq).
    Since the score is (max_d / min_d)^2, and d = sqrt(d_sq),
    (max_d / min_d)^2 = max(d_sq) / min(d_sq).
    So we can work directly with squared distances for the ratio,
    which avoids sqrt in the objective calculation, though log-sum-exp 
    is applied to d_sq.
    """
    pts = x.reshape(N, D)
    d_sq = pairwise_distances_sq(pts)
    d_sq = np.maximum(d_sq, 1e-12) # Avoid log(0) or issues
    
    # Soft max of d_sq
    max_d_sq = np.max(d_sq)
    soft_max_d_sq = max_d_sq + (1.0 / beta) * np.log(np.sum(np.exp(beta * (d_sq - max_d_sq))))
    
    # Soft min of d_sq
    min_d_sq = np.min(d_sq)
    soft_min_d_sq = min_d_sq - (1.0 / beta) * np.log(np.sum(np.exp(-beta * (d_sq - min_d_sq))))
    
    if soft_min_d_sq < 1e-8:
        return 1e10
        
    ratio_sq = soft_max_d_sq / soft_min_d_sq
    return ratio_sq

def make_ring_topology(n_inner, n_outer, r_inner, r_outer, angle_offset=0.0):
    """Create ring topology: center + inner ring + outer ring."""
    pts = [np.array([0.0, 0.0])]
    for i in range(n_inner):
        angle = 2 * np.pi * i / n_inner + angle_offset
        pts.append(np.array([r_inner * np.cos(angle), r_inner * np.sin(angle)]))
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer + angle_offset + np.pi / n_outer
        pts.append(np.array([r_outer * np.cos(angle), r_outer * np.sin(angle)]))
    return np.array(pts)

def make_hexagonal_cluster(params):
    """
    Generate 16 points as a hexagonal lattice cluster.
    params: [aspect, shear, outer_scale]
    """
    aspect = params[0]
    shear = params[1]
    outer_scale = params[2]
    
    base_points = []
    # Row 0 (top): 4 points
    for i in range(4):
        x = (i - 1.5) * 1.0
        y = 2.0
        base_points.append([x, y])
    # Row 1: 3 points, offset by 0.5
    for i in range(3):
        x = (i - 1.0) * 1.0 + 0.5
        y = 2.0 - np.sqrt(3)/2
        base_points.append([x, y])
    # Row 2: 4 points
    for i in range(4):
        x = (i - 1.5) * 1.0
        y = 2.0 - np.sqrt(3)
        base_points.append([x, y])
    # Row 3: 3 points, offset by 0.5
    for i in range(3):
        x = (i - 1.0) * 1.0 + 0.5
        y = 2.0 - 1.5*np.sqrt(3)
        base_points.append([x, y])
    # Row 4 (bottom): 2 points
    for i in range(2):
        x = (i - 0.5) * 1.0
        y = 2.0 - 2.0*np.sqrt(3)
        base_points.append([x, y])
    
    base_points = np.array(base_points, dtype=float)
    centroid = base_points.mean(axis=0)
    base_points -= centroid
    
    base_points[:, 1] *= aspect
    shear_matrix = np.array([[1, np.tan(shear)], [0, 1]])
    base_points = base_points @ shear_matrix.T
    
    dists_from_center = np.linalg.norm(base_points, axis=1)
    median_dist = np.median(dists_from_center)
    outer_mask = dists_from_center > median_dist
    
    if np.any(outer_mask):
        outer_points = base_points[outer_mask]
        outer_dists = np.linalg.norm(outer_points, axis=1)
        outer_dists[outer_dists < 1e-12] = 1e-12
        scale_factors = outer_scale / outer_dists
        base_points[outer_mask] *= scale_factors[:, np.newaxis]
        
    return base_points

def make_seeds():
    """Generate multiple diverse initial configurations."""
    seeds = []
    rng = np.random.default_rng(42)
    
    # 1. 4x4 grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    seeds.append(grid)
    
    # 2. Hexagonal lattice subset
    hex_pts = []
    for row in range(4):
        x_offset = (row % 2) * 0.5
        for col in range(4):
            hex_pts.append([col + x_offset, row * np.sqrt(3)/2])
    hex_pts = np.array(hex_pts[:16])
    seeds.append(hex_pts)
    
    # 3. Concentric circles: 1 center + 5 + 10
    pts = [np.array([0.0, 0.0])]
    for i in range(5):
        angle = 2 * np.pi * i / 5
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(10):
        angle = 2 * np.pi * i / 10 + np.pi/10
        pts.append(np.array([2.2 * np.cos(angle), 2.2 * np.sin(angle)]))
    seeds.append(np.array(pts))
    
    # 4. Two concentric octagons
    pts = []
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pts.append(np.array([1.0 * np.cos(angle), 1.0 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.0 * np.cos(angle), 2.0 * np.sin(angle)]))
    seeds.append(np.array(pts))
    
    # 5. 1 center + 6 + 9
    seeds.append(make_ring_topology(6, 9, 1.0, 2.0))
    
    # 6. 1 center + 7 + 8
    seeds.append(make_ring_topology(7, 8, 1.0, 2.2))
    
    # 7. 1 center + 8 + 7
    seeds.append(make_ring_topology(8, 7, 1.0, 2.0))
    
    # 8. 2 center + 6 + 8
    pts = [np.array([-0.5, 0.0]), np.array([0.5, 0.0])]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append(np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle)]))
    for i in range(8):
        angle = 2 * np.pi * i / 8 + np.pi/8
        pts.append(np.array([2.8 * np.cos(angle), 2.8 * np.sin(angle)]))
    seeds.append(np.array(pts))
    
    # 9. Regular 16-gon
    pts = []
    for i in range(16):
        angle = 2 * np.pi * i / 16
        pts.append(np.array([np.cos(angle), np.sin(angle)]))
    seeds.append(np.array(pts))
    
    # 10-13. Hexagonal cluster variants
    hex_params_list = [
        [1.0, 0.0, 1.0],
        [1.1, 0.1, 1.05],
        [0.9, -0.1, 0.95],
        [1.2, 0.2, 1.1],
    ]
    for params in hex_params_list:
        pts = make_hexagonal_cluster(np.array(params))
        seeds.append(pts)
        
    # 14-16. Particle Equilibrium Seeds
    for rep_str in [1.0, 2.0]:
        seed_pts = rng.uniform(-2, 2, (N, 2))
        # Simple relaxation
        for _ in range(100):
            forces = np.zeros_like(seed_pts)
            for i in range(N):
                for j in range(i+1, N):
                    diff = seed_pts[i] - seed_pts[j]
                    dist_sq = np.sum(diff**2)
                    dist = np.sqrt(dist_sq) + 1e-6
                    force_mag = rep_str / (dist_sq**2 + 0.1)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
            seed_pts += 0.05 * forces
            # Normalize scale to avoid drift
            seed_pts -= np.mean(seed_pts, axis=0)
            scale = np.sqrt(np.mean(np.sum(seed_pts**2, axis=1)))
            if scale > 1e-6:
                seed_pts /= scale
        seeds.append(seed_pts)
        
    return seeds

def simulated_annealing(init_pts, max_iters=5000, T_start=0.5, T_end=0.01, alpha=0.9998):
    """
    Simulated Annealing to refine point positions.
    Uses exact objective for acceptance.
    """
    pts = init_pts.copy()
    current_score = objective_exact(pts.flatten())
    best_pts = pts.copy()
    best_score = current_score
    T = T_start
    
    # Precompute indices for speed? No, random selection is fine.
    for i in range(max_iters):
        # Time check
        if time.time() > deadline - 5:
            break
            
        # Move a random point
        idx = np.random.randint(0, N)
        # Step size decreases with temperature? Or fixed? 
        # Let's scale step with T to allow larger jumps early on.
        step_size = 0.1 * T / T_start
        displacement = np.random.normal(0, step_size, 2)
        
        new_pts = pts.copy()
        new_pts[idx] += displacement
        
        # Normalize to prevent drift (optional, but helps keep points in view)
        # new_pts -= np.mean(new_pts, axis=0)
        
        new_score = objective_exact(new_pts.flatten())
        
        # Acceptance criterion
        if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
            current_score = new_score
            pts = new_pts
            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        
        T *= alpha
        if T < T_end:
            T = T_end
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    global deadline
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_x = None
    best_val = np.inf
    
    def update_best(x, val):
        nonlocal best_x, best_val
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # Generate seeds
    seeds = make_seeds()
    
    # Score seeds and sort
    scored_seeds = []
    for s in seeds:
        # Ensure shape is correct
        if s.shape != (N, D):
            continue
        val = objective_exact(s.flatten())
        scored_seeds.append((val, s))
        
    scored_seeds.sort(key=lambda x: x[0])
    
    # Process top seeds
    # We have ~16 seeds. Let's process the top 8 or 10.
    num_to_process = min(10, len(scored_seeds))
    
    for i in range(num_to_process):
        if time.time() > deadline - 20:
            break
            
        curr_score, pts = scored_seeds[i]
        
        if curr_score < best_val:
            best_val = curr_score
            best_x = pts.flatten().copy()
            
        # Phase 1: Simulated Annealing
        # SA is stochastic, so run it.
        # Use a moderate number of iterations.
        sa_iters = 3000
        if time.time() > deadline - 30:
            sa_iters = 1000
            
        sa_pts, sa_score = simulated_annealing(pts, max_iters=sa_iters)
        
        if sa_score < best_val:
            best_val = sa_score
            best_x = sa_pts.flatten().copy()
            
        # Phase 2: Polish with L-BFGS-B on Smooth Objective
        # This uses gradients to fine-tune.
        if time.time() < deadline - 10:
            try:
                # Use the best of SA result or original seed? SA result is better.
                start_x = sa_pts.flatten()
                
                def smooth_obj(x):
                    return objective_smooth(x, beta=100.0)
                
                res = minimize(
                    smooth_obj,
                    start_x,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Evaluate exact score of the polished result
                exact_score = objective_exact(res.x)
                if exact_score < best_val:
                    best_val = exact_score
                    best_x = res.x.copy()
                    
                # Phase 3: Final Nelder-Mead polish on exact objective
                # This snaps to the local minimum of the non-smooth function
                if time.time() < deadline - 5:
                    res2 = minimize(
                        objective_exact,
                        res.x,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                    )
                    final_score = objective_exact(res2.x)
                    if final_score < best_val:
                        best_val = final_score
                        best_x = res2.x.copy()
                        
            except Exception:
                pass

    # If we still have time, do a global SA on the best found so far?
    # Or just return.
    
    if best_x is None:
        # Fallback to grid
        grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        return grid
    
    return best_x.reshape(N, D)
