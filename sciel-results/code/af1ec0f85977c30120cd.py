"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer for 150s limit
    
    def compute_radius(c):
        """Compute the minimum radius for a configuration."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    def compute_gradient(c, beta=100.0):
        """Compute gradient using soft-min (log-sum-exp) for smoothness."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        # Soft-min of pair distances
        min_d = np.min(d)
        # Use log-sum-exp for soft-min
        d_clipped = np.clip(d, min_d * 0.5, None)
        log_sum_exp = min_d + np.log(np.sum(np.exp(-(d_clipped - min_d) * beta), axis=None))
        soft_min_pair = -log_sum_exp / beta
        
        # Soft-min of wall distances
        min_w = np.min(dist_to_walls)
        w_clipped = np.clip(dist_to_walls, min_w * 0.5, None)
        log_sum_exp_w = min_w + np.log(np.sum(np.exp(-(w_clipped - min_w) * beta)))
        soft_min_wall = -log_sum_exp_w / beta
        
        # Soft-min of both
        all_vals = np.concatenate([soft_min_pair * np.ones(1), soft_min_wall * np.ones(1)])
        min_all = np.min(all_vals)
        log_sum_exp_all = min_all + np.log(np.sum(np.exp(-(all_vals - min_all) * beta)))
        soft_min_all = -log_sum_exp_all / beta
        
        # Gradient with respect to c
        G = np.zeros_like(c)
        
        # Pair contribution
        weights = np.exp(-(d - min_d) * beta)
        weights /= np.sum(weights)
        np.fill_diagonal(weights, 0)
        
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] > 1e-9:
                    vec = c[i] - c[j]
                    norm = d[i, j]
                    w = weights[i, j]
                    G[i] += w * vec / norm
                    G[j] -= w * vec / norm
        
        # Wall contribution
        w_weights = np.exp(-(dist_to_walls - min_w) * beta)
        w_weights /= np.sum(w_weights)
        
        for i in range(N):
            if w_weights[i] > 1e-10:
                for k in range(3):
                    if c[i, k] > 0:
                        G[i, k] -= w_weights[i]
                    else:
                        G[i, k] += w_weights[i]
        
        norm_G = np.linalg.norm(G)
        if norm_G > 1e-12:
            G /= norm_G
        
        return G, soft_min_all
    
    def compute_active_gradient(c, r, eps=1e-4):
        """Compute gradient using active-set (exact min)."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
        np.fill_diagonal(d, np.inf)
        
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        
        G = np.zeros_like(c)
        
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < 2 * r + eps:
                    vec = c[i] - c[j]
                    norm = d[i, j]
                    if norm > 1e-9:
                        G[i] += vec / norm
                        G[j] -= vec / norm
        
        for i in range(N):
            if dist_to_walls[i] < r + eps:
                for k in range(3):
                    if c[i, k] > 0:
                        G[i, k] -= 1.0
                    else:
                        G[i, k] += 1.0
        
        norm_G = np.linalg.norm(G)
        if norm_G > 1e-12:
            G /= norm_G
        
        return G
    
    # --- Phase 0: Generate diverse initial configurations ---
    
    # 1. Simple Cubic Grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N].copy()
    
    # 2. FCC Lattice initialization
    def generate_fcc(n_points):
        """Generate FCC lattice points and select n_points."""
        # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
        # Generate a 4x4x4 block of FCC unit cells
        m_fcc = 4
        pts = []
        for i in range(m_fcc + 1):
            for j in range(m_fcc + 1):
                for k in range(m_fcc + 1):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k])
                    pts.append([i + 0.5, j, k + 0.5])
                    pts.append([i, j + 0.5, k + 0.5])
        
        pts = np.array(pts)
        
        # Center the points
        center = pts.mean(axis=0)
        pts = pts - center
        
        # Scale to fit in [-0.5, 0.5] with margin
        max_extent = np.max(np.abs(pts))
        target_extent = 0.5 * 0.95
        scale = target_extent / max_extent
        pts = pts * scale
        
        # Select n_points randomly
        indices = np.random.choice(len(pts), n_points, replace=False)
        return pts[indices]
    
    init_fcc = generate_fcc(N)
    
    # 3. BCC Lattice initialization
    def generate_bcc(n_points):
        """Generate BCC lattice points and select n_points."""
        m_bcc = 3
        pts = []
        for i in range(m_bcc + 1):
            for j in range(m_bcc + 1):
                for k in range(m_bcc + 1):
                    pts.append([i, j, k])
                    if i < m_bcc and j < m_bcc and k < m_bcc:
                        pts.append([i + 0.5, j + 0.5, k + 0.5])
        
        pts = np.array(pts)
        
        center = pts.mean(axis=0)
        pts = pts - center
        
        max_extent = np.max(np.abs(pts))
        target_extent = 0.5 * 0.95
        scale = target_extent / max_extent
        pts = pts * scale
        
        indices = np.random.choice(len(pts), n_points, replace=False)
        return pts[indices]
    
    init_bcc = generate_bcc(N)
    
    # 4. Random initialization
    init_rand = np.random.uniform(-0.45, 0.45, (N, 3))
    
    initializations = [init_grid, init_fcc, init_bcc, init_rand]
    
    best_overall_r = -1.0
    best_overall_centers = None
    
    # Quick local improvement for each initialization
    for init_c in initializations:
        if time.time() - start_time > time_limit * 0.15:
            break
            
        c = init_c.copy()
        r = compute_radius(c)
        
        # Quick gradient ascent
        step = 0.01
        for _ in range(300):
            if time.time() - start_time > time_limit * 0.15:
                break
                
            G, _ = compute_gradient(c, beta=50.0)
            new_c = c + step * G
            new_c = np.clip(new_c, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_r = compute_radius(new_c)
            if new_r > r:
                c = new_c
                r = new_r
            else:
                step *= 0.7
                if step < 1e-6:
                    break
        
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = c.copy()

    centers = best_overall_centers.copy()
    best_r = best_overall_r
    best_centers = centers.copy()
    
    # --- Phase 1: Simulated Annealing with vectorized operations ---
    T = 0.015
    cooling_rate = 0.995
    min_T = 1e-7
    sa_deadline = time_limit * 0.65
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < sa_deadline and T > min_T:
        G, _ = compute_gradient(centers, beta=30.0)
        
        r = compute_radius(centers)
        
        # Propose move
        step_size = 0.008 * (T / 0.015) + 0.001
        noise_scale = 0.008 * (T / 0.015)
        noise = np.random.normal(0, noise_scale, centers.shape)
        new_centers = centers + step_size * G + noise
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = compute_radius(new_centers)
        
        delta = new_r - r
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        T *= cooling_rate
        
        # Occasional restart if stuck
        if no_improve_count > max_no_improve and T > min_T * 10:
            # Inject a larger kick
            kick = np.random.normal(0, 0.02, centers.shape)
            centers = centers + kick
            centers = np.clip(centers, -0.5 + 1e-9, 0.5 - 1e-9)
            no_improve_count = 0
    
    # --- Phase 2: Active-set local optimization ---
    local_deadline = time_limit * 0.85
    
    centers = best_centers.copy()
    step = 0.005
    
    for _ in range(2000):
        if time.time() - start_time > local_deadline:
            break
        
        r = compute_radius(centers)
        G = compute_active_gradient(centers, r, eps=1e-4)
        
        if np.linalg.norm(G) < 1e-10:
            break
        
        new_centers = centers + step * G
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = compute_radius(new_centers)
        
        if new_r > r:
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
            step = min(step * 1.05, 0.01)
        else:
            step *= 0.5
            if step < 1e-8:
                break
    
    # --- Phase 3: Final polish with very fine steps ---
    polish_deadline = time_limit - 2.0
    centers = best_centers.copy()
    step = 0.001
    
    for _ in range(5000):
        if time.time() - start_time > polish_deadline:
            break
        
        r = compute_radius(centers)
        G = compute_active_gradient(centers, r, eps=1e-5)
        
        if np.linalg.norm(G) < 1e-12:
            break
        
        new_centers = centers + step * G
        new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = compute_radius(new_centers)
        
        if new_r > r:
            centers = new_centers
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            step *= 0.5
            if step < 1e-10:
                break
    
    # Final safety check
    final_centers = best_centers.copy()
    final_centers = np.clip(final_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return final_centers
# EVOLVE-BLOCK-END
