import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)

    def compute_score(pts):
        """Compute the packing radius score."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        return np.min(all_vals)

    def optimize(pts, alpha_init, k_init, k_end, cooling, time_budget,
                 stall_threshold=100, noise_scale_base=0.01):
        """Main optimization loop using soft-min gradient ascent."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            # Compute distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute soft-min weights
            shifted = all_vals - current_min
            # Use a stable log-sum-exp like weighting
            # We want weights proportional to exp(-k * (val - min))
            # To avoid underflow/overflow, shift by min
            # Since shifted >= 0, exp(-k*shifted) is in [0, 1]
            weights = np.exp(-k * shifted)
            sum_w = np.sum(weights)
            if sum_w > 1e-15:
                weights /= sum_w
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
            
            # Compute gradients
            grads = np.zeros_like(pts)
            
            # Pair contributions
            pair_weights = weights[:num_pairs]
            d_pair = dists[i_idx, j_idx]
            mask_nonzero = d_pair > 1e-10
            if np.any(mask_nonzero):
                # Vectorized gradient accumulation
                # d/dp_i (||p_i - p_j||/2) = (p_i - p_j) / (2 * ||p_i - p_j||)
                # d/dp_j (||p_i - p_j||/2) = (p_j - p_i) / (2 * ||p_i - p_j||)
                
                # We need to sum w_k * grad_k for all pairs involving i
                # Using np.add.at for scattered addition
                diff_m = pts[i_idx[mask_nonzero]] - pts[j_idx[mask_nonzero]]
                d_m = d_pair[mask_nonzero]
                w_m = pair_weights[mask_nonzero]
                
                # Gradient for i: sum (w * diff / d)
                # Gradient for j: sum (-w * diff / d)
                
                grad_i = diff_m * (w_m / d_m)[:, np.newaxis]
                grad_j = -diff_m * (w_m / d_m)[:, np.newaxis]
                
                np.add.at(grads, i_idx[mask_nonzero], grad_i)
                np.add.at(grads, j_idx[mask_nonzero], grad_j)
            
            # Boundary contributions
            bound_weights = weights[num_pairs:]
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            # d/dp (1 - ||p||) = -p / ||p||
            bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
            grads += bound_grads
            
            # Normalize gradient to have unit norm for stable step size
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-10:
                grads /= grad_norm
            else:
                break # Converged?
            
            # Update
            pts += alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase k (sharpen the min)
            progress = min(1.0, (time.time() - start_time) / time_budget)
            k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005)
        
        return best_pts, best_score

    def local_search(pts, time_budget):
        """Simple local search: repeatedly push apart the closest pair."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            pair_vals = dists[i_idx, j_idx] / 2.0
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
            
            # Find the minimum constraint
            min_idx = np.argmin(all_vals)
            
            # Apply a small push
            step = 0.001
            if min_idx < num_pairs:
                i = i_idx[min_idx]
                j = j_idx[min_idx]
                vec = pts[i] - pts[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec /= norm
                    pts[i] += step * vec
                    pts[j] -= step * vec
            else:
                i = min_idx - num_pairs
                vec = pts[i]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec /= norm
                    pts[i] -= step * vec # Push inward
            
            # Project
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
        
        return best_pts, best_score

    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            # Keep points closest to center? Or random?
            # For packing, we want to fill the circle.
            # Let's select N points that are well-distributed.
            # A simple heuristic: sort by distance to center, take first N? 
            # This might bias towards center. 
            # Better: random selection from the lattice points?
            # Or: use a KDTree to find a max-min set?
            # For now, random selection is robust.
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts

    def random_init():
        """Random initialization with rejection sampling."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)

    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.90, 0.95, 1.0, 1.05]
    
    for mult in spacing_multipliers:
        if time.time() - start_time > time_limit * 0.50:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        time_budget = min(15.0, time_limit * 0.15)
        pts_opt, score = optimize(pts,
                                   alpha_init=0.02,
                                   k_init=5.0,
                                   k_end=500.0,
                                   cooling=0.999,
                                   time_budget=time_budget,
                                   stall_threshold=50,
                                   noise_scale_base=0.01)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.70:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.70:
                break
            
            pts = random_init()
            time_budget = min(10.0, time_limit * 0.10)
            pts_opt, score = optimize(pts,
                                       alpha_init=0.01,
                                       k_init=2.0,
                                       k_end=200.0,
                                       cooling=0.998,
                                       time_budget=time_budget,
                                       stall_threshold=30,
                                       noise_scale_base=0.02)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish with gradient-based optimization (sharper)
    if best_pts is not None and time.time() - start_time < time_limit * 0.85:
        remaining_time = time_limit * 0.90 - time.time()
        if remaining_time > 3.0:
            pts_polished, score_polished = optimize(best_pts,
                                                     alpha_init=0.005,
                                                     k_init=500.0,
                                                     k_end=2000.0,
                                                     cooling=0.995,
                                                     time_budget=min(remaining_time, 15.0),
                                                     stall_threshold=100,
                                                     noise_scale_base=0.001)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
    
    # Phase 4: Local search to escape local minima
    if best_pts is not None and time.time() - start_time < time_limit * 0.95:
        remaining_time = time_limit * 0.98 - time.time()
        if remaining_time > 3.0:
            pts_ls, score_ls = local_search(best_pts, min(remaining_time, 10.0))
            
            if score_ls > best_score:
                best_score = score_ls
                best_pts = pts_ls

    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
