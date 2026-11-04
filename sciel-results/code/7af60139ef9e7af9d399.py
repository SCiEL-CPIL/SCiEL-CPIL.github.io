"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for 150s limit

    def min_radius(c):
        """Compute exact minimum radius for a set of centers."""
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        dist_to_walls = np.min(np.minimum(c + 0.5, 0.5 - c), axis=1)
        min_wall_dist = np.min(dist_to_walls)
        return min(min_pair_dist / 2.0, min_wall_dist)

    def compute_gradient(centers, r, margin=0.02):
        """Compute subgradient of the min-radius objective (vectorized).
        
        For pairs with distance < 2*(r + margin), push apart.
        For spheres with wall distance < r + margin, push inward.
        """
        G = np.zeros_like(centers)
        
        # Pairwise repulsion (vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(d, np.inf)
        
        # Mask: pairs that are too close
        threshold = 2.0 * (r + margin)
        mask = d < threshold
        # Exclude diagonal (already inf)
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            # Normalize direction vectors
            safe_d = np.where(mask, d, 1.0)
            # Unit vectors: diff[i,j] / d[i,j] points from j to i
            unit = diff / safe_d[:, :, np.newaxis]
            # Weight by closeness (stronger push when closer)
            weight = np.where(mask, 1.0 / (safe_d + 1e-12), 0.0)
            weighted_unit = unit * weight[:, :, np.newaxis]
            
            # Sum contributions for each center
            # G[i] += sum_j weighted_unit[i,j] (push i away from j)
            G += np.sum(weighted_unit, axis=1)
            # G[j] -= sum_i weighted_unit[i,j]
            # By symmetry, G[j] -= sum_i weighted_unit[i,j] = -sum_i weighted_unit[j,i]
            # So total G[i] = sum_j weighted_unit[i,j] - sum_j weighted_unit[j,i]
            # But weighted_unit[j,i] = -weighted_unit[i,j] * (d[i,j]/d[j,i]) ... 
            # Actually diff[j,i] = -diff[i,j], so unit[j,i] = -unit[i,j]
            # weight[j,i] = weight[i,j]
            # So weighted_unit[j,i] = -weighted_unit[i,j]
            # G[i] should get: +sum_j w[i,j]*u[i,j] from pairs (i,j) where i>j
            # and: -sum_j w[j,i]*u[j,i] = +sum_j w[i,j]*u[i,j] from pairs (j,i) where j<i
            # Total: G[i] = 2 * sum_{j: mask[i,j]} w[i,j]*u[i,j]
            # But we already computed G[i] = sum_j weighted_unit[i,j]
            # We need G[i] = sum_j w[i,j]*u[i,j] - sum_j w[j,i]*u[j,i]
            # = sum_j w[i,j]*u[i,j] + sum_j w[i,j]*u[i,j]  (since u[j,i]=-u[i,j], w[j,i]=w[i,j])
            # = 2 * sum_j w[i,j]*u[i,j]
            # So multiply by 2? No, let's just do it properly:
            pass
            
        # Redo gradient computation more carefully
        G = np.zeros_like(centers)
        if np.any(mask):
            # For each pair (i,j) with i<j and mask[i,j]:
            # G[i] += w * u[i,j]  (push i away from j)
            # G[j] -= w * u[i,j]  (push j away from i)
            # where u[i,j] = (c[i]-c[j])/d[i,j], w = 1/d[i,j]
            # Vectorized: use np.add.at
            rows, cols = np.where(mask)
            # Ensure i < j for consistency
            valid = rows < cols
            rows = rows[valid]
            cols = cols[valid]
            
            if len(rows) > 0:
                vecs = diff[rows, cols]  # c[rows] - c[cols]
                dists = d[rows, cols]
                weights = 1.0 / (dists + 1e-12)
                grads = vecs * weights[:, np.newaxis]
                
                np.add.at(G, rows, grads)
                np.add.at(G, cols, -grads)
        
        # Wall repulsion
        dist_to_walls = np.min(np.minimum(centers + 0.5, 0.5 - centers), axis=1)
        wall_mask = dist_to_walls < (r + margin)
        
        if np.any(wall_mask):
            # For each sphere near a wall, push it inward
            for idx in np.where(wall_mask)[0]:
                for k in range(3):
                    # Distance to each wall face
                    dist_pos = centers[idx, k] + 0.5   # distance to +face
                    dist_neg = 0.5 - centers[idx, k]   # distance to -face
                    if dist_pos < dist_neg:
                        # Closest wall is +face, push in -direction
                        G[idx, k] -= 1.0
                    else:
                        # Closest wall is -face, push in +direction
                        G[idx, k] += 1.0
        
        return G

    def optimize_from(centers, time_budget):
        """Run direct subgradient ascent with adaptive steps from initial centers."""
        centers = centers.copy()
        current_r = min_radius(centers)
        best_r_local = current_r
        best_c_local = centers.copy()
        
        # Adaptive step size
        step = 0.005
        patience = 0
        max_patience = 50
        
        n_iter = 0
        max_iter = 200000
        
        while time.time() - start_time < time_budget and n_iter < max_iter:
            n_iter += 1
            margin = max(0.005, 0.02 * (1.0 - n_iter / max_iter))
            G = compute_gradient(centers, current_r, margin)
            gnorm = np.linalg.norm(G)
            
            if gnorm < 1e-10:
                break
            
            G_norm = G / gnorm
            new_centers = centers + step * G_norm
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = min_radius(new_centers)
            
            if new_r > current_r:
                centers = new_centers
                current_r = new_r
                step = min(step * 1.05, 0.02)
                patience = 0
                if current_r > best_r_local:
                    best_r_local = current_r
                    best_c_local = centers.copy()
            else:
                patience += 1
                step *= 0.9
                if patience > max_patience:
                    break
        
        return best_c_local, best_r_local

    # Multiple restarts with different initializations
    best_centers = None
    best_r = 0.0
    
    # Initialization 1: Jittered grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Initialization 2: BCC-like lattice
    # BCC has 2 atoms per unit cell. For 50 spheres, use ~3x3x3 BCC = 54 sites, take 50.
    # BCC positions in [0,1]^3: corners (0,0,0) etc + body center (0.5,0.5,0.5)
    bcc_sites = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                # Corner positions
                bcc_sites.append([ix * 0.25 - 0.5, iy * 0.25 - 0.5, iz * 0.25 - 0.5])
                # Body center
                bcc_sites.append([ix * 0.25 - 0.25, iy * 0.25 - 0.25, iz * 0.25 - 0.25])
    bcc_sites = np.array(bcc_sites)
    # Scale to fit in [-0.5, 0.5]
    # The BCC sites span from -0.5 to 0.75, need to rescale
    # Actually let's use a proper BCC in [-0.5, 0.5]
    # Use 3x3x3 grid of BCC cells, each cell has 2 points
    bcc_pts = []
    cell_size = 1.0 / 3.0
    for ix in range(3):
        for iy in range(3):
            for iz in range(3):
                base = np.array([ix, iy, iz]) * cell_size - 0.5
                bcc_pts.append(base)
                bcc_pts.append(base + np.array([cell_size, cell_size, cell_size]) * 0.5)
    bcc_pts = np.array(bcc_pts)  # 54 points
    bcc_pts = bcc_pts[:N].copy()
    bcc_pts += np.random.normal(0, 0.005, bcc_pts.shape)
    bcc_pts = np.clip(bcc_pts, -0.49, 0.49)
    
    inits = [
        grid[:N].copy() + np.random.normal(0, 0.01, (N, 3)),
        bcc_pts,
        grid[:N].copy() + np.random.normal(0, 0.02, (N, 3)),
        bcc_pts + np.random.normal(0, 0.01, (N, 3)),
    ]
    
    for init_idx, init_c in enumerate(inits):
        init_c = np.clip(init_c, -0.49, 0.49)
        elapsed = time.time() - start_time
        remaining = time_limit - 5 - elapsed
        if remaining < 5:
            break
        
        # Allocate time: more time for later restarts if budget allows
        if init_idx == 0:
            budget = remaining * 0.35
        elif init_idx == 1:
            budget = remaining * 0.30
        elif init_idx == 2:
            budget = remaining * 0.20
        else:
            budget = remaining * 0.15
        
        budget = max(budget, 3.0)
        c_opt, r_opt = optimize_from(init_c, time_limit - 5 - elapsed + budget)
        
        if r_opt > best_r:
            best_r = r_opt
            best_centers = c_opt.copy()
    
    # Final polish on best result
    if best_centers is not None:
        elapsed = time.time() - start_time
        remaining = time_limit - 3 - elapsed
        if remaining > 2:
            c_opt, r_opt = optimize_from(best_centers, time_limit - 3 - elapsed + remaining)
            if r_opt > best_r:
                best_r = r_opt
                best_centers = c_opt.copy()
    
    return best_centers
# EVOLVE-BLOCK-END
