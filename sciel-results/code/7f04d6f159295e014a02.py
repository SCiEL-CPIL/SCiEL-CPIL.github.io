import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        if not np.all(np.isfinite(centers)):
            return 0.0
        if centers.shape != (N, 3):
            return 0.0
        
        # Distance to walls
        dists_to_walls = 0.5 - np.abs(centers)
        min_wall_dist = np.min(dists_to_walls)
        
        # Pairwise distances
        dists = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        return min(min_wall_dist, min_pair_dist / 2.0)

    def soft_min_objective(centers_flat, beta, iu):
        centers = centers_flat.reshape(-1, 3)
        
        # 1. Wall distances: 0.5 - |c_i|
        dists_to_walls = 0.5 - np.abs(centers)
        wall_vals = dists_to_walls.flatten()
        
        # 2. Pairwise distances / 2
        dists = cdist(centers, centers, metric='euclidean')
        pair_vals = dists[iu] / 2.0
        
        all_vals = np.concatenate([wall_vals, pair_vals])
        
        # Calculate Soft Min: (1/beta) * ln(sum(exp(-beta * x)))
        neg_beta_x = -beta * all_vals
        max_val = np.max(neg_beta_x)
        
        exp_terms = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_terms)
        
        if sum_exp == 0:
            return 1e10
            
        obj = (max_val + np.log(sum_exp)) / beta
        
        return obj

    def build_bcc_init(n=100, noise_scale=0.005, seed=42):
        nx, ny, nz = 5, 5, 4
        margin = 0.02
        xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
        ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
        zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
        
        all_centers = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    all_centers.append([xs[i], ys[j], zs[k]])
                    if i < nx-1 and j < ny-1 and k < nz-1:
                        all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2])
        
        all_centers = np.array(all_centers)
        rng = np.random.default_rng(seed)
        
        if len(all_centers) >= N:
            idx = rng.choice(len(all_centers), N, replace=False)
            centers = all_centers[idx]
        else:
            centers = all_centers
            while len(centers) < N:
                centers = np.vstack([centers, centers[0]])
        
        centers += rng.normal(0, noise_scale, centers.shape)
        centers = np.clip(centers, -0.499, 0.499)
        
        return centers

    def build_halton_init(n=100, seed=42):
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result
        
        rng = np.random.default_rng(seed)
        centers = np.zeros((n, 3))
        for i in range(1, n+1):
            centers[i-1, 0] = -0.5 + halton(i, 2)
            centers[i-1, 1] = -0.5 + halton(i, 3)
            centers[i-1, 2] = -0.5 + halton(i, 5)
        
        centers += rng.normal(0, 0.005, centers.shape)
        centers = np.clip(centers, -0.499, 0.499)
        
        return centers

    def optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=6):
        best_centers = centers.copy()
        best_radius = compute_radius(best_centers)
        
        iu = np.triu_indices(N, k=1)
        betas = np.geomspace(beta_start, beta_end, n_beta_steps)
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            
            x0 = best_centers.flatten()
            
            try:
                res = minimize(
                    soft_min_objective,
                    x0,
                    args=(beta, iu),
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
                
                r = compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers.copy()
            except Exception:
                pass
        
        return best_centers, best_radius

    def polish_repulsion(centers, deadline, iu):
        centers = centers.copy()
        r_current = compute_radius(centers)
        best_r = r_current
        best_centers = centers.copy()
        
        step_size = 0.002
        decay = 0.95
        min_step = 1e-8
        margin = 1.01
        
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 2.0 and step_size > min_step and no_improve_count < max_no_improve:
            r = compute_radius(centers)
            if r > best_r + 1e-12:
                best_r = r
                best_centers = centers.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Compute forces
            forces = np.zeros_like(centers)
            
            # Wall forces
            for k in range(3):
                dists = 0.5 - np.abs(centers[:, k])
                limit = max(r_current * margin, 0.01)
                mask = dists < limit
                if np.any(mask):
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1.0
                    strengths = (limit - dists[mask]) / limit
                    forces[mask, k] += dirs * strengths
                    
            # Pairwise forces using KD-tree
            tree = cKDTree(centers)
            search_radius = 2 * max(r_current * margin, 0.01)
            neighbor_pairs = tree.query_pairs(r=search_radius)
            
            for i, j in neighbor_pairs:
                if i == j:
                    continue
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-9 and dist < search_radius:
                    strength = (search_radius - dist) / search_radius
                    unit_diff = diff / dist
                    forces[i] += strength * unit_diff
                    forces[j] -= strength * unit_diff
            
            # Normalize forces
            norms = np.linalg.norm(forces, axis=1, keepdims=True)
            norms[norms == 0] = 1
            forces = forces / norms
            
            centers += step_size * forces
            centers = np.clip(centers, -0.4999, 0.4999)
            
            step_size *= decay
            
        return best_centers, best_r

    # --- Main Logic ---

    # 1. Initialize with BCC
    init_centers = build_bcc_init()
    best_centers = init_centers.copy()
    best_r = compute_radius(best_centers)
    
    # 2. Run optimization from BCC
    c_opt, r_opt = optimize_from_initial(init_centers, deadline)
    if r_opt > best_r:
        best_r = r_opt
        best_centers = c_opt.copy()
    
    # 3. If time permits, try Halton initialization
    if time.time() < deadline - 30.0:
        init_halton = build_halton_init()
        c_opt_h, r_opt_h = optimize_from_initial(init_halton, deadline)
        if r_opt_h > best_r:
            best_r = r_opt_h
            best_centers = c_opt_h.copy()

    # 4. Repulsion Polish
    if time.time() < deadline - 5.0:
        iu = np.triu_indices(N, k=1)
        c_polish, r_polish = polish_repulsion(best_centers, deadline, iu)
        if r_polish > best_r:
            best_r = r_polish
            best_centers = c_polish.copy()

    # 5. Final high-precision polish (optional, if time permits)
    if time.time() < deadline - 10.0:
        iu = np.triu_indices(N, k=1)
        beta = 10000.0
        try:
            res = minimize(
                soft_min_objective,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            r = compute_radius(new_centers)
            if r > best_r:
                best_r = r
                best_centers = new_centers
        except Exception:
            pass

    # 6. Final validation and safety
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    if not np.all(np.isfinite(best_centers)):
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N].astype(np.float64)

    return best_centers
