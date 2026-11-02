import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(c):
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        try:
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
        except Exception:
            r_pair = 0.0
        return min(r_wall, r_pair)

    def soft_min_objective(x, beta=50.0):
        c = x.reshape(N, 3)
        wall_dists = 0.5 - np.abs(c)
        b = np.min(wall_dists, axis=1)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        a = dists / 2.0
        
        a_vals = a[np.triu_indices(N, k=1)]
        b_vals = b
        all_vals = np.concatenate([a_vals, b_vals])
        
        min_val = np.min(all_vals)
        exp_terms = np.exp(-beta * (all_vals - min_val))
        soft_min_val = -min_val - (1.0 / beta) * np.log(np.sum(exp_terms))
        obj = -soft_min_val
        
        grad_all = -exp_terms / np.sum(exp_terms)
        grad_a = grad_all[:len(a_vals)]
        grad_b = grad_all[len(a_vals):]
        
        grad = np.zeros_like(c)
        
        # Wall gradient (vectorized)
        min_comp = np.argmin(wall_dists, axis=1)
        for i in range(N):
            k = min_comp[i]
            s = np.sign(c[i, k])
            if s == 0:
                s = 1.0
            grad[i, k] += grad_b[i] * (-s)
        
        # Pairwise gradient (vectorized)
        iu_i, iu_j = np.triu_indices(N, k=1)
        d = dists[iu_i, iu_j]
        valid = d > 1e-12
        if np.any(valid):
            direction = diff[iu_i[valid], iu_j[valid], :] / d[valid, np.newaxis]
            # grad_a[idx] * direction / 2 for i, -grad_a[idx] * direction / 2 for j
            np.add.at(grad, iu_i[valid], grad_a[valid, np.newaxis] * direction * 0.5)
            np.add.at(grad, iu_j[valid], -grad_a[valid, np.newaxis] * direction * 0.5)
        
        return obj, grad.ravel()

    def polish(x0, time_limit, beta=100.0, max_iter=300):
        def objective_and_grad(x):
            if time.time() > time_limit:
                return 1e10, np.zeros_like(x)
            return soft_min_objective(x, beta=beta)
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.499999, 0.499999)] * (N * 3),
                options={'maxiter': max_iter, 'maxfun': 800, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            return new_centers
        except Exception:
            return x0.reshape(N, 3)

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        base_step = 0.03 * step_scale
        decay_rate = 0.995
        stagnation_count = 0
        it = 0
        
        while time.time() < time_limit:
            it += 1
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            
            try:
                tree = cKDTree(centers)
                dists, _ = tree.query(centers, k=2)
                min_pair_dist = np.min(dists[:, 1])
                r_pair = min_pair_dist / 2.0
            except Exception:
                break
                
            current_r = min(r_wall, r_pair)
            
            if current_r <= best_r * 1.000001:
                stagnation_count += 1
                if stagnation_count > 80:
                    break
            else:
                stagnation_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            force = np.zeros_like(centers)
            target_r = current_r + 0.001
            
            # Wall forces
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                mask = d_wall < target_r
                if np.any(mask):
                    viol = target_r - d_wall[mask]
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1.0
                    force[mask, k] += viol * dirs
            
            # Pair forces
            target_dist = 2 * target_r
            try:
                pairs = tree.query_pairs(target_dist)
            except Exception:
                pairs = []
            
            for i, j in pairs:
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10 and d < target_dist:
                    f_mag = (target_dist - d) / d
                    force[i] += f_mag * diff
                    force[j] -= f_mag * diff
            
            force_norm = np.linalg.norm(force)
            if force_norm < 1e-12:
                break
            force = force / force_norm
            
            step = base_step * (decay_rate ** (stagnation_count / 10.0))
            new_centers = centers + step * force
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            new_r = compute_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
            else:
                step *= 0.5
                new_centers = centers + step * force
                new_centers = np.clip(new_centers, -0.499999, 0.499999)
                new_r = compute_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                else:
                    centers += np.random.normal(0, 0.001, centers.shape)
                    centers = np.clip(centers, -0.499999, 0.499999)
                    r = compute_radius(centers)
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    def anneal_and_refine(centers, time_limit, initial_temp=0.04, cooling=0.998):
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        it = 0
        
        while time.time() < time_limit and temp > 1e-5:
            it += 1
            n_move = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = np.random.choice(N, n_move, replace=False)
            step_size = temp * 0.4
            new_centers = centers.copy()
            new_centers[indices] += np.random.normal(0, step_size, (n_move, 3))
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            
            new_r = compute_radius(new_centers)
            
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            temp *= cooling
            
            if it % 15 == 0 and time.time() < time_limit - 1:
                refine_limit = min(time.time() + 0.3, time_limit)
                refined_centers, refined_r = force_directed_refine(centers, refine_limit, step_scale=0.3)
                if refined_r > current_r:
                    current_r = refined_r
                    centers = refined_centers
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    def binary_search_polish(centers, time_limit):
        """Binary search on radius with feasibility check."""
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        
        r_low = best_r
        r_high = best_r + 0.005
        
        for _ in range(30):
            if time.time() > time_limit:
                break
            r_mid = (r_low + r_high) / 2
            
            # Try to find a feasible configuration with radius r_mid
            # Use force-directed to try to achieve this radius
            test_centers = best_centers.copy()
            test_r = compute_radius(test_centers)
            
            # If current radius already exceeds r_mid, it's feasible
            if test_r >= r_mid:
                r_low = r_mid
                continue
            
            # Try to improve toward r_mid using force-directed
            fd_limit = min(time.time() + 1.0, time_limit)
            fd_centers, fd_r = force_directed_refine(test_centers, fd_limit, step_scale=0.5)
            
            if fd_r >= r_mid:
                r_low = r_mid
                if fd_r > best_r:
                    best_r = fd_r
                    best_centers = fd_centers
            else:
                r_high = r_mid
        
        return best_centers, best_r

    # --- Main Execution ---
    np.random.seed(42)
    
    best_centers_global = None
    best_radius_global = 0.0
    
    # Generate initial configurations
    initializations = []
    
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    for jitter in [0.0, 0.01, 0.03]:
        centers = grid_centers + np.random.normal(0, jitter, grid_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # BCC lattice
    a = 0.28
    bcc_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                bcc_points.append([i*a, j*a, k*a])
                bcc_points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    
    bcc_array = np.array(bcc_points)
    mask = np.all(np.abs(bcc_array) <= 0.49, axis=1)
    bcc_array = bcc_array[mask]
    
    if len(bcc_array) >= N:
        idx = np.random.choice(len(bcc_array), N, replace=False)
        bcc_centers = bcc_array[idx]
        for jitter in [0.0, 0.02]:
            centers = bcc_centers + np.random.normal(0, jitter, bcc_centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            initializations.append(centers)
    
    # FCC-like lattice
    a_fcc = 0.33
    fcc_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                fcc_points.append([i*a_fcc, j*a_fcc, k*a_fcc])
                fcc_points.append([i*a_fcc + a_fcc/2, j*a_fcc + a_fcc/2, k*a_fcc])
                fcc_points.append([i*a_fcc + a_fcc/2, j*a_fcc, k*a_fcc + a_fcc/2])
                fcc_points.append([i*a_fcc, j*a_fcc + a_fcc/2, k*a_fcc + a_fcc/2])
    
    fcc_array = np.array(fcc_points)
    mask = np.all(np.abs(fcc_array) <= 0.49, axis=1)
    fcc_array = fcc_array[mask]
    
    if len(fcc_array) >= N:
        idx = np.random.choice(len(fcc_array), N, replace=False)
        fcc_centers = fcc_array[idx]
        for jitter in [0.0, 0.02]:
            centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            initializations.append(centers)
    
    # Random
    for _ in range(3):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)

    # Optimization loop
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - 30 - time.time()
        if remaining < 5:
            break
        
        r_init = compute_radius(init_centers)
        
        # SA phase
        anneal_time = min(remaining * 0.35, 12.0)
        anneal_limit = time.time() + anneal_time
        sa_centers, sa_r = anneal_and_refine(init_centers, anneal_limit, initial_temp=0.05, cooling=0.997)
        
        if sa_r > best_radius_global:
            best_radius_global = sa_r
            best_centers_global = sa_centers
        
        # Force-directed refinement
        if time.time() < deadline - 25:
            refine_limit = min(time.time() + 5.0, deadline - 25)
            fd_centers, fd_r = force_directed_refine(sa_centers, refine_limit, step_scale=0.8)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Soft-min polish
        if time.time() < deadline - 20 and best_centers_global is not None:
            polish_limit = min(time.time() + 4.0, deadline - 20)
            polished = polish(best_centers_global.ravel(), polish_limit, beta=80.0, max_iter=200)
            polished_r = compute_radius(polished)
            if polished_r > best_radius_global:
                best_radius_global = polished_r
                best_centers_global = polished

    # Final intensive polish
    if best_centers_global is not None and time.time() < deadline - 10:
        # Force-directed pass
        if time.time() < deadline - 5:
            fd_centers, fd_r = force_directed_refine(best_centers_global, deadline - 5, step_scale=0.3)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Binary search polish
        if time.time() < deadline - 2:
            bs_centers, bs_r = binary_search_polish(best_centers_global, deadline - 2)
            if bs_r > best_radius_global:
                best_radius_global = bs_r
                best_centers_global = bs_centers
        
        # Final high-precision polish
        if time.time() < deadline - 1:
            polish_limit = deadline - 1
            polished = polish(best_centers_global.ravel(), polish_limit, beta=200.0, max_iter=300)
            polished_r = compute_radius(polished)
            if polished_r > best_radius_global:
                best_radius_global = polished_r
                best_centers_global = polished

    if best_centers_global is None:
        return grid_centers

    best_centers_global = np.clip(best_centers_global, -0.5, 0.5)
    return best_centers_global

if __name__ == "__main__":
    start = time.time()
    result = construct()
    end = time.time()
    r = compute_radius(result)
    print(f"Radius: {r}, Time: {end-start}s")
