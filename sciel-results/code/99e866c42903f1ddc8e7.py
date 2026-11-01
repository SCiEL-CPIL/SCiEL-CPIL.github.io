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

    def soft_min_objective(x_flat, beta=100.0):
        """
        Smooth surrogate: maximize soft-min of all constraints.
        Returns (objective, gradient) where objective = -soft_min (to minimize).
        """
        c = x_flat.reshape(N, 3)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)  # (N, 3)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        pair_dists = dists / 2.0  # (N, N)
        
        # Collect all constraint values
        # Wall: 300 values, Pair: 4950 values (unique)
        iu = np.triu_indices(N, k=1)
        neg_wall = -wall_dists.ravel()
        neg_pair = -pair_dists[iu]
        all_neg = np.concatenate([neg_wall, neg_pair])
        
        # Soft-min via log-sum-exp
        max_neg = np.max(all_neg)
        exps = np.exp((all_neg - max_neg) / beta)
        sum_exps = np.sum(exps)
        logsum = max_neg + beta * np.log(sum_exps)
        soft_min_val = -logsum / beta
        
        # Weights for gradient
        weights = exps / sum_exps
        
        # Gradient computation
        grad = np.zeros_like(c)
        
        # Wall gradient: d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
        w_wall = weights[:N*3].reshape(N, 3)
        grad -= w_wall * np.sign(c)
        
        # Pair gradient
        w_pair = weights[N*3:]
        # Vectorized pair gradient accumulation
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = dists[i, j]
            if d > 1e-12:
                direction = (c[i] - c[j]) / d
                grad[i] += w_pair[idx] * direction / 2.0
                grad[j] -= w_pair[idx] * direction / 2.0
        
        obj_val = -soft_min_val
        grad_obj = -grad
        
        return obj_val, grad_obj.ravel()

    def lbfgs_polish(centers, time_limit, beta=100.0, max_iter=500):
        """L-BFGS-B polish with smooth surrogate."""
        centers = centers.copy()
        x0 = centers.ravel()
        
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
                bounds=[(-0.49999, 0.49999)] * (N * 3),
                options={'maxiter': max_iter, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.49999, 0.49999)
            new_r = compute_radius(new_centers)
            if new_r > compute_radius(centers):
                return new_centers, new_r
        except Exception:
            pass
        
        return centers, compute_radius(centers)

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        """Force-directed local search."""
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        base_step = 0.05 * step_scale
        decay_rate = 0.999
        stagnation_count = 0
        
        while time.time() < time_limit:
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
                if stagnation_count > 50:
                    break
            else:
                stagnation_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            force = np.zeros_like(centers)
            target_r = current_r + 0.003
            
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                mask = d_wall < target_r
                if np.any(mask):
                    viol = target_r - d_wall[mask]
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1.0
                    force[mask, k] += viol * dirs
            
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
            new_centers = np.clip(new_centers, -0.49999, 0.49999)
            new_r = compute_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
            else:
                step *= 0.5
                new_centers = centers + step * force
                new_centers = np.clip(new_centers, -0.49999, 0.49999)
                new_r = compute_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                else:
                    centers += np.random.normal(0, 0.002, centers.shape)
                    centers = np.clip(centers, -0.49999, 0.49999)
                    r = compute_radius(centers)
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    def anneal_and_refine(centers, time_limit, initial_temp=0.04, cooling=0.998):
        """Simulated Annealing with periodic force-directed refinement."""
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
            new_centers = np.clip(new_centers, -0.49999, 0.49999)
            
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
                refine_limit = min(time.time() + 0.2, time_limit)
                refined_centers, refined_r = force_directed_refine(centers, refine_limit, step_scale=0.3)
                if refined_r > current_r:
                    current_r = refined_r
                    centers = refined_centers
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
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
    
    # FCC-like lattice
    a = 0.33
    fcc_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                fcc_points.append([i*a, j*a, k*a])
                fcc_points.append([i*a + a/2, j*a + a/2, k*a])
                fcc_points.append([i*a + a/2, j*a, k*a + a/2])
                fcc_points.append([i*a, j*a + a/2, k*a + a/2])
    
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

    # Optimization loop with emphasis on L-BFGS polish
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        
        r_init = compute_radius(init_centers)
        
        # Short SA phase
        anneal_time = min(remaining * 0.3, 8.0)
        anneal_limit = time.time() + anneal_time
        sa_centers, sa_r = anneal_and_refine(init_centers, anneal_limit, initial_temp=0.05, cooling=0.997)
        
        if sa_r > best_radius_global:
            best_radius_global = sa_r
            best_centers_global = sa_centers
        
        # Force-directed refinement
        if time.time() < deadline - 15:
            refine_limit = min(time.time() + 3.0, deadline - 15)
            fd_centers, fd_r = force_directed_refine(sa_centers, refine_limit, step_scale=0.8)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # L-BFGS polish - key improvement
        if time.time() < deadline - 10 and best_centers_global is not None:
            polish_limit = min(time.time() + 5.0, deadline - 10)
            pol_centers, pol_r = lbfgs_polish(best_centers_global, polish_limit, beta=80.0, max_iter=200)
            if pol_r > best_radius_global:
                best_radius_global = pol_r
                best_centers_global = pol_centers

    # Final intensive polish
    if best_centers_global is not None and time.time() < deadline - 5:
        # High-precision L-BFGS with tighter beta
        polish_limit = deadline - 5
        pol_centers, pol_r = lbfgs_polish(best_centers_global, polish_limit, beta=150.0, max_iter=400)
        if pol_r > best_radius_global:
            best_radius_global = pol_r
            best_centers_global = pol_centers
        
        # One more force-directed pass
        if time.time() < deadline - 2:
            fd_centers, fd_r = force_directed_refine(best_centers_global, deadline - 2, step_scale=0.3)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Final ultra-precise L-BFGS
        if time.time() < deadline - 1:
            polish_limit = deadline - 1
            pol_centers, pol_r = lbfgs_polish(best_centers_global, polish_limit, beta=200.0, max_iter=200)
            if pol_r > best_radius_global:
                best_radius_global = pol_r
                best_centers_global = pol_centers

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
