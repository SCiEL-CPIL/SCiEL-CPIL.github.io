import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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
        """
        Minimize the soft-min of all pairwise distances and wall distances.
        The soft-min is a smooth approximation to min(d_ij, wall_i).
        We maximize this, so we minimize -soft_min.
        
        Actually, we want to maximize r = min(min_pair_dist/2, min_wall_dist).
        Let's define:
          d_ij = |c_i - c_j|
          w_i = 0.5 - |c_i| (component-wise, but we take min over components)
          Actually w_i = min_k (0.5 - |c_{ik}|)
        
        r = min( min_{i<j} d_ij / 2, min_i w_i )
        
        Let's work with:
          a_ij = d_ij / 2  (for i < j)
          b_i = w_i
        
        r = min( min a_ij, min b_i )
        
        Soft min of a set of values {v_1, ..., v_M}:
          soft_min(v) = -1/beta * log( sum_k exp(-beta * v_k) )
        
        As beta -> inf, soft_min -> min.
        
        We want to maximize r, so minimize -soft_min.
        """
        c = x.reshape(N, 3)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)  # shape (N, 3)
        b = np.min(wall_dists, axis=1)  # shape (N,)
        
        # Pairwise distances / 2
        # Use cKDTree for efficiency in gradient? No, we need gradient.
        # For 100 points, 4950 pairs. Let's compute all.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        a = dists / 2.0  # shape (N, N), with inf on diagonal
        
        # Collect all values
        a_vals = a[np.triu_indices(N, k=1)]  # 4950 values
        b_vals = b  # 100 values
        
        all_vals = np.concatenate([a_vals, b_vals])
        
        # Soft min
        beta = 50.0
        # For numerical stability
        min_val = np.min(all_vals)
        exp_terms = np.exp(-beta * (all_vals - min_val))
        soft_min_val = -min_val - (1.0 / beta) * np.log(np.sum(exp_terms))
        
        # Objective: minimize -soft_min_val
        obj = -soft_min_val
        
        # Gradient
        # d(soft_min)/d v_k = exp(-beta * (v_k - min_val)) / sum(exp(-beta * (v_j - min_val)))
        # = exp_terms[k] / sum(exp_terms)
        # So d(obj)/d v_k = -exp_terms[k] / sum(exp_terms)
        
        grad_all = -exp_terms / np.sum(exp_terms)
        
        grad_a = grad_all[:len(a_vals)]
        grad_b = grad_all[len(a_vals):]
        
        grad = np.zeros_like(c)
        
        # Gradient from b (wall distances)
        # b_i = min_k (0.5 - |c_{ik}|)
        # This is non-smooth. Use the component that achieves the min.
        # For soft-min gradient, we need d b_i / d c_i.
        # b_i = min_k w_{ik}, where w_{ik} = 0.5 - |c_{ik}|
        # d w_{ik} / d c_{ik} = -sign(c_{ik})
        # d b_i / d c_{ik} = -sign(c_{ik}) if k is the argmin, else 0 (for hard min)
        # For soft-min, we use a weighted average. But b_i is already a min, not a soft-min.
        # Hmm, this is getting complicated. Let's use a simpler approach for the gradient.
        
        # Actually, let's just use numerical gradient or a simpler objective.
        # Alternative: use the penalty method approach but with a smoother objective.
        
        # Let's try a different approach: minimize the sum of squared violations
        # for a target radius, and use binary search.
        
        # For now, return a placeholder gradient (zeros) and we'll use a different optimizer.
        # Actually, let's compute the gradient properly.
        
        # For b_i = min_k (0.5 - |c_{ik}|):
        # Let k_i be the index of the minimum component for sphere i.
        # Then d b_i / d c_{i, k_i} = -sign(c_{i, k_i})
        # And d obj / d c_{i, k_i} = grad_b[i] * (-sign(c_{i, k_i}))
        
        for i in range(N):
            # Find the component that achieves the min wall distance
            min_comp = np.argmin(wall_dists[i])
            if wall_dists[i, min_comp] < 0.5 - 1e-10:  # Only if it's actually the min
                grad[i, min_comp] += grad_b[i] * (-np.sign(c[i, min_comp]))
                if c[i, min_comp] == 0:
                    grad[i, min_comp] += grad_b[i] * 1.0  # Arbitrary direction
        
        # Gradient from a (pairwise distances)
        # a_ij = |c_i - c_j| / 2
        # d a_ij / d c_i = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (2 * 2 * a_ij) = (c_i - c_j) / (4 * a_ij)
        # Wait: d |c_i - c_j| / d c_i = (c_i - c_j) / |c_i - c_j|
        # So d a_ij / d c_i = (c_i - c_j) / (2 * |c_i - c_j|)
        
        # grad_a is indexed by triu indices
        iu = np.triu_indices(N, k=1)
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = dists[i, j]
            if d > 1e-12:
                direction = (c[i] - c[j]) / d
                # d a_ij / d c_i = direction / 2
                # d a_ij / d c_j = -direction / 2
                grad[i] += grad_a[idx] * direction / 2.0
                grad[j] += grad_a[idx] * (-direction / 2.0)
        
        return obj, grad.ravel()

    def polish(x0, time_limit, beta=100.0, max_iter=500):
        """Polish using L-BFGS-B on the soft-min objective."""
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
                options={'maxiter': max_iter, 'maxfun': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            return new_centers
        except Exception:
            return x0.reshape(N, 3)

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        """Force-directed local search."""
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        base_step = 0.05 * step_scale
        decay_rate = 0.999
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
                if stagnation_count > 100:
                    break
            else:
                stagnation_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            force = np.zeros_like(centers)
            target_r = current_r + 0.002
            
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
                    centers += np.random.normal(0, 0.002, centers.shape)
                    centers = np.clip(centers, -0.499999, 0.499999)
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
            
            if it % 20 == 0 and time.time() < time_limit - 1:
                refine_limit = min(time.time() + 0.3, time_limit)
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

    # Optimization loop
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 25:
            break
        
        remaining = deadline - 25 - time.time()
        if remaining < 5:
            break
        
        r_init = compute_radius(init_centers)
        
        # Short SA phase
        anneal_time = min(remaining * 0.3, 10.0)
        anneal_limit = time.time() + anneal_time
        sa_centers, sa_r = anneal_and_refine(init_centers, anneal_limit, initial_temp=0.05, cooling=0.997)
        
        if sa_r > best_radius_global:
            best_radius_global = sa_r
            best_centers_global = sa_centers
        
        # Force-directed refinement
        if time.time() < deadline - 20:
            refine_limit = min(time.time() + 4.0, deadline - 20)
            fd_centers, fd_r = force_directed_refine(sa_centers, refine_limit, step_scale=0.8)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Soft-min polish
        if time.time() < deadline - 15 and best_centers_global is not None:
            polish_limit = min(time.time() + 5.0, deadline - 15)
            polished = polish(best_centers_global.ravel(), polish_limit, beta=50.0, max_iter=200)
            polished_r = compute_radius(polished)
            if polished_r > best_radius_global:
                best_radius_global = polished_r
                best_centers_global = polished

    # Final intensive polish
    if best_centers_global is not None and time.time() < deadline - 5:
        # Force-directed pass
        if time.time() < deadline - 3:
            fd_centers, fd_r = force_directed_refine(best_centers_global, deadline - 3, step_scale=0.3)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Final high-precision polish
        if time.time() < deadline - 1:
            polish_limit = deadline - 1
            polished = polish(best_centers_global.ravel(), polish_limit, beta=200.0, max_iter=500)
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
