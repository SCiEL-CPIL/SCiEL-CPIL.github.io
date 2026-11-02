import numpy as np
import time
from scipy.spatial import cKDTree

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

    def soft_min_objective(x, beta=100.0):
        """
        Minimize the soft-min of all pairwise distances and wall distances.
        soft_min(d_1, ..., d_m) = -(1/beta) * log(sum(exp(-beta * d_i)))
        As beta -> inf, this approaches the true min.
        """
        c = x.reshape(N, 3)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        # All distances
        all_dists = np.concatenate([wall_dists.ravel(), pair_dists])
        
        # Soft min
        min_d = np.min(all_dists)
        max_d = np.max(all_dists)
        
        # Shift for numerical stability
        shifted = all_dists - min_d
        exp_vals = np.exp(-beta * shifted)
        log_sum = np.log(np.sum(exp_vals))
        soft_min_val = min_d - (1.0 / beta) * log_sum
        
        # Gradient
        grad = np.zeros_like(c)
        
        # Wall gradient
        # d(soft_min)/dwall_dist_k = -exp(-beta*(wall_k - min_d)) / sum(exp(-beta*(d_i - min_d)))
        # d(wall_k)/dc = -sign(c_k)
        wall_exp = np.exp(-beta * (wall_dists - min_d))
        total_exp = np.sum(exp_vals)
        for k in range(3):
            grad[:, k] += -wall_exp[:, k] / total_exp * (-np.sign(c[:, k]))
        
        # Pair gradient
        # d(soft_min)/dpair_dist_ij = -exp(-beta*(pair_ij - min_d)) / sum(exp(-beta*(d_i - min_d)))
        # d(pair_dist_ij)/dc_i = (c_i - c_j) / pair_dist_ij
        pair_exp = np.exp(-beta * (pair_dists - min_d))
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = pair_dists[idx]
            if d > 1e-12:
                coeff = -pair_exp[idx] / total_exp
                direction = (c[i] - c[j]) / d
                grad[i] += coeff * direction
                grad[j] += -coeff * direction
        
        return soft_min_val, grad.ravel()

    def coordinate_descent_minmax(centers, time_limit, max_iter=500):
        """
        Directly push apart the binding pair and inward wall-violating points.
        This is a structurally different local search.
        """
        centers = centers.copy()
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        
        for it in range(max_iter):
            if time.time() > time_limit:
                break
            
            r = compute_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
            
            # Find binding constraints
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            
            tree = cKDTree(centers)
            dists, _ = tree.query(centers, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
            
            current_r = min(r_wall, r_pair)
            
            # If wall is binding, push spheres inward
            if r_wall <= r_pair * 1.001:
                for k in range(3):
                    mask = wall_dists[:, k] < current_r + 0.005
                    if np.any(mask):
                        viol = (current_r + 0.005) - wall_dists[mask, k]
                        dirs = -np.sign(centers[mask, k])
                        dirs[dirs == 0] = 1.0
                        centers[mask, k] += 0.01 * viol * dirs
                centers = np.clip(centers, -0.4999, 0.4999)
            else:
                # If pair is binding, push apart
                min_pair_idx = np.argmin(dists[:, 1])
                i = min_pair_idx
                j = tree.query(centers[i], k=2)[1][1]
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10:
                    direction = diff / d
                    centers[i] += 0.005 * direction
                    centers[j] -= 0.005 * direction
                centers = np.clip(centers, -0.4999, 0.4999)
            
            # Also handle all near-binding pairs
            near_pairs = tree.query_pairs(current_r + 0.01)
            for pi, pj in near_pairs:
                diff = centers[pi] - centers[pj]
                d = np.linalg.norm(diff)
                if d < 2 * current_r + 0.01 and d > 1e-10:
                    direction = diff / d
                    push = 0.003 * (2 * current_r + 0.01 - d)
                    centers[pi] += push * direction
                    centers[pj] -= push * direction
            centers = np.clip(centers, -0.4999, 0.4999)
            
            # Check for improvement
            new_r = compute_radius(centers)
            if new_r < best_r * 0.999:
                centers = best_centers.copy()
        
        return best_centers, best_r

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        """Force-directed local search."""
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        base_step = 0.03 * step_scale
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
                if stagnation_count > 150:
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
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            new_r = compute_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
            else:
                step *= 0.5
                new_centers = centers + step * force
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                new_r = compute_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                else:
                    centers += np.random.normal(0, 0.003, centers.shape)
                    centers = np.clip(centers, -0.4999, 0.4999)
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
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
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

    def optimize_from_initial(centers, time_limit):
        """Full optimization pipeline from a single initial configuration."""
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        
        # Phase 1: Simulated Annealing
        if time.time() < time_limit - 10:
            anneal_limit = min(time.time() + 8.0, time_limit - 10)
            sa_centers, sa_r = anneal_and_refine(centers, anneal_limit, initial_temp=0.05, cooling=0.997)
            if sa_r > best_r:
                best_r = sa_r
                best_centers = sa_centers
        
        # Phase 2: Force-directed refinement
        if time.time() < time_limit - 8:
            refine_limit = min(time.time() + 3.0, time_limit - 8)
            fd_centers, fd_r = force_directed_refine(best_centers, refine_limit, step_scale=0.8)
            if fd_r > best_r:
                best_r = fd_r
                best_centers = fd_centers
        
        # Phase 3: Coordinate descent minmax
        if time.time() < time_limit - 5:
            cd_limit = min(time.time() + 2.0, time_limit - 5)
            cd_centers, cd_r = coordinate_descent_minmax(best_centers, cd_limit)
            if cd_r > best_r:
                best_r = cd_r
                best_centers = cd_centers
        
        # Phase 4: Final force-directed polish
        if time.time() < time_limit - 2:
            polish_limit = min(time.time() + 1.5, time_limit - 2)
            pl_centers, pl_r = force_directed_refine(best_centers, polish_limit, step_scale=0.3)
            if pl_r > best_r:
                best_r = pl_r
                best_centers = pl_centers
        
        # Phase 5: Final coordinate descent
        if time.time() < time_limit - 0.5:
            cd2_limit = min(time.time() + 0.5, time_limit - 0.5)
            cd2_centers, cd2_r = coordinate_descent_minmax(best_centers, cd2_limit)
            if cd2_r > best_r:
                best_r = cd2_r
                best_centers = cd2_centers
        
        return best_centers, best_r

    def build_initial_grid():
        """5x5x4 grid with jitter."""
        np.random.seed(42)
        nx, ny, nz = 5, 5, 4
        xs = np.linspace(-0.4, 0.4, nx)
        ys = np.linspace(-0.4, 0.4, ny)
        zs = np.linspace(-0.3, 0.3, nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        return grid_centers

    def build_bcc_init():
        """BCC-like lattice initialization."""
        a = 0.32
        bcc_points = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    bcc_points.append([i*a, j*a, k*a])
                    bcc_points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
        
        bcc_array = np.array(bcc_points)
        mask = np.all(np.abs(bcc_array) <= 0.49, axis=1)
        bcc_array = bcc_array[mask]
        
        if len(bcc_array) >= N:
            idx = np.random.choice(len(bcc_array), N, replace=False)
            return bcc_array[idx]
        return build_initial_grid()

    def build_halton_init():
        """Halton sequence initialization."""
        def halton(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f /= base
                result += f * (i % base)
                i //= base
            return result
        
        points = []
        for n in range(1, N + 1):
            x = halton(n, 2) * 0.9 - 0.45
            y = halton(n, 3) * 0.9 - 0.45
            z = halton(n, 5) * 0.9 - 0.45
            points.append([x, y, z])
        
        return np.array(points)

    # --- Main Execution ---
    np.random.seed(42)
    
    best_centers_global = None
    best_radius_global = 0.0
    
    # Generate initial configurations
    initializations = []
    
    grid_centers = build_initial_grid()
    initializations.append(grid_centers)
    initializations.append(grid_centers + np.random.normal(0, 0.02, grid_centers.shape))
    
    bcc_centers = build_bcc_init()
    initializations.append(bcc_centers)
    initializations.append(bcc_centers + np.random.normal(0, 0.02, bcc_centers.shape))
    
    halton_centers = build_halton_init()
    initializations.append(halton_centers)
    initializations.append(halton_centers + np.random.normal(0, 0.03, halton_centers.shape))
    
    # Random initializations
    for _ in range(2):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
    
    # Optimization loop
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break
        
        opt_limit = time.time() + min(remaining * 0.85, 12.0)
        opt_centers, opt_r = optimize_from_initial(init_centers, opt_limit)
        
        if opt_r > best_radius_global:
            best_radius_global = opt_r
            best_centers_global = opt_centers
    
    # Final intensive polish
    if best_centers_global is not None and time.time() < deadline - 3:
        # Force-directed pass
        if time.time() < deadline - 2:
            fd_centers, fd_r = force_directed_refine(best_centers_global, deadline - 2, step_scale=0.3)
            if fd_r > best_radius_global:
                best_radius_global = fd_r
                best_centers_global = fd_centers
        
        # Final coordinate descent
        if time.time() < deadline - 0.5:
            cd_centers, cd_r = coordinate_descent_minmax(best_centers_global, deadline - 0.5)
            if cd_r > best_radius_global:
                best_radius_global = cd_r
                best_centers_global = cd_centers
    
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
