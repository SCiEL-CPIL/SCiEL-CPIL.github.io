import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        if len(centers) < 2:
            return min_wall
            
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def soft_min_energy(centers, beta):
        """
        Compute a smooth approximation of the negative radius.
        We want to minimize the energy, which corresponds to maximizing the radius.
        Energy = - soft_min(min_wall, min_pair/2)
        Using log-sum-exp for stability.
        """
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        
        # Pair distances
        # To avoid O(N^2) memory blowup for large N, we might need to be careful, 
        # but N=200 is small enough for full distance matrix (40000 entries).
        # However, for gradient computation, we need the full matrix or efficient accumulation.
        # Let's compute all pair distances.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        
        # Exclude self-pairs
        np.fill_diagonal(dists, np.inf)
        
        # We want min of (wall_dists, dists/2)
        # Let's combine them into a single vector of "constraints"
        # Constraint values: c_k
        # We want to maximize min(c_k).
        # Soft min: -log( sum(exp(-beta * c_k)) ) / beta
        # This is a convex function of c_k if beta > 0? No, -log-sum-exp is concave.
        # Wait, we want to MINIMIZE the energy.
        # If we define Energy = - soft_min(c), then minimizing Energy is maximizing soft_min(c).
        # soft_min(c) = -1/beta * log( sum(exp(-beta * c)) )
        # This is a smooth lower bound on min(c).
        
        # To avoid overflow, we use the log-sum-exp trick.
        # Let z = -beta * c.
        # log_sum_exp(z) = max(z) + log(sum(exp(z - max(z))))
        
        # Combine wall and pair constraints
        # Wall constraints: wall_dists (shape N)
        # Pair constraints: dists/2 (shape N*N, but symmetric and diagonal inf)
        # Let's just use the unique pairs to save memory/time slightly, or just all.
        # N=200 -> 200*200 = 40000. 40000 exps is fine.
        
        all_constraints = np.concatenate([wall_dists, dists/2.0])
        
        # Filter out inf
        valid_mask = np.isfinite(all_constraints)
        if not np.all(valid_mask):
            all_constraints = all_constraints[valid_mask]
            
        if len(all_constraints) == 0:
            return 0.0, np.zeros(2*N)

        # z = -beta * c
        z = -beta * all_constraints
        z_max = np.max(z)
        # If z_max is very large (negative c), exp might overflow? 
        # c should be positive if inside square.
        # exp(z - z_max) is <= 1.
        
        log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
        
        energy = log_sum_exp / beta
        
        # Gradient calculation
        # d(energy)/d(c_k) = - (1/beta) * d(log_sum_exp)/d(c_k)
        # d(log_sum_exp)/d(c_k) = d/d(c_k) [ -beta * c_max - log(sum(exp(-beta(c_j - c_max)))) ] ... wait
        # Let's use the standard formula for soft min gradient.
        # soft_min(c) = -1/beta * log( sum(exp(-beta * c_k)) )
        # grad_c soft_min = ( exp(-beta * c_k) / sum(exp(-beta * c_j)) )
        
        # We need grad_centers.
        # c_k depends on centers.
        # This is complex to compute analytically for all pairs.
        # Let's use finite differences for the gradient? 
        # No, that's too slow for L-BFGS-B (needs many iterations).
        # We need analytic gradient.
        
        # Let's restructure.
        # E = - soft_min(c)
        # dE/dc = - grad_c soft_min
        # grad_c soft_min is a vector of weights w_k = exp(-beta * c_k) / Z
        # where Z = sum(exp(-beta * c_j))
        
        # We need dE/d(centers) = sum_k (dE/dc_k) * (dc_k/d(centers))
        # dc_k/d(centers) is sparse.
        
        # This analytic gradient is tedious to implement correctly and efficiently.
        # Alternative: Use a simpler smooth approximation?
        # Or use `scipy.optimize.minimize` with method 'L-BFGS-B' and provide analytic gradient.
        
        # Let's try to compute the gradient efficiently.
        # Z = sum(exp(-beta * c_j))
        # w_j = exp(-beta * c_j) / Z
        
        # For wall constraint k (point i, wall w):
        # c_k = 0.5 - |center_i,coord|
        # d c_k / d center_i,coord = - sign(center_i,coord)
        
        # For pair constraint k (points i, j):
        # c_k = ||center_i - center_j|| / 2
        # d c_k / d center_i = (center_i - center_j) / (2 * ||center_i - center_j||)
        # d c_k / d center_j = (center_j - center_i) / (2 * ||center_i - center_j||)
        
        # This requires iterating over all pairs. N=200, 20k pairs.
        # In Python, this loop might be slow.
        # Can we vectorize?
        
        # Let's try a different approach:
        # Use a "push-apart" force model which is effectively the gradient of a repulsive potential.
        # Potential V = sum_{i<j} 1/d_{ij}^p + sum_i 1/d_{wall,i}^p
        # Minimizing V pushes points apart.
        # But this doesn't directly maximize the MIN distance. It maximizes the SUM of distances (roughly).
        # However, for packing, maximizing the sum often correlates with maximizing the min, especially if we use a high power p or a soft-min.
        
        # Let's stick to the soft-min energy but implement the gradient carefully.
        # Actually, for N=200, a pure Python loop for gradient might be too slow if L-BFGS-B takes 1000+ iterations.
        # 1000 iterations * 20k pairs * Python overhead = too slow.
        
        # Alternative: Use `minimize` with `method='L-BFGS-B'` and `jac` provided.
        # We need to vectorize the gradient accumulation.
        
        # Let's try a different smooth objective:
        # Maximize r such that all constraints >= r.
        # This is equivalent to minimizing max(0, r - c_k).
        # Smooth approximation: sum( softplus(beta * (r - c_k)) )
        # This is a convex function of r and centers.
        # We can optimize over r and centers.
        # Variable x = [centers, r]
        
        def objective_and_grad(x):
            centers = x[:2*N].reshape(N, 2)
            r = x[2*N]
            
            # Wall distances
            wall_dists = 0.5 - np.abs(centers)
            
            # Pair distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, np.inf)
            pair_dists = dists / 2.0
            
            # Constraints: c_k >= r
            # Violation: max(0, r - c_k)
            # Smooth: softplus(beta * (r - c_k))
            # softplus(z) = log(1 + exp(z))
            
            # We want to minimize sum(softplus(beta * (r - c_k)))
            # Plus a penalty if r is negative? No, r should be positive.
            # Also, we want to encourage large r.
            # The softplus term is 0 when r <= c_k.
            # If we just minimize this, r will go to -infinity.
            # So we need a term that pushes r up.
            # Or, we can fix r and check feasibility? No, that's not smooth.
            
            # Better: Minimize sum(softplus(beta * (r - c_k))) - gamma * r
            # If gamma is large, it pushes r up until constraints are violated.
            
            beta = 50.0
            gamma = 1.0
            
            # Wall violations
            v_wall = r - wall_dists
            # Pair violations
            v_pair = r - pair_dists
            
            # softplus
            # To avoid overflow in exp, use stable softplus
            def stable_softplus(z):
                return np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))
                
            E_wall = np.sum(stable_softplus(beta * v_wall))
            E_pair = np.sum(stable_softplus(beta * v_pair))
            
            energy = E_wall + E_pair - gamma * r
            
            # Gradient
            # d/dx softplus(beta * (r - c)) = beta * sigmoid(beta * (r - c)) * d(r-c)/dx
            # sigmoid(z) = 1 / (1 + exp(-z))
            
            def stable_sigmoid(z):
                # stable sigmoid
                return np.where(z > 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
                
            sig_wall = stable_sigmoid(beta * v_wall)
            sig_pair = stable_sigmoid(beta * v_pair)
            
            # d energy / d r
            # d E_wall / dr = sum(beta * sig_wall)
            # d E_pair / dr = sum(beta * sig_pair)
            # d (-gamma * r) / dr = -gamma
            grad_r = beta * (np.sum(sig_wall) + np.sum(sig_pair)) - gamma
            
            # d energy / d centers
            # d E_wall / d center_i = sum_k (beta * sig_wall[k]) * d(v_wall[k])/d(center_i)
            # v_wall[k] = r - (0.5 - |center_i,coord|)
            # d v_wall / d center = - d(0.5 - |center|)/d center = sign(center)
            # Wait, wall_dists = 0.5 - abs(centers).
            # d wall_dists / d center = -sign(centers)
            # d v_wall / d center = - d wall_dists / d center = sign(centers)
            
            # So grad_centers_wall = - beta * sig_wall * sign(centers)
            # Note: sig_wall is shape (N,), centers is (N,2).
            # We need to broadcast sig_wall to (N,2) or multiply component-wise.
            # v_wall[i] corresponds to point i.
            # d v_wall[i] / d center_i = sign(center_i)
            
            grad_centers = np.zeros_like(centers)
            
            # Wall contribution
            # sig_wall[i] is scalar for point i.
            # grad_centers[i] += -beta * sig_wall[i] * sign(centers[i])
            # sign(0) is 0.
            signs = np.sign(centers)
            grad_centers += -beta * sig_wall[:, np.newaxis] * signs
            
            # Pair contribution
            # v_pair[i,j] = r - dist(i,j)/2
            # d v_pair[i,j] / d center_i = - d(dist(i,j)/2) / d center_i
            # d dist / d center_i = (center_i - center_j) / dist
            # So d v_pair / d center_i = - (center_i - center_j) / (2 * dist)
            
            # We need to sum over all j for each i.
            # grad_centers[i] += sum_j (-beta * sig_pair[i,j] * d v_pair[i,j] / d center_i)
            # = sum_j (-beta * sig_pair[i,j] * (-(center_i - center_j) / (2 * dist[i,j])))
            # = sum_j (beta * sig_pair[i,j] * (center_i - center_j) / (2 * dist[i,j]))
            
            # Vectorized:
            # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (N, N, 2)
            # dists = np.linalg.norm(diff, axis=-1) # (N, N)
            # np.fill_diagonal(dists, 1.0) # avoid div by 0, will be masked by sig_pair (which is 0 or small for diagonal? No, diagonal is inf, so v_pair is -inf, sig is 0)
            # Actually, if dist is inf, v_pair is -inf, sig is 0.
            # But we filled diagonal with inf.
            
            # Let's compute the term: (center_i - center_j) / (2 * dist[i,j])
            # This is the unit vector from j to i, scaled by 0.5.
            # Let U = diff / (2 * dists[:, :, np.newaxis])
            # U[i,j] = (center_i - center_j) / (2 * dist[i,j])
            
            # Then grad_centers[i] += sum_j (beta * sig_pair[i,j] * U[i,j])
            
            # We can do this with matrix multiplication?
            # sig_pair is (N, N). U is (N, N, 2).
            # We want for each i: sum_j (sig_pair[i,j] * U[i,j,0]) and sum_j (sig_pair[i,j] * U[i,j,1])
            
            # grad_x = np.sum(sig_pair[:, :, np.newaxis] * U, axis=1)
            # grad_centers += beta * grad_x
            
            # Let's verify dimensions.
            # sig_pair: (N, N)
            # U: (N, N, 2)
            # product: (N, N, 2)
            # sum axis=1: (N, 2)
            
            # One issue: dists diagonal is inf. 1/inf = 0. So U diagonal is 0.
            # sig_pair diagonal: v_pair = r - inf = -inf. sig = 0.
            # So diagonal contribution is 0.
            
            # However, computing U for all pairs is O(N^2) memory and time.
            # N=200, 40000 entries. This is fine.
            
            # Let's implement this.
            
            # Re-calculate diff and dists for gradient (or reuse from energy calc? 
            # In objective_and_grad, we calculated them. But we need them here.
            # We can calculate them once.
            
            # Note: The `diff` and `dists` calculated in energy part are local to that scope if not returned.
            # Let's recalculate or return them. Recalculating is safer.
            
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            # Avoid division by zero for diagonal (though sig is 0 there)
            # Add epsilon to diagonal
            dists_safe = dists.copy()
            np.fill_diagonal(dists_safe, 1.0)
            
            U = diff / (2.0 * dists_safe[:, :, np.newaxis])
            
            # Weighted sum
            grad_pair = np.sum(sig_pair[:, :, np.newaxis] * U, axis=1)
            
            grad_centers += beta * grad_pair
            
            grad = np.concatenate([grad_centers.flatten(), [grad_r]])
            
            return energy, grad
        
        return objective_and_grad

    def generate_hex_grid(n_rows, n_cols):
        d = 1.0 / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        elif len(pts) < N:
            return None
        return pts

    def local_search_polish(centers, max_rounds=20):
        """High-precision polish using direct local moves."""
        centers = centers.copy()
        current_r = compute_radius(centers)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            tree = cKDTree(centers)
            
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                orig = centers[i].copy()
                best_pos = orig
                best_r = current_r
                
                # Find nearest neighbors
                dists, idxs = tree.query(orig, k=10)
                
                step = 0.001 / (round_num + 1)
                directions = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)
                ]
                directions = [np.array(d) / np.linalg.norm(d) for d in directions]
                for _ in range(8):
                    angle = np.random.rand() * 2 * np.pi
                    directions.append(np.array([math.cos(angle), math.sin(angle)]))
                
                for d in directions:
                    for step_mult in [0.5, 1.0, 2.0, 4.0]:
                        new_center = orig + d * step * step_mult
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                            
                        centers[i] = new_center
                        new_r = compute_radius(centers)
                        
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos = new_center.copy()
                            improved = True
                            
                centers[i] = best_pos
                
            current_r = best_r
            if not improved:
                break
                
        return centers, current_r

    best_centers = None
    best_radius = 0.0

    # Generate diverse initial configurations
    configs = []
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = generate_hex_grid(n_rows, n_cols)
        if pts is not None:
            configs.append(pts)
            
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    configs.append(grid[:N])
    
    # Add jittered hex
    if configs:
        jittered = configs[0].copy()
        jittered += np.random.randn(N, 2) * 0.01
        jittered = np.clip(jittered, -0.49, 0.49)
        configs.append(jittered)

    # Optimize each configuration
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 30:
            break
            
        r_init = compute_radius(init_pts)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init_pts.copy()
            
        if time.time() < deadline - 25:
            try:
                # Flatten centers and add r
                # Initial r estimate
                r_guess = r_init * 0.9
                x0 = np.concatenate([init_pts.flatten(), [r_guess]])
                
                # Bounds for centers
                bounds = [(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)]
                
                # Run L-BFGS-B
                res = minimize(
                    objective_and_grad, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=True, 
                    bounds=bounds, 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                
                if res.success or res.fun < objective_and_grad(x0)[0]:
                    new_centers = res.x[:2*N].reshape(N, 2)
                    new_r = compute_radius(new_centers)
                    
                    if new_r > best_radius:
                        best_radius = new_r
                        best_centers = new_centers.copy()
                        
            except Exception as e:
                pass

        # Final polish on best so far
        if best_centers is not None and time.time() < deadline - 10:
            polished, r_polished = local_search_polish(best_centers, max_rounds=10)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()

    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 2:
        polished, r_polished = local_search_polish(best_centers, max_rounds=20)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished.copy()

    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
