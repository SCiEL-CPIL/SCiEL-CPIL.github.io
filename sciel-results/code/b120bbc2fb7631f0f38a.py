import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    max_time = 110.0  # Safety margin
    
    # Function to compute the exact score
    def compute_score(points):
        # points: (N, 2)
        # Pairwise distances
        dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Wall distances
        radii = np.linalg.norm(points, axis=1)
        min_wall = np.min(1.0 - radii)
        
        return min(min_pair, min_wall)

    # Soft-min objective and gradient for L-BFGS-B
    # We maximize F(x) = -1/beta * log(sum(exp(-beta * v_k)))
    # This is equivalent to minimizing G(x) = 1/beta * log(sum(exp(-beta * v_k)))
    def softmin_obj_grad(x, beta):
        points = x.reshape(N, 2)
        
        # Calculate all distances v_k
        # v_k are: pairwise distances / 2, and wall distances (1 - norm)
        
        # Pairwise
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 0) # Will be masked or ignored in sum if we handle it, but let's just compute valid ones
        
        # To be efficient and avoid memory blowup for N=50 (which is small), we can just compute
        # But for gradient, we need to sum over all pairs.
        
        # Let's compute the sum of exp(-beta * v)
        # v_pairs = dists / 2
        # v_walls = 1 - norm(points)
        
        # To avoid underflow/overflow issues with large beta, we can shift by max v?
        # No, v are small (order 0.1). beta is up to ~100. beta*v ~ 10. exp(-10) is fine.
        # If v is negative (outside circle), exp(-beta * neg) = exp(pos), can be huge.
        # But we want to penalize that.
        
        # Let's compute weights w_k = exp(-beta * v_k)
        
        # Pairwise
        v_pairs = dists / 2.0
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        v_pairs = v_pairs[mask]
        
        # Wall
        radii = np.linalg.norm(points, axis=1)
        v_walls = 1.0 - radii
        
        # Combine all v_k
        # Total sum S = sum(exp(-beta * v_pairs)) + sum(exp(-beta * v_walls))
        
        # For numerical stability with large beta, we can use the log-sum-exp trick if needed,
        # but for N=50, direct sum is likely fine unless beta is huge.
        
        # Calculate objective
        # G = (1/beta) * log(S)
        # We want to minimize G, which is maximizing F = -G
        
        # Gradient of G with respect to x_m:
        # dG/dx_m = (1/beta) * (1/S) * dS/dx_m
        # dS/dx_m = sum_k exp(-beta v_k) * (-beta) * dv_k/dx_m
        # So dG/dx_m = - (1/S) * sum_k exp(-beta v_k) * dv_k/dx_m
        
        # Let's compute S and the weighted sum of gradients.
        
        # Compute exp terms
        # To prevent overflow if v_walls is very negative, we can clip v_walls or handle carefully.
        # However, if points are far outside, the optimizer should push them in.
        
        # Let's compute the sum of weights
        # Weights for pairs
        w_pairs = np.exp(-beta * v_pairs)
        # Weights for walls
        w_walls = np.exp(-beta * v_walls)
        
        S = np.sum(w_pairs) + np.sum(w_walls)
        
        # If S is 0 (underflow), return 0 grad
        if S == 0:
            return 0.0, np.zeros_like(x)
        
        obj_val = (1.0 / beta) * np.log(S)
        
        # Gradient
        grad = np.zeros_like(x)
        
        # Contribution from pairs
        # For each pair (i,j), v_ij = 0.5 * ||p_i - p_j||
        # dv_ij/dp_i = 0.5 * (p_i - p_j) / ||p_i - p_j||
        # dv_ij/dp_j = 0.5 * (p_j - p_i) / ||p_i - p_j||
        
        # We need to sum w_ij * dv_ij/dp_i for all j!=i for each i.
        
        # Efficient calculation:
        # grad_pair = 0
        # for i in range(N):
        #   for j in range(N):
        #     if i != j:
        #       w = exp(-beta * 0.5 * ||p_i - p_j||)
        #       d = p_i - p_j
        #       norm_d = ||d||
        #       if norm_d > eps:
        #         grad_pair[i] += w * 0.5 * d / norm_d
        #         grad_pair[j] += w * 0.5 * (-d) / norm_d  # or w * 0.5 * (p_j - p_i) / norm_d
        
        # Vectorized approach:
        # dists is NxN
        # diff is NxNx2
        # w_pairs_matrix = exp(-beta * dists / 2)
        # Set diagonal to 0
        
        w_pairs_mat = np.exp(-beta * dists / 2.0)
        np.fill_diagonal(w_pairs_mat, 0)
        
        # Gradient contribution from pair i,j to point i:
        # sum_j w_ij * 0.5 * (p_i - p_j) / ||p_i - p_j||
        
        # We can compute this using matrix operations.
        # Let D be dists matrix.
        # Let Diff be diff matrix (i,j,k) -> p_i[k] - p_j[k]
        # Term for i: sum_j w_ij * Diff[i,j] / (2 * D[i,j])
        
        # Avoid division by zero
        # Where D is 0 (diagonal), w is 0, so term is 0.
        # Where D is very small, we might have issues.
        
        # Let's use a safe division
        # grad_pair[i] = sum_j w_ij * Diff[i,j] / (2 * D[i,j])
        
        # We can compute this as:
        # factor = w_pairs_mat / (2 * dists)
        # factor diagonal is 0/0 -> nan. Set to 0.
        # np.fill_diagonal(factor, 0)
        # grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Let's do it carefully
        factor = np.zeros_like(dists)
        # Only compute where dists > 1e-12
        valid = dists > 1e-12
        factor[valid] = w_pairs_mat[valid] / (2.0 * dists[valid])
        
        # grad_pair[i] = sum_j factor[i,j] * (p_i - p_j)
        # diff[i,j] = p_i - p_j
        grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Contribution from walls
        # v_i = 1 - ||p_i||
        # dv_i/dp_i = -p_i / ||p_i||
        # grad_wall[i] = - w_walls[i] * p_i / ||p_i||
        
        radii_safe = np.where(radii > 1e-12, radii, 1e-12)
        grad_wall = - (w_walls / radii_safe)[:, np.newaxis] * points
        
        # Total gradient of G
        # dG/dx = - (1/S) * (grad_pair + grad_wall)
        grad = - (1.0 / S) * (grad_pair + grad_wall)
        
        return obj_val, grad

    best_score = -np.inf
    best_points = None
    
    # Initializations
    # 1. Jittered Hexagonal
    # 2. Rings
    # 3. Random
    
    def init_rings():
        points = []
        # Center
        points.append([0.0, 0.0])
        # Rings
        r = 0.1
        count = 1
        while count < N:
            # How many points in this ring?
            # Approx 6 * ring_index
            ring_idx = len(points) // 6 if len(points) > 1 else 1
            # Let's just add points in a circle
            n_points = 6 * (len(points) // 6 + 1)
            # Adjust to not exceed N
            n_points = min(n_points, N - len(points))
            if n_points <= 0: break
            
            angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            # Add some jitter
            jitter = np.random.uniform(-0.01, 0.01, (n_points, 2))
            ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)]) + jitter
            points.extend(ring_pts.tolist())
            r += 0.1
        points = np.array(points[:N])
        return points

    def init_random():
        return np.random.uniform(-0.8, 0.8, (N, 2))

    def init_hex():
        # Simple hex grid
        pts = []
        s = 0.15
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                if x**2 + y**2 < 0.9:
                    pts.append([x, y])
        # If not enough, add random
        while len(pts) < N:
            pts.append(np.random.uniform(-0.8, 0.8, 2))
        return np.array(pts[:N])

    initializers = [init_rings, init_random, init_hex]
    
    # Beta schedule
    betas = [5, 10, 20, 40, 80]
    
    restarts_per_beta = 3
    
    for beta in betas:
        for _ in range(restarts_per_beta):
            if time.time() - start_time > max_time:
                break
            
            # Choose initializer
            init_func = initializers[np.random.randint(len(initializers))]
            x0 = init_func().flatten()
            
            # Optimization
            try:
                res = minimize(
                    softmin_obj_grad,
                    x0,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x_opt = res.x
                points_opt = x_opt.reshape(N, 2)
                
                score = compute_score(points_opt)
                if score > best_score:
                    best_score = score
                    best_points = points_opt
            except Exception as e:
                pass
        
        if time.time() - start_time > max_time:
            break

    # If no valid points found (shouldn't happen), return random
    if best_points is None:
        best_points = init_random()
        
    return best_points

# EVOLVE-BLOCK-END
