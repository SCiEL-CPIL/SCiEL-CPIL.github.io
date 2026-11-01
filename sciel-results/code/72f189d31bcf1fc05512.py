N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0

    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))

    if n < 2:
        return min_boundary

    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf

    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_energy(params, beta):
    """
    Compute soft-min energy and gradient.
    Minimizes -softmin(distances) = 1/beta * log(sum(exp(-beta * d)))
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    # Pair distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)
    d_pairs = np.sqrt(d2)
    np.fill_diagonal(d_pairs, 1.0) # dummy value, will be masked

    # Wall distances
    # Left: 0.5 + x, Right: 0.5 - x, Down: 0.5 + y, Up: 0.5 - y
    d_left = 0.5 + x
    d_right = 0.5 - x
    d_down = 0.5 + y
    d_up = 0.5 - y

    # Collect all distances
    # To avoid huge arrays, we can process in chunks or just build the vector.
    # N=200, pairs ~ 20k. Walls 800. Total ~ 21k. This is fine.
    
    # We need the gradients.
    # Let D be the vector of all distances.
    # E = 1/beta * log(sum(exp(-beta * D)))
    # dE/dx = 1/beta * sum( (-beta * dD_k/dx) * exp(-beta * D_k) ) / sum(exp(-beta * D_k))
    #       = - sum( (dD_k/dx) * w_k )
    # where w_k = exp(-beta * D_k) / sum(exp(-beta * D_j))
    
    # For numerical stability, shift by min D.
    # But we need to compute D first.
    
    # Build D vector
    # Pairs: upper triangle
    iu = np.triu_indices(N, 1)
    D_pairs = d_pairs[iu]
    
    # Walls
    D_walls = np.concatenate([d_left, d_right, d_down, d_up])
    
    D = np.concatenate([D_pairs, D_walls])
    
    # Clip to avoid 0 or negative (shouldn't happen if inside, but just in case)
    D = np.maximum(D, 1e-8)
    
    D_min = np.min(D)
    # Shift
    D_shifted = D - D_min
    
    # Compute weights
    # exp(-beta * D_shifted)
    exponents = -beta * D_shifted
    # Clip exponents to avoid underflow/overflow issues in sum, though exp(-large) is 0.
    # Max exponent is 0. Min is -beta * (max_D - min_D).
    exp_vals = np.exp(exponents)
    
    Z = np.sum(exp_vals)
    weights = exp_vals / Z
    
    # Energy
    E = D_min + (1.0 / beta) * np.log(Z)
    
    # Gradient
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    # Pair gradients
    # w_pairs = weights[:len(D_pairs)]
    w_pairs = weights[:len(D_pairs)]
    
    # dD_pairs/dx_i = (x_i - x_j) / d_ij
    # We have dx, dy matrices.
    # d_pairs is the matrix.
    # We need to sum over j for each i.
    
    # Reconstruct dx, dy for pairs only? Or use full matrix and mask?
    # Using full matrix is easier.
    
    # grad_pair_x[i] = - sum_j w_ij * (x_i - x_j) / d_ij
    # Note: w_ij is symmetric in the sum if we consider all pairs, but here we only have upper triangle in D.
    # The weight w_k corresponds to pair (i,j) with i<j.
    # Contribution to grad[i]: - w_k * (x_i - x_j) / d_k
    # Contribution to grad[j]: - w_k * (x_j - x_i) / d_k
    
    # Let's use the full d_pairs matrix for gradient calculation, but we only have weights for upper triangle.
    # We can create a weight matrix W where W[i,j] = w_k if (i,j) is in upper tri, else 0.
    # Then grad_x[i] = - sum_j W[i,j] * (x_i - x_j) / d_pairs[i,j]
    
    W = np.zeros((N, N))
    W[iu] = w_pairs
    W = W + W.T # Symmetric? 
    # Wait, the distance d_ij is the same. The weight w_k is for the pair.
    # The soft min sum is over unique pairs.
    # So the "force" on i from j is proportional to w_k * gradient of d_ij wrt x_i.
    # gradient of d_ij wrt x_i is (x_i - x_j)/d_ij.
    # So yes, W should be symmetric if we assign w_k to both W[i,j] and W[j,i]?
    # No, the sum in the energy is $\sum_{k \in \text{pairs}} e^{-\beta d_k}$.
    # The derivative is $\sum_{k} \frac{\partial E}{\partial d_k} \frac{\partial d_k}{\partial x_i}$.
    # $\frac{\partial E}{\partial d_k} = -w_k$.
    # So for pair (i,j), contribution to grad[i] is $-w_k \frac{x_i - x_j}{d_k}$.
    # Contribution to grad[j] is $-w_k \frac{x_j - x_i}{d_k}$.
    # So if we build W such that W[i,j] = w_k for i<j, then:
    # grad_x[i] = - sum_j W[i,j] * (x_i - x_j) / d_ij - sum_j W[j,i] * (x_i - x_j) / d_ij ?
    # No, W[j,i] is 0 in our construction if we only fill upper tri.
    # So we need to handle both i<j and j<i.
    # Let's just fill the full W matrix with the weights.
    W_full = np.zeros((N, N))
    W_full[iu] = w_pairs
    W_full = W_full + W_full.T
    
    # Now d_pairs is the full matrix (with diagonal 1.0, but W is 0 there).
    # We need to divide by d_pairs.
    # inv_d = 1 / d_pairs
    # Set diagonal to 0 to avoid division by 1 (though W is 0 there).
    inv_d = np.zeros((N, N))
    mask = d_pairs > 1e-8
    inv_d[mask] = 1.0 / d_pairs[mask]
    
    # grad_x[i] = - sum_j W_full[i,j] * (x_i - x_j) * inv_d[i,j]
    # = - sum_j W_full[i,j] * (x_i * inv_d[i,j] - x_j * inv_d[i,j])
    # = - x_i * sum_j W_full[i,j] * inv_d[i,j] + sum_j W_full[i,j] * x_j * inv_d[i,j]
    
    # Let's just do the direct sum, it's N=200, so 40k ops.
    # (x_i - x_j) * inv_d
    diff_x = dx * inv_d
    diff_y = dy * inv_d
    
    grad_x -= np.sum(W_full * diff_x, axis=1)
    grad_y -= np.sum(W_full * diff_y, axis=1)
    
    # Wall gradients
    w_walls = weights[len(D_pairs):]
    # Order: d_left (N), d_right (N), d_down (N), d_up (N)
    w_left = w_walls[:N]
    w_right = w_walls[N:2*N]
    w_down = w_walls[2*N:3*N]
    w_up = w_walls[3*N:4*N]
    
    # d(d_left)/dx = 1, d(d_right)/dx = -1
    # d(d_down)/dy = 1, d(d_up)/dy = -1
    
    # grad_x += - sum(w_left * 1) - sum(w_right * -1) = - sum(w_left) + sum(w_right)
    grad_x -= np.sum(w_left) - np.sum(w_right)
    grad_y -= np.sum(w_down) - np.sum(w_up)
    
    total_grad = np.concatenate([grad_x, grad_y])
    
    return E, total_grad

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0

    for d in np.linspace(0.06, 0.08, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2

        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1

        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.49, 0.49)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    if best_c is not None:
        return best_c

    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _local_polish(c, max_iters=2000, time_limit=20.0):
    """
    Local search to maximize minimum distance.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)

    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        dist_matrix = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)

        min_val = np.min(dist_matrix)
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

        bdists = np.minimum(
            np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
            np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1])
        )
        min_boundary_val = np.min(bdists)
        b_idx = np.argmin(bdists)

        if min_val / 2.0 < min_boundary_val:
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                unit_vec = vec / norm
                step = 1e-4
                new_ci = c[i] + unit_vec * step
                new_cj = c[j] - unit_vec * step

                if np.all(np.abs(new_ci) < 0.5) and np.all(np.abs(new_cj) < 0.5):
                    c[i] = new_ci
                    c[j] = new_cj
                    new_r = _compute_radius(c)
                    if new_r > current_r:
                        current_r = new_r
                    else:
                        c[i] = c[i] - unit_vec * step
                        c[j] = c[j] + unit_vec * step
        else:
            x, y = c[b_idx]
            d_left = 0.5 - x
            d_right = x + 0.5
            d_down = 0.5 - y
            d_up = y + 0.5

            min_d = min(d_left, d_right, d_down, d_up)
            step = 1e-4
            if min_d == d_left:
                new_pt = c[b_idx] + np.array([step, 0])
            elif min_d == d_right:
                new_pt = c[b_idx] + np.array([-step, 0])
            elif min_d == d_down:
                new_pt = c[b_idx] + np.array([0, step])
            else:
                new_pt = c[b_idx] + np.array([0, -step])

            if np.all(np.abs(new_pt) < 0.5):
                c[b_idx] = new_pt
                new_r = _compute_radius(c)
                if new_r > current_r:
                    current_r = new_r
                else:
                    c[b_idx] = c[b_idx] - np.sign(new_pt - c[b_idx]) * step

    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 145

    c0 = _initialize_hexagonal()
    r0 = _compute_radius(c0)
    best_c = c0.copy()
    best_r = r0

    if time.time() < deadline - 30:
        try:
            bounds = [(-0.499, 0.499)] * (2 * N)
            
            # Annealing beta
            betas = [10, 50, 100, 200]
            current_params = c0.flatten()
            
            for beta in betas:
                if time.time() > deadline - 10:
                    break
                
                # Define wrapper for current beta
                def objective(params, beta_val=beta):
                    return _soft_min_energy(params, beta_val)[0]
                
                def gradient(params, beta_val=beta):
                    return _soft_min_energy(params, beta_val)[1]
                
                res = minimize(
                    objective,
                    current_params,
                    method='L-BFGS-B',
                    jac=gradient,
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
                )
                
                current_params = np.clip(res.x, -0.499, 0.499)
                c_opt = current_params.reshape(N, 2)
                r_opt = _compute_radius(c_opt)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt.copy()

        except Exception:
            pass

    if time.time() < deadline - 5:
        c_polished, r_polished = _local_polish(best_c, max_iters=2000, time_limit=min(20.0, deadline - time.time() - 5))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()

    best_c = np.clip(best_c, -0.5, 0.5)

    return best_c
