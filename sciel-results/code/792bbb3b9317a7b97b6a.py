"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def _compute_radius(centres):
    """Compute the exact maximum common radius for given centres."""
    if len(centres) == 0:
        return 0.0
    
    # Boundary distance: min(0.5 - |c_ik|)
    d_boundary = np.min(0.5 - np.abs(centres), axis=1)
    min_boundary = np.min(d_boundary)
    
    # Pairwise distances
    # Use vectorized computation for speed
    # diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    # d = np.linalg.norm(diff, axis=-1)
    # min_pair = np.min(d[np.triu_indices_from(d, k=1)])
    
    # To save memory and time, we can do it in chunks or just use the full matrix for N=100
    # 100x100 matrix is small.
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=-1)
    # Mask diagonal
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d)
    
    return min(min_pair / 2.0, min_boundary)

def _soft_min_radius_and_grad(centres, beta):
    """
    Vectorized soft-min radius and gradient.
    Returns (value, gradient).
    """
    n = len(centres)
    if n == 0:
        return 0.0, np.zeros_like(centres)
    
    # 1. Boundary constraints
    # For each point i, distance to boundary is min_k (0.5 - |c_ik|)
    # Actually, the global min is min_i min_k (0.5 - |c_ik|).
    # We treat each (i, k) as a separate constraint term for the soft min.
    # There are 3*N boundary terms.
    
    # 2. Pairwise constraints
    # For each pair (i, j), distance is ||c_i - c_j|| / 2.
    # There are N*(N-1)/2 pairwise terms.
    
    # Total terms M = 3*N + N*(N-1)/2
    # For N=100, M = 300 + 4950 = 5250.
    
    # We need to compute the soft min: -log(sum(exp(-beta * v))) / beta
    # and its gradient.
    
    # Let's collect values v and gradients g.
    # v: array of shape (M,)
    # g: array of shape (M, 3*N)
    
    vals = np.empty(3 * n + n * (n - 1) // 2)
    grads = np.empty((3 * n + n * (n - 1) // 2, 3 * n))
    
    idx = 0
    
    # Boundary terms
    # v_ik = 0.5 - |c_ik|
    # grad w.r.t c_ik is -sign(c_ik)
    
    c_abs = np.abs(centres)
    b_vals = 0.5 - c_abs # shape (N, 3)
    vals[idx : idx + 3 * n] = b_vals.flatten()
    
    # Gradient for boundary terms
    # g_ik = -sign(c_ik)
    b_grads = -np.sign(centres)
    # Flatten to (3N,) and expand to (3N, 3N) where each row has one non-zero?
    # No, we need the full gradient vector for each term.
    # Term (i, k) only affects coordinate i*3+k.
    # So the gradient row for term (i, k) is zero except at i*3+k.
    
    # Efficient way:
    # Create a sparse-like structure or just fill the array.
    # Since M is small, we can just loop over the terms for gradient construction if needed,
    # but we want speed.
    
    # Let's construct the full gradient matrix for boundary terms.
    # b_grads_flat has shape (3N,).
    # We need a matrix of shape (3N, 3N) where row j has 1 at col j (scaled by -sign).
    # Actually, the gradient of v_j w.r.t x is a vector.
    # For boundary term j (corresponding to index i, k), grad is vector e_j * (-sign(c_j)).
    
    # Let's just fill the `grads` array for boundary parts.
    # Row r in 0..3N-1 corresponds to boundary term r.
    # The gradient is zero everywhere except at index r, where it is -sign(centres[r//3, r%3]).
    
    b_grad_mat = np.zeros((3 * n, 3 * n))
    np.fill_diagonal(b_grad_mat, -np.sign(centres).flatten())
    grads[idx : idx + 3 * n] = b_grad_mat
    idx += 3 * n
    
    # Pairwise terms
    # i < j
    # v_ij = ||c_i - c_j|| / 2
    # grad w.r.t c_i = (c_i - c_j) / (2 * ||c_i - c_j||)
    # grad w.r.t c_j = (c_j - c_i) / (2 * ||c_i - c_j||)
    
    # Get indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)
    
    diffs = centres[i_idx] - centres[j_idx]
    dists = np.linalg.norm(diffs, axis=1)
    
    # Avoid division by zero
    safe_dists = np.where(dists < 1e-12, 1.0, dists)
    
    p_vals = dists / 2.0
    vals[idx : idx + num_pairs] = p_vals
    
    # Gradients
    # For each pair p (corresponding to i, j), the gradient vector has non-zeros at i and j.
    # g_i = diffs[p] / (2 * dists[p])
    # g_j = -diffs[p] / (2 * dists[p])
    
    p_grad_mat = np.zeros((num_pairs, 3 * n))
    
    # Vectorized assignment for gradients
    # This is tricky with simple indexing because we are writing to rows.
    # We can use advanced indexing.
    # For each pair p:
    # p_grad_mat[p, i_idx[p]*3 : i_idx[p]*3+3] = diffs[p] / (2 * safe_dists[p])
    # p_grad_mat[p, j_idx[p]*3 : j_idx[p]*3+3] = -diffs[p] / (2 * safe_dists[p])
    
    # Let's do this in a loop for clarity and correctness, as num_pairs is ~5000.
    # 5000 iterations in Python is fast (<0.1s).
    for p in range(num_pairs):
        i = i_idx[p]
        j = j_idx[p]
        d = diffs[p]
        dist = safe_dists[p]
        grad_vec = d / (2.0 * dist)
        p_grad_mat[p, i * 3 : i * 3 + 3] = grad_vec
        p_grad_mat[p, j * 3 : j * 3 + 3] = -grad_vec
        
    grads[idx : idx + num_pairs] = p_grad_mat
    
    # Compute soft min
    # v = vals
    # f = -log(sum(exp(-beta * v))) / beta
    # Let u = -beta * v
    # f = (log(sum(exp(u))) - max(u)) / beta  ... wait.
    # sum(exp(u)) = exp(max(u)) * sum(exp(u - max(u)))
    # log(sum(exp(u))) = max(u) + log(sum(exp(u - max(u))))
    # f = -(max(u) + log(sum(exp(u - max(u))))) / beta
    
    u = -beta * vals
    u_max = np.max(u)
    exp_u = np.exp(u - u_max)
    sum_exp = np.sum(exp_u)
    log_sum = u_max + np.log(sum_exp)
    
    soft_min_val = -log_sum / beta
    
    # Gradient
    # df/dx = sum_p ( p_p * (-beta) * dv_p/dx ) / beta
    #        = - sum_p ( p_p * dv_p/dx )
    # where p_p = exp(u_p) / sum(exp(u_q))
    
    probs = exp_u / sum_exp
    
    # grad = - (probs @ grads)
    # probs shape (M,), grads shape (M, 3N)
    # result shape (3N,)
    grad = -np.dot(probs, grads)
    
    return soft_min_val, grad


def _fcc_lattice_init():
    """
    Generate FCC lattice points within the cube and select the best 100.
    """
    points = []
    a1 = np.array([0.5, 0.5, 0.0])
    a2 = np.array([0.5, 0.0, 0.5])
    a3 = np.array([0.0, 0.5, 0.5])
    
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.5])
    ]
    
    for offset in offsets:
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    p = i * a1 + j * a2 + k * a3 + offset
                    if np.all(p >= -0.5 - 1e-9) and np.all(p <= 0.5 + 1e-9):
                        p_clamped = np.clip(p, -0.49, 0.49)
                        points.append(p_clamped)
    
    points = np.array(points)
    
    if len(points) > 0:
        rounded = np.round(points, decimals=6)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        points = points[unique_idx]
    
    if len(points) < N:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if len(points) > N:
        d_boundary = np.min(0.5 - np.abs(points), axis=1)
        idx = np.argsort(-d_boundary)[:N]
        points = points[idx]
    
    return points[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    centres = _fcc_lattice_init().astype(np.float64)
    
    best_radius = _compute_radius(centres)
    best_centres = centres.copy()
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Stages with increasing beta
    betas = [20.0, 50.0, 100.0, 200.0, 400.0]
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        def objective(x):
            c = x.reshape(-1, 3)
            val, _ = _soft_min_radius_and_grad(c, beta)
            return -val
            
        def grad(x):
            c = x.reshape(-1, 3)
            _, g = _soft_min_radius_and_grad(c, beta)
            return g
        
        from scipy.optimize import minimize
        
        x0 = best_centres.flatten()
        
        # Use L-BFGS-B
        try:
            res = minimize(
                objective, x0, jac=grad, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < objective(x0):
                new_centres = res.x.reshape(-1, 3)
                new_centres = np.clip(new_centres, -0.5, 0.5)
                r = _compute_radius(new_centres)
                if r > best_radius:
                    best_radius = r
                    best_centres = new_centres
        except Exception:
            pass
            
    # Final polish: A few iterations of direct repulsion/local search
    # This helps if the soft-min is still slightly off the exact min.
    # We iterate: find the pair with min distance, push them apart.
    # Also push points away from boundaries.
    
    if time.time() < deadline - 5.0:
        c = best_centres.copy()
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            # Compute all pair distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(d, np.inf)
            
            # Find min pair
            min_idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = min_idx
            if i == j: continue
            dist = d[i, j]
            if dist < 1e-6:
                # Random nudge
                c[i] += np.random.normal(0, 1e-4, 3)
                c[j] += np.random.normal(0, 1e-4, 3)
            else:
                # Push apart
                vec = (c[i] - c[j]) / dist
                # How much to push?
                # We want to increase the min distance.
                # A simple heuristic: push by a small fraction of the current min distance gap?
                # Or just a fixed small step?
                # Let's try to equalize the distances? No, just push.
                step = 0.01 * dist
                c[i] += vec * step
                c[j] -= vec * step
            
            # Push away from boundaries
            for k in range(3):
                # Find point closest to boundary
                # dist to +0.5: 0.5 - c[k]
                # dist to -0.5: 0.5 + c[k]
                d_pos = 0.5 - c[:, k]
                d_neg = 0.5 + c[:, k]
                # We want to maximize min(d_pos, d_neg)
                # If d_pos < d_neg, push towards -0.5? No, push towards center.
                # If d_pos is small, c is close to 0.5. Push to 0.5 - eps.
                # Gradient of min(d_pos, d_neg) is -sign(c) if we want to increase distance.
                # Actually, if c > 0, dist is 0.5 - c. To increase, decrease c.
                # If c < 0, dist is 0.5 + c. To increase, increase c.
                # So push towards 0.
                # Force: -sign(c) * step
                # But we only want to act if it's the bottleneck?
                # For simplicity, apply a small restoring force to center for all.
                # This might hurt if a point is well placed.
                # Let's only push if it's the global bottleneck?
                pass # Skip boundary push in this simple loop to avoid complexity
            
            c = np.clip(c, -0.5, 0.5)
            
            r = _compute_radius(c)
            if r > best_radius:
                best_radius = r
                best_centres = c
    
    return best_centres
# EVOLVE-BLOCK-END
