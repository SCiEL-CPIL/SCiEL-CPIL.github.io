"""Packing 50 equal circles in a unit circle via ring-based parametric optimization + smooth polish + constraint push."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
SEED = 42
DEADLINE_MARGIN = 15.0


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    return max(min(min_pair, min_boundary), 0.0)


def _softmin_logsumexp(values, beta):
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum == 0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta=50.0):
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _adam_optimize(pts, deadline, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, beta_start=10.0, beta_end=100.0):
    """
    Optimize points using Adam gradient descent with a smooth soft-min objective.
    Returns the optimized points.
    """
    x = pts.copy().flatten()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    
    # We will ramp up the sharpness (beta) of the soft-min over time
    total_steps = 5000
    # Estimate steps per second to determine how many steps we can afford
    # Rough estimate: 1 step ~ 1ms? Let's just run until deadline.
    
    start_time = time.time()
    
    while time.time() < deadline - 10:
        t += 1
        # Ramp beta from beta_start to beta_end
        progress = min(1.0, t / total_steps)
        beta = beta_start + (beta_end - beta_start) * progress
        
        # Compute gradient of the negative soft-min score
        # Score is min of pairwise distances/2 and boundary margins.
        # We want to maximize this, so minimize -score.
        # Objective: - softmin(dists/2, margins)
        
        pts_curr = x.reshape(N, 2)
        
        # Pairwise distances
        diff = pts_curr[:, np.newaxis, :] - pts_curr[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary margins
        norms = np.sqrt(np.sum(pts_curr ** 2, axis=1) + 1e-12)
        margins = 1.0 - norms
        
        # Combine all constraints
        # We use a soft-min approximation: min(x) ~ -1/beta * log(sum(exp(-beta*x)))
        # Gradient of softmin(x) w.r.t x_i is proportional to exp(-beta*x_i) / sum(exp(-beta*x_j))
        
        # To avoid overflow/underflow, we compute logits = -beta * values
        # values = [dists/2, margins]
        
        pair_vals = dists / 2.0
        all_vals = np.concatenate([pair_vals.ravel(), margins])
        
        # Filter out non-finite (diagonal is inf)
        finite_mask = np.isfinite(all_vals)
        if not np.any(finite_mask):
            break
        
        logits = -beta * all_vals
        max_logit = np.max(logits[finite_mask])
        
        # exp_sum
        exp_logits = np.zeros_like(logits)
        exp_logits[finite_mask] = np.exp(logits[finite_mask] - max_logit)
        exp_sum = np.sum(exp_logits)
        
        if exp_sum == 0:
            break
            
        # Gradient of softmin w.r.t all_vals
        # d(softmin)/dx_i = -1/beta * d(logsumexp)/dx_i
        # d(logsumexp)/dx_i = exp(-beta*x_i) / sum(exp(-beta*x_j))
        # So d(softmin)/dx_i = -1/beta * exp(-beta*x_i) / sum(...)
        
        probs = exp_logits / exp_sum
        # This is the probability weight for each constraint
        # Gradient of objective (-softmin) w.r.t x is:
        # grad_obj = -1 * grad_softmin
        # grad_softmin w.r.t val_i = -1/beta * prob_i
        # So grad_obj w.r.t val_i = 1/beta * prob_i
        
        grad_vals = probs / beta
        
        # Now backpropagate grad_vals to x (points)
        grad_x = np.zeros_like(x)
        
        # Part 1: Pairwise distances
        # val_k corresponds to dist(i,j)/2 for i<j
        # d(dist(i,j)/2)/d(pts[i]) = (pts[i] - pts[j]) / (2 * dist(i,j))
        # Note: dist(i,j) = |pts[i] - pts[j]|
        # Let d = dist(i,j). val = d/2.
        # d(val)/d(pts[i]) = 1/2 * (pts[i] - pts[j]) / d = (pts[i] - pts[j]) / (2d)
        
        # We need to map grad_vals back to the pair indices
        # The first N*(N-1)/2 elements of all_vals are the pairs
        # We can reconstruct the gradient matrix G such that G[i,j] is the gradient contribution
        
        # Actually, it's easier to iterate or use vectorized ops if possible.
        # Given N=50, N^2=2500. Iteration is fine for 5000 steps? 2500*5000 = 12.5M ops.
        # Vectorized is better.
        
        # Create a matrix of gradients for pairs
        # grad_pair[i,j] = grad_val_{i,j} * (pts[i] - pts[j]) / (2 * dist(i,j))
        
        grad_pair_mat = np.zeros((N, N, 2))
        
        # We need to map the flat index in grad_vals to (i, j)
        # This is a bit tedious. Let's just compute the gradient for each point by summing contributions.
        
        # Alternative: Compute gradient of softmin directly on the tensor?
        # Let's stick to the explicit loop or a clever vectorization.
        # Since N is small, let's do a vectorized computation of the gradient of the pairwise part.
        
        # diff is (N, N, 2). dists is (N, N).
        # We have grad_vals for pairs. Let's create a matrix grad_dists_mat (N, N)
        # where grad_dists_mat[i, j] = grad_vals[k] if (i, j) is the k-th pair.
        
        # We can fill this by iterating over i < j.
        # Or, we can note that the gradient of the softmin of a set of values V is a vector G_V.
        # The gradient w.r.t points is sum over constraints c of G_V[c] * d(V[c])/d(points).
        
        # Let's compute the gradient contribution from pairwise distances first.
        # d(dist(i,j))/d(pts[i]) = (pts[i] - pts[j]) / dist(i,j)
        # d(dist(i,j)/2)/d(pts[i]) = (pts[i] - pts[j]) / (2 * dist(i,j))
        
        # We need the weight w_{ij} = grad_vals[index(i,j)]
        
        # Let's create a weight matrix W (N, N)
        W = np.zeros((N, N))
        # Fill W for i < j
        for i in range(N):
            for j in range(i + 1, N):
                # Find index in all_vals
                # all_vals = [dists/2 (upper tri), margins]
                # Index in dists/2: sum of previous rows
                # This loop is 1225 iterations. 5000 steps * 1225 = 6M ops. Acceptable.
                # But we can do it faster.
                pass
        
        # Faster way:
        # Create indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        pair_indices = np.arange(len(i_idx))
        
        # Map pair_indices to the flat array indices in all_vals
        # all_vals starts with pairs. So indices are just 0..len(pairs)-1
        pair_grads = grad_vals[:len(i_idx)]
        
        # Now, for each pair (i, j), the gradient contribution to pts[i] is:
        # pair_grads[k] * (pts[i] - pts[j]) / (2 * dists[i, j])
        
        # We can accumulate this into grad_x.
        
        # Vectorized accumulation:
        # For each k, we add to grad_x[2*i_k] and grad_x[2*i_k+1]
        
        # Let's use advanced indexing
        # grad_x is flat (N*2,)
        
        # diff_ij = pts[i] - pts[j]
        # factor = pair_grads / (2 * dists[i, j])
        # contrib = factor * diff_ij
        
        # We need to add contrib to pts[i] and subtract from pts[j]
        
        # Let's reshape x to (N, 2) for easier handling
        x_pts = x.reshape(N, 2)
        grad_x_pts = np.zeros_like(x_pts)
        
        # diff is already computed (N, N, 2)
        # We need dists[i, j] for i < j
        # We have dists matrix.
        
        # Create a mask for i < j
        mask = np.zeros((N, N), dtype=bool)
        mask[i_idx, j_idx] = True
        
        # We need to multiply pair_grads with the corresponding elements.
        # Let's create a matrix G_mat (N, N) where G_mat[i, j] = pair_grads[k] if mask[i, j]
        G_mat = np.zeros((N, N))
        G_mat[i_idx, j_idx] = pair_grads
        
        # Now, term = G_mat * (diff / (2 * dists))
        # Note: dists has 0 on diagonal, so division is safe if we mask it.
        # diff/dists is the unit vector.
        
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vectors = diff / (2 * dists[:, :, np.newaxis])
        
        # Set diagonal to 0
        np.fill_diagonal(unit_vectors[:, :, 0], 0)
        np.fill_diagonal(unit_vectors[:, :, 1], 0)
        
        # term = G_mat[:,:,np.newaxis] * unit_vectors
        # This gives (N, N, 2)
        terms = G_mat[:, :, np.newaxis] * unit_vectors
        
        # Sum over j for each i?
        # grad_x_pts[i] = sum_j terms[i, j]
        # Wait, the gradient w.r.t pts[i] is sum over j != i of d(V)/d(pts[i]) * w_{ij}
        # V_{ij} = dist(i,j)/2.
        # d(V_{ij})/d(pts[i]) = (pts[i] - pts[j]) / (2 dist(i,j)) = unit_vectors[i, j]
        # So yes, grad_x_pts[i] = sum_j G_mat[i, j] * unit_vectors[i, j]
        
        grad_x_pts = np.sum(terms, axis=1)
        
        # Now add boundary gradient
        # Boundary margin m_i = 1 - norm(pts[i])
        # d(m_i)/d(pts[i]) = - pts[i] / norm(pts[i])
        # grad_vals for boundaries are grad_vals[len(pairs):]
        boundary_grads = grad_vals[len(i_idx):]
        
        # grad_boundary_pts[i] = boundary_grads[i] * (- pts[i] / norm(pts[i]))
        # norm is norms
        with np.errstate(divide='ignore', invalid='ignore'):
            boundary_terms = -boundary_grads[:, np.newaxis] * (x_pts / norms[:, np.newaxis])
        
        # Add to grad_x_pts
        grad_x_pts += boundary_terms
        
        # Total gradient w.r.t x (flat)
        grad_x = grad_x_pts.flatten()
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_x
        v = beta2 * v + (1 - beta2) * (grad_x ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Project back to valid region?
        # The soft-min penalizes being outside, so it should stay inside.
        # But to be safe, we can clip norms to 0.999
        x_pts_new = x.reshape(N, 2)
        norms_new = np.linalg.norm(x_pts_new, axis=1)
        mask_out = norms_new > 0.999
        if np.any(mask_out):
            x_pts_new[mask_out] *= 0.999 / norms_new[mask_out]
            x = x_pts_new.flatten()
            
    return x.reshape(N, 2)


def _anneal_points(pts, deadline, max_beta_stages=6):
    """Multi-stage annealing optimization from initial points."""
    current_pts = pts.copy()
    current_score = _score(current_pts)
    
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 600.0]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 6000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    # High-precision polish
    for beta in [1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 5:
            break
        x0 = current_pts.ravel()
        try:
            result = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 25000}
            )
            new_pts = result.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            new_score = _score(new_pts)
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        except Exception:
            pass
    
    return current_pts, current_score


def _constraint_push(pts, r_target, deadline, max_iters=500):
    """Push points to achieve target radius via constraint satisfaction."""
    pts = pts.copy()
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        min_pair_margin = np.min(pair_margins)
        min_boundary_margin = np.min(boundary_margins)
        
        if min_pair_margin < min_boundary_margin - 1e-14:
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    
    return pts


def _ring_initialization():
    """Generate ring-based initial configuration."""
    points = np.zeros((N, 2))
    points[0] = [0.0, 0.0]
    r1 = 0.25
    for i in range(6):
        angle = i * 2 * np.pi / 6
        points[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
    r2 = 0.5
    for i in range(12):
        angle = (i + 0.5) * 2 * np.pi / 12
        points[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
    r3 = 0.72
    for i in range(12):
        angle = i * 2 * np.pi / 12
        points[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
    r4 = 0.88
    for i in range(19):
        angle = (i + 0.5) * 2 * np.pi / 19
        points[31 + i] = [r4 * np.cos(angle), r4 * np.sin(angle)]
    return points


def _hexagonal_initial():
    """Generate initial hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points, radius = points[mask], radius[mask]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    np.random.seed(SEED)
    
    best_points = None
    best_score = -1.0
    
    # Strategy 1: Hexagonal initialization + Adam optimization
    if time.time() < deadline - 30.0:
        try:
            initial = _hexagonal_initial()
            # Run Adam optimization
            adam_deadline = time.time() + min(40.0, deadline - time.time() - 10.0)
            adam_points = _adam_optimize(initial, adam_deadline)
            adam_score = _score(adam_points)
            
            if adam_score > best_score:
                best_score = adam_score
                best_points = adam_points.copy()
        except Exception:
            pass
    
    # Strategy 2: Ring-based initialization + Adam optimization
    if time.time() < deadline - 20.0:
        try:
            initial = _ring_initialization()
            adam_deadline = time.time() + min(30.0, deadline - time.time() - 10.0)
            adam_points = _adam_optimize(initial, adam_deadline)
            adam_score = _score(adam_points)
            
            if adam_score > best_score:
                best_score = adam_score
                best_points = adam_points.copy()
        except Exception:
            pass
            
    # Strategy 3: Polish the best result found so far with L-BFGS
    if best_points is not None and time.time() < deadline - 10.0:
        result_points, result_score = _anneal_points(best_points, deadline)
        
        if result_score > best_score:
            best_score = result_score
            best_points = result_points.copy()

    # Strategy 4: Random restarts with Adam optimization (if time permits)
    for restart in range(2):
        if time.time() > deadline - 15.0:
            break
        
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.random.uniform(0.05, 0.85, N)
        initial = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        try:
            adam_deadline = time.time() + min(10.0, deadline - time.time() - 5.0)
            adam_points = _adam_optimize(initial, adam_deadline)
            adam_score = _score(adam_points)
            
            if adam_score > best_score:
                best_score = adam_score
                best_points = adam_points.copy()
        except Exception:
            pass

    # Final constraint push to improve the best solution
    if best_points is not None and time.time() < deadline - 5.0:
        r_target = best_score
        pushed = _constraint_push(best_points, r_target, deadline, max_iters=200)
        pushed_score = _score(pushed)
        if pushed_score > best_score:
            best_score = pushed_score
            best_points = pushed.copy()
        
        # One more round of smooth polish after push
        if time.time() < deadline - 3.0:
            final_points, final_score = _anneal_points(best_points, deadline)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = _hexagonal_initial()
    
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points
