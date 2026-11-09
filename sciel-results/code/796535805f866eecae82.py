"""Optimized covering of the unit sphere with 130 points.
Uses multi-restart repulsive-force (RFD) optimization followed by
soft-max covering-radius polishing.
"""
import time
import numpy as np

N = 130


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(points)
        facets = hull.simplices

        max_circumradius = 0.0
        for simplex in facets:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]

            a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
            b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
            c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))

            s = (a + b + c) / 2
            denom = 4 * np.sin(s) * np.sin(s - a) * np.sin(s - b) * np.sin(s - c)

            if abs(denom) > 1e-15:
                sin_R = np.sin(a) * np.sin(b) * np.sin(c) / denom
                sin_R = np.clip(sin_R, 0, 1)
                R = np.arcsin(sin_R)
            else:
                R = max(a, b, c) / 2

            if R > max_circumradius:
                max_circumradius = R

        return np.degrees(max_circumradius)

    except Exception:
        # Fallback: sample-based estimate
        rng = np.random.RandomState(42)
        n_samples = 20000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)

        min_dists = np.inf
        for i in range(0, n_samples, 2000):
            chunk = samples[i:i + 2000]
            d = 1 - np.einsum('ij,kj->ik', chunk, points)
            d = np.clip(d, -1, 1)
            min_d = np.min(d, axis=1)
            min_dists = np.minimum(min_dists, min_d)

        max_min_dist = np.max(min_dists)
        return np.degrees(np.arccos(np.clip(max_min_dist, -1, 1)))


def _fibonacci_sphere(n, seed=0):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2 * np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _soft_max_surrogate(points, samples, beta):
    """Compute a smooth surrogate for the covering radius using soft-max.
    
    Returns (surrogate_value, gradient) where gradient is (N,3).
    """
    # samples: (M, 3) unit vectors
    # For each sample, compute distance to each point
    # d[i,j] = 1 - samples[i] · points[j]  (chord-like, monotone with angle)
    d = 1 - np.einsum('ij,kj->ik', samples, points)
    d = np.clip(d, 0, 2)
    
    # For each sample, soft-max over the negative distances (i.e., soft-min of distances)
    # We want to maximize the min-distance, so we use soft-min:
    # soft_min_j(d[i,j]) = -1/beta * log(sum_j exp(-beta * d[i,j]))
    # The covering radius surrogate is the max over samples of soft_min
    
    neg_beta_d = -beta * d  # shape (M, N)
    # Subtract max for numerical stability
    max_neg = np.max(neg_beta_d, axis=1, keepdims=True)
    exp_neg = np.exp(neg_beta_d - max_neg)
    log_sum = max_neg.squeeze() + np.log(np.sum(exp_neg, axis=1))
    soft_min = -log_sum / beta  # shape (M,)
    
    # Now we want the max over samples: use another soft-max
    # surrogate = -1/beta2 * log(sum_i exp(-beta2 * soft_min[i]))
    beta2 = beta
    neg_beta2_sm = -beta2 * soft_min
    max_neg2 = np.max(neg_beta2_sm, keepdims=True)
    exp_neg2 = np.exp(neg_beta2_sm - max_neg2)
    surrogate = -(max_neg2.squeeze() + np.log(np.sum(exp_neg2))) / beta2
    
    # Gradient computation:
    # d(surrogate)/d(points[j]) = sum_i (w_i * d(soft_min_i)/d(points[j]))
    # where w_i = exp(beta2 * soft_min_i) / sum_k exp(beta2 * soft_min_k)
    # (note the sign: surrogate = -1/beta2 * log(sum exp(-beta2*sm)), so
    #  d surrogate/d sm_i = exp(-beta2*sm_i)/sum = exp(beta2*(sm_i - max_neg2))/sum... 
    #  Let me be more careful)
    
    # surrogate = -1/beta2 * log(sum_i exp(-beta2 * sm_i))
    # Let z_i = -beta2 * sm_i
    # surrogate = -1/beta2 * (max_z + log(sum exp(z_i - max_z)))
    # d surrogate / d sm_i = -1/beta2 * d/dsm_i [max_z + log(sum exp(z_i - max_z))]
    # z_i = -beta2 * sm_i, so d z_i / d sm_i = -beta2
    # d surrogate / d sm_i = -1/beta2 * (-beta2) * exp(z_i - max_z) / sum_j exp(z_j - max_z)
    #                      = exp(z_i - max_z) / sum_j exp(z_j - max_z)
    #                      = exp_neg2_i / sum(exp_neg2)
    
    weights = exp_neg2.squeeze() / np.sum(exp_neg2)  # shape (M,)
    
    # d(soft_min_i) / d(points[j]):
    # soft_min_i = -1/beta * log(sum_j exp(-beta * d_ij))
    # where d_ij = 1 - samples[i]·points[j]
    # d d_ij / d points[j] = -samples[i]
    # d(soft_min_i)/d points[j] = -1/beta * d/dpj [log(sum_k exp(-beta * d_ik))]
    #   = -1/beta * sum_k exp(-beta*d_ik) * (-beta) * d d_ik/dpj / sum_k exp(-beta*d_ik)
    #   = -1/beta * sum_k exp(-beta*d_ik) * (-beta) * (-samples[i]) / sum_k exp(-beta*d_ik)
    #   = -1/beta * beta * samples[i] * sum_k exp(-beta*d_ik) / sum_k exp(-beta*d_ik)
    #   = -samples[i]
    # Wait, that can't be right. Let me redo.
    # soft_min_i = -1/beta * log(sum_j exp(-beta * d_ij))
    # Let S_i = sum_j exp(-beta * d_ij)
    # d soft_min_i / d d_ij = -1/beta * (-beta * exp(-beta*d_ij)) / S_i = exp(-beta*d_ij)/S_i
    # d d_ij / d points[j] = -samples[i]
    # d soft_min_i / d points[j] = sum_j' d(soft_min_i)/d(d_ij') * d(d_ij')/d(points[j])
    #   = (exp(-beta*d_ij)/S_i) * (-samples[i])   (only j'=j contributes)
    #   = -exp(-beta*d_ij)/S_i * samples[i]
    
    # So gradient of surrogate w.r.t. points[j]:
    # grad_j = sum_i weights_i * (-exp(-beta*d_ij)/S_i) * samples[i]
    
    # S_i = sum_k exp(-beta * d_ik) = sum_k exp(neg_beta_d[i,k])
    # We already computed exp_neg = exp(neg_beta_d - max_neg), so S_i = exp(max_neg_i) * sum(exp_neg[i,:])
    
    S = np.exp(max_neg.squeeze()) * np.sum(exp_neg, axis=1)  # shape (M,)
    # exp(-beta*d_ij)/S_i = exp_neg[i,j] / sum(exp_neg[i,:])
    probs = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)  # shape (M, N)
    
    # grad_j = -sum_i weights_i * probs[i,j] * samples[i]
    # = -(weights @ probs) @ samples  ... no
    # grad shape (N, 3): grad[j] = -sum_i weights[i] * probs[i,j] * samples[i]
    # = -((weights * probs) @ samples.T).T  ... let me think
    # weights: (M,), probs: (M, N), samples: (M, 3)
    # weighted_probs = weights[:, None] * probs  (M, N)
    # grad = -(weighted_probs.T @ samples)  (N, 3)
    
    weighted_probs = weights[:, None] * probs  # (M, N)
    grad = -(weighted_probs.T @ samples)  # (N, 3)
    
    return surrogate, grad


def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms


def _optimize_repulsion(points, max_time, seed=0):
    """Optimize using repulsive forces (RFD: maximize minimum pairwise distance).
    
    Uses gradient ascent on the minimum pairwise distance via soft-min.
    """
    start_time = time.time()
    points = _project_to_sphere(points.copy())
    best_points = points.copy()
    best_min_dist = _min_pairwise_dist(points)
    
    beta = 50.0  # soft-min sharpness
    lr = 0.005
    n_samples_for_check = 5000
    
    for iteration in range(20000):
        if time.time() - start_time > max_time:
            break
        
        # Compute pairwise distances
        # d[i,j] = 1 - points[i]·points[j] for i != j
        d = 1 - points @ points.T
        np.fill_diagonal(d, np.inf)
        d = np.clip(d, 0, 2)
        
        # Soft-min over j for each i (excluding self)
        neg_beta_d = -beta * d
        max_neg = np.max(neg_beta_d, axis=1, keepdims=True)
        exp_neg = np.exp(neg_beta_d - max_neg)
        S = np.sum(exp_neg, axis=1, keepdims=True)
        probs = exp_neg / S  # (N, N)
        
        # soft_min_i = -1/beta * (max_neg_i + log(S_i))
        soft_min = -(max_neg.squeeze() + np.log(S.squeeze())) / beta
        
        # Gradient of sum_i soft_min_i w.r.t. points[j]:
        # d soft_min_i / d points[j] = -probs[i,j] * points[j] + probs[i,j] * points[i]
        # (because d_ij = 1 - pi·pj, d/dpj = -pi, and d/dpi = -pj)
        # Actually: d soft_min_i / d d_ij = probs[i,j]
        # d d_ij / d points[j] = -points[i]
        # d d_ij / d points[i] = -points[j]
        # grad_{points[j]} = sum_i probs[i,j] * (-points[i])  (from i≠j terms where j is the second index)
        #                   + sum_i probs[j,i] * (-points[i])  (from j≠i terms where j is the first index)
        # Wait, let me be more careful.
        # soft_min_i = -1/beta * log(sum_{k≠i} exp(-beta * d_ik))
        # d(soft_min_i)/d(points[j]) for j≠i:
        #   = -1/beta * d/dpj log(sum_{k≠i} exp(-beta*d_ik))
        #   = -1/beta * (-beta * exp(-beta*d_ij) * d(d_ij)/dpj) / sum_{k≠i} exp(-beta*d_ik)
        #   = exp(-beta*d_ij)/sum * d(d_ij)/dpj
        #   = probs[i,j] * (-points[i])
        # For j=i: d(soft_min_i)/d(points[i]) = sum_{k≠i} probs[i,k] * d(d_ik)/dp_i
        #   = sum_{k≠i} probs[i,k] * (-points[k])
        
        # Total gradient w.r.t. points[j]:
        # grad_j = sum_{i≠j} probs[i,j] * (-points[i]) + sum_{k≠j} probs[j,k] * (-points[k])
        #        = -(probs.T @ points)[j] - (probs @ points)[j] + 2*points[j] (since diagonal is 0)
        # Wait: sum_{i≠j} probs[i,j]*(-points[i]) = -(probs.T @ points)[j] (diagonal of probs is 0)
        # sum_{k≠j} probs[j,k]*(-points[k]) = -(probs @ points)[j] (diagonal is 0)
        # So grad_j = -(probs.T @ points)[j] - (probs @ points)[j]
        
        grad = -(probs.T @ points) - (probs @ points)  # (N, 3)
        
        # Project gradient to tangent space
        for j in range(N):
            grad[j] -= np.dot(grad[j], points[j]) * points[j]
        
        # Gradient ascent (we want to maximize min distance)
        points = points + lr * grad
        points = _project_to_sphere(points)
        
        # Periodically check
        if iteration % 200 == 0:
            min_dist = _min_pairwise_dist(points)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_points = points.copy()
        
        # Adaptive learning rate
        if iteration % 500 == 0:
            lr *= 0.95
    
    return best_points, best_min_dist


def _min_pairwise_dist(points):
    """Compute the minimum pairwise angular distance."""
    d = 1 - points @ points.T
    np.fill_diagonal(d, np.inf)
    d = np.clip(d, 0, 2)
    return np.arccos(np.clip(1 - np.min(d), -1, 1))


def _optimize_soft_max(points, max_time, beta=30.0):
    """Polish the configuration by minimizing the soft-max covering radius surrogate."""
    start_time = time.time()
    points = _project_to_sphere(points.copy())
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Generate fixed sample points for the surrogate
    rng = np.random.RandomState(12345)
    n_samples = 5000
    samples = rng.randn(n_samples, 3)
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    lr = 0.002
    for iteration in range(15000):
        if time.time() - start_time > max_time:
            break
        
        surrogate, grad = _soft_max_surrogate(points, samples, beta)
        
        # Project gradient to tangent space
        for j in range(N):
            grad[j] -= np.dot(grad[j], points[j]) * points[j]
        
        # Gradient descent on the surrogate (minimize covering radius)
        points = points - lr * grad
        points = _project_to_sphere(points)
        
        if iteration % 300 == 0:
            radius = _covering_radius(points)
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
        
        # Adapt beta: increase it over time for sharper max
        if iteration % 1000 == 0 and beta < 100:
            beta *= 1.3
    
    return best_points, best_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.inf
    
    # Multi-restart: try several initial configurations
    n_restarts = 5
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        remaining = deadline - (start_time + elapsed)
        if remaining < 10:
            break
        
        # Choose initial configuration
        if restart == 0:
            init_points = _fibonacci_sphere(N, seed=restart)
        elif restart == 1:
            init_points = _random_sphere(N, seed=100 + restart)
        elif restart == 2:
            init_points = _fibonacci_sphere(N, seed=1000 + restart)
        elif restart == 3:
            init_points = _random_sphere(N, seed=200 + restart)
        else:
            init_points = _fibonacci_sphere(N, seed=5000 + restart)
        
        # Phase 1: Repulsion optimization (RFD)
        t1 = time.time()
        time_alloc_1 = min(remaining * 0.5, 40)
        pts, min_dist = _optimize_repulsion(init_points, max_time=time_alloc_1, seed=restart)
        
        # Phase 2: Soft-max covering radius polishing
        t2 = time.time()
        remaining2 = deadline - (start_time + (t2 - start_time))
        time_alloc_2 = max(remaining2 - 5, 5)
        pts, radius = _optimize_soft_max(pts, max_time=time_alloc_2)
        
        if radius < best_radius:
            best_radius = radius
            best_points = pts.copy()
        
        # Check time
        if time.time() > deadline - 10:
            break
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    # Final normalization
    best_points = _project_to_sphere(best_points)
    
    return best_points
