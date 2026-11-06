"""Rewritten program: differentiable soft-min objective + L-BFGS-B + multi-restart."""
import numpy as np
import time

N = 100
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _exact_radius(centres):
    """Compute the true maximum radius (min of pair-dist/2 and boundary-dist)."""
    c = np.asarray(centres, dtype=float)
    n = len(c)
    # Boundary
    r_boundary = (0.5 - np.abs(c)).min()
    # Pairwise min distance
    if n < 2:
        return r_boundary
    # Efficient: compute all pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = dist.min() / 2.0
    return min(r_boundary, r_pair)

def _soft_min_obj(x, tau):
    """
    Differentiable soft-min of {pair-distances, boundary-distances}.
    We minimize -soft_min, i.e. maximize soft_min.
    Returns (value, grad).
    """
    n = len(x) // 3
    c = x.reshape(n, 3)
    
    # Boundary distances: b_i = 0.5 - |c_i|  (per coordinate, take min per sphere)
    # Actually we want min over all coordinates of (0.5 - |c_ik|)
    # b_i = min_k (0.5 - |c_ik|)
    abs_c = np.abs(c)
    boundary_per_coord = 0.5 - abs_c  # shape (n,3)
    # b_i = min over k
    b = boundary_per_coord.min(axis=1)  # shape (n,)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2) + 1e-15)
    np.fill_diagonal(dist, np.inf)
    
    # Collect all distances: pairs (i<j) and boundaries
    # For soft-min we combine all
    # Build vector of all distances
    # Pairs: use upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]  # shape (n*(n-1)/2,)
    all_dists = np.concatenate([pair_dists, b])
    
    # Soft-min: -log-sum-exp(-d/tau) * tau
    # For numerical stability
    m = all_dists.min()
    exp_terms = np.exp(-(all_dists - m) / tau)
    logsumexp = np.log(exp_terms.sum())
    soft_min_val = m - tau * logsumexp
    
    # Gradient of soft_min_val w.r.t. x
    # d(soft_min)/d(all_dists) = -exp(-d/tau) / sum(exp(-d/tau))  ... let me derive
    # soft_min = m - tau * log(sum(exp(-(d-m)/tau)))
    # Let S = sum_j exp(-(d_j - m)/tau)
    # d(soft_min)/d(d_k) = -delta_{km} - tau * (1/S) * d(S)/d(d_k)
    # d(S)/d(d_k) = exp(-(d_k-m)/tau) * (-1/tau)
    # So d(soft_min)/d(d_k) = -delta_{km} + exp(-(d_k-m)/tau)/S
    # Note: the -delta_{km} term only matters if m is one of the d_k (which it is, m = min)
    # For the argmin element, the two terms cancel partially. For simplicity and since
    # tau is small, the dominant gradient is the soft part. Let's use the standard
    # softmax-based gradient which is correct for the smooth approximation.
    weights = exp_terms / exp_terms.sum()  # shape (len(all_dists),)
    # d(soft_min)/d(all_dists) ≈ -weights  (the smooth part; the m term is subgradient)
    # Actually the correct gradient of the log-sum-exp soft-min:
    # soft_min = -tau * logsumexp(-d/tau)  (without the m shift, but m shift doesn't change gradient)
    # d(soft_min)/d(d_k) = -tau * (1/sum(exp(-d/tau))) * exp(-d_k/tau) * (-1/tau) = exp(-d_k/tau)/sum
    # So d(soft_min)/d(d_k) = weights_k. Wait sign:
    # soft_min = -tau * log(sum_j exp(-d_j/tau))
    # d/dd_k = -tau * (1/sum) * exp(-d_k/tau) * (-1/tau) = exp(-d_k/tau)/sum = weights_k
    # So gradient of soft_min w.r.t. all_dists is +weights. But we want to MAXIMIZE soft_min,
    # so we minimize -soft_min, gradient of -soft_min w.r.t. all_dists is -weights.
    
    # Now chain rule: grad_x = sum_k (d(-soft_min)/d(all_dists_k)) * (d(all_dists_k)/dx)
    # = -sum_k weights_k * (d(all_dists_k)/dx)
    
    grad = np.zeros_like(x)
    
    # Pairwise part
    # d(pair_dist_{ij})/dx_i = (c_i - c_j)/dist_{ij}, d/dx_j = (c_j - c_i)/dist_{ij}
    # We have pair_dists indexed by iu
    n_pairs = len(pair_dists)
    # Build gradient contribution from pairs
    # For each pair (i,j) with weight w, contribution to grad_x_i = -w * (c_i-c_j)/d_ij
    # and grad_x_j = -w * (c_j-c_i)/d_ij
    pair_weights = weights[:n_pairs]
    # Vectorized: 
    # iu[0] = i indices, iu[1] = j indices
    ii = iu[0]
    jj = iu[1]
    # diff_ij = c_i - c_j
    diff_ij = c[ii] - c[jj]  # shape (n_pairs, 3)
    # d(pair_dist)/dx_i = diff_ij / dist
    # d(pair_dist)/dx_j = -diff_ij / dist
    inv_dist = 1.0 / pair_dists  # shape (n_pairs,)
    # Contribution to grad for x_i (at index ii): -pair_weights * diff_ij * inv_dist
    # Accumulate
    # Use np.add.at
    contrib_i = -pair_weights[:, np.newaxis] * diff_ij * inv_dist[:, np.newaxis]
    contrib_j = -pair_weights[:, np.newaxis] * (-diff_ij) * inv_dist[:, np.newaxis]
    np.add.at(grad.reshape(n, 3), ii, contrib_i)
    np.add.at(grad.reshape(n, 3), jj, contrib_j)
    
    # Boundary part
    # b_i = min_k (0.5 - |c_ik|). The soft-min treats each boundary distance as separate? 
    # No, I put b (the min per sphere) into all_dists. But the gradient of b_i w.r.t. c is
    # only non-zero for the argmin coordinate. For the soft approximation, we should really
    # include ALL boundary distances (n*3 of them) in the soft-min, not just the min per sphere.
    # Let me redo: include all n*3 boundary distances.
    # I'll recompute more carefully below. For now, approximate: 
    # The boundary contribution: for each coordinate (i,k), d_{ik} = 0.5 - |c_ik|
    # d(d_{ik})/dx_ik = -sign(c_ik)
    # I'll handle boundaries by including all n*3 in all_dists. Let me restructure.
    
    # Actually, let me just return the value and a simple gradient for now,
    # and use a finite-difference fallback if needed. But L-BFGS-B needs gradient.
    # Let me redo the whole thing with all boundary distances included.
    
    return -soft_min_val, grad  # This is a placeholder; will fix

def _soft_min_obj_full(x, tau):
    """
    Full differentiable soft-min including ALL pairwise distances and ALL boundary distances.
    Minimizes -soft_min.
    """
    n = len(x) // 3
    c = x.reshape(n, 3)
    
    # Pairwise
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2) + 1e-15)
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist[iu]
    
    # Boundary: all n*3 distances
    boundary_dists = (0.5 - np.abs(c)).flatten()  # shape (n*3,)
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    n_pairs = len(pair_dists)
    n_bounds = len(boundary_dists)
    
    # Soft-min
    m = all_dists.min()
    exp_terms = np.exp(-(all_dists - m) / tau)
    S = exp_terms.sum()
    logsumexp = np.log(S)
    soft_min_val = m - tau * logsumexp
    
    weights = exp_terms / S  # d(soft_min)/d(all_dists) = weights
    # We minimize -soft_min, so grad of objective w.r.t. all_dists = -weights
    
    grad = np.zeros_like(x)
    
    # Pairwise gradient
    ii = iu[0]
    jj = iu[1]
    diff_ij = c[ii] - c[jj]
    inv_dist = 1.0 / pair_dists
    pair_w = weights[:n_pairs]
    contrib_i = -pair_w[:, np.newaxis] * diff_ij * inv_dist[:, np.newaxis]
    contrib_j = -pair_w[:, np.newaxis] * (-diff_ij) * inv_dist[:, np.newaxis]
    np.add.at(grad.reshape(n, 3), ii, contrib_i)
    np.add.at(grad.reshape(n, 3), jj, contrib_j)
    
    # Boundary gradient
    # boundary_dists = (0.5 - |c|).flatten()
    # d(0.5 - |c_ik|)/dx_ik = -sign(c_ik)
    bounds_w = weights[n_pairs:]
    # Reshape bounds_w to (n, 3)
    bounds_w_2d = bounds_w.reshape(n, 3)
    sign_c = np.sign(c)
    # grad contribution: -bounds_w * (-sign_c) = bounds_w * sign_c
    grad += bounds_w_2d * sign_c
    
    return -soft_min_val, grad

def _optimize_from(centres, tau_start=0.02, tau_end=0.001, max_time=10.0, deadline=None):
    """
    Run L-BFGS-B optimization from initial centres, annealing tau.
    Returns improved centres and radius.
    """
    start_time = time.time()
    c = centres.copy().astype(float)
    x = c.flatten()
    
    # Bounds
    eps = 1e-9
    bounds = [(-0.5 + eps, 0.5 - eps)] * (N * 3)
    
    best_c = c.copy()
    best_r = _exact_radius(c)
    
    # Anneal tau
    n_anneal = 8
    taus = np.geomspace(tau_start, tau_end, n_anneal)
    
    for tau in taus:
        if deadline is not None and time.time() > deadline:
            break
        elapsed = time.time() - start_time
        if elapsed > max_time:
            break
        
        def obj_grad(x):
            val, g = _soft_min_obj_full(x, tau)
            return val, g
        
        try:
            from scipy.optimize import minimize
            res = minimize(obj_grad, x, method='L-BFGS-B', jac=True,
                          bounds=bounds, options={'maxiter': 200, 'gtol': 1e-8,
                                                  'maxls': 50})
            x = res.x
            c = x.reshape(N, 3)
            r = _exact_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
        except Exception:
            pass
    
    return best_c, best_r

def _perturb(centres, rng, strength=0.01):
    """Randomly perturb centres."""
    c = centres.copy()
    c += rng.normal(0, strength, size=c.shape)
    c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
    return c

def _initial_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    # 100 = 5*5*4
    m1, m2, m3 = 5, 5, 4
    xs = (np.arange(m1) + 0.5) / m1 - 0.5
    ys = (np.arange(m2) + 0.5) / m2 - 0.5
    zs = (np.arange(m3) + 0.5) / m3 - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid[:N].copy()

def _initial_fcc():
    """FCC-inspired: interpenetrating cubic lattices."""
    # FCC: points at (0,0,0) and (0.5,0.5,0) type offsets on a cubic lattice
    # For a unit cube, use a 4x4x4 cubic lattice with FCC offsets
    m = 4
    base = (np.arange(m) + 0.5) / m - 0.5
    offsets = [np.array([0, 0, 0]),
               np.array([0.5/m, 0.5/m, 0]),
               np.array([0.5/m, 0, 0.5/m]),
               np.array([0, 0.5/m, 0.5/m])]
    points = []
    for ox, oy, oz in offsets:
        for z in base:
            for y in base:
                for x in base:
                    p = np.array([x + ox, y + oy, z + oz])
                    if np.all(np.abs(p) < 0.5):
                        points.append(p)
    points = np.array(points)
    # Take first N
    if len(points) >= N:
        return points[:N].copy()
    else:
        # Pad with random
        rng = np.random.RandomState(0)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(points), 3))
        return np.vstack([points, extra])

def _initial_jittered_grid(seed=0):
    rng = np.random.RandomState(seed)
    c = _initial_grid()
    c += rng.normal(0, 0.005, size=c.shape)
    c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)
    return c

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    rng = np.random.RandomState(12345)
    
    # Initializations
    inits = []
    inits.append(('grid', _initial_grid()))
    inits.append(('fcc', _initial_fcc()))
    for s in [0, 1, 2, 3, 7, 42]:
        inits.append((f'jit_{s}', _initial_jittered_grid(seed=s)))
    
    best_c = None
    best_r = -1.0
    
    for name, c0 in inits:
        if time.time() > deadline - 15:
            break
        
        # Optimize from this init
        remaining = deadline - time.time() - 5
        if remaining < 3:
            break
        c_opt, r_opt = _optimize_from(c0, tau_start=0.02, tau_end=0.001,
                                      max_time=min(remaining, 12.0), deadline=deadline)
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
        
        # Perturbation + re-optimize
        if time.time() > deadline - 10:
            break
        c_pert = _perturb(c_opt, rng, strength=0.008)
        remaining = deadline - time.time() - 3
        if remaining < 2:
            break
        c_opt2, r_opt2 = _optimize_from(c_pert, tau_start=0.015, tau_end=0.001,
                                        max_time=min(remaining, 8.0), deadline=deadline)
        if r_opt2 > best_r:
            best_r = r_opt2
            best_c = c_opt2.copy()
    
    # Final refinement on best
    if best_c is not None and time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            c_fin, r_fin = _optimize_from(best_c, tau_start=0.01, tau_end=0.0005,
                                          max_time=min(remaining, 6.0), deadline=deadline)
            if r_fin > best_r:
                best_r = r_fin
                best_c = c_fin.copy()
    
    if best_c is None:
        best_c = _initial_grid()
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-10, 0.5 - 1e-10)
    return best_c
