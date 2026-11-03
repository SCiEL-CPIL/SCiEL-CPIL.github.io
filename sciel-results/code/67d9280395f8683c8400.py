"""Packing 50 equal circles in a unit circle via smooth soft-min optimization with Adam."""
N = 50
import numpy as np


def _hex_lattice_init():
    """Generate N points from a hexagonal lattice inside the unit disk."""
    spacing = 0.22
    k = int(1.0 / spacing) + 2
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 0.95:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) < N:
        extra = np.random.uniform(-0.9, 0.9, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    # Shuffle and take N
    idx = np.random.permutation(len(pts))
    return pts[idx[:N]]


def _score(pts):
    """Compute packing score (largest common radius)."""
    d = np.sqrt((pts[:, 0:1] - pts[0:1, 0]) ** 2 + (pts[:, 1:2] - pts[0:2, 1]) ** 2)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_bound = np.min(1.0 - r)
    return min(min_pair, min_bound)


def _adam_step(pts, lr, m, v, t, beta1, beta2, eps):
    """One Adam step on pts (N,2). Returns updated pts, m, v."""
    # Compute objective gradient
    # Soft-min of [d_ij/2 for i<j, 1-|p_i| for i]
    T = 0.015
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:2, 1]
    d = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)
    np.fill_diagonal(d, 1e10)
    
    # Pairwise values
    iu = np.triu_indices(N, k=1)
    pair_vals = d[iu] / 2.0
    
    # Boundary values
    r = np.hypot(pts[:, 0], pts[:, 1])
    bound_vals = 1.0 - r
    
    all_vals = np.concatenate([pair_vals, bound_vals])
    
    # Soft-min: -T * log(sum(exp(-v/T)))
    neg = -all_vals / T
    neg = np.clip(neg, -80, 80)
    exp_neg = np.exp(neg)
    logsum = np.log(np.sum(exp_neg) + 1e-30)
    soft_min = -T * logsum
    obj = -soft_min
    
    # Gradient weights (softmax of -v/T)
    w = exp_neg / (np.sum(exp_neg) + 1e-30)
    
    # Gradient of obj w.r.t. all_vals = -w
    gw = -w
    
    # Accumulate gradient w.r.t. pts
    gpts = np.zeros_like(pts)
    
    # Pairwise part
    n_pair = len(iu)
    # For pair (i,j): value = d_ij/2
    # d(value)/d(pts_i) = 0.5 * (pts_j - pts_i) / d_ij
    # d(value)/d(pts_j) = 0.5 * (pts_i - pts_j) / d_ij
    for idx in range(n_pair):
        i, j = iu[0][idx], iu[1][idx]
        dij = d[i, j]
        if dij < 1e-8:
            continue
        coeff = gw[idx]
        diff = pts[i] - pts[j]
        gpts[i] += coeff * 0.5 * diff / dij
        gpts[j] += coeff * 0.5 * (-diff) / dij
    
    # Boundary part
    for i in range(N):
        ri = r[i]
        if ri < 1e-8:
            continue
        gpts[i] += gw[n_pair + i] * (-pts[i] / ri)
    
    # Adam update
    m = beta1 * m + (1 - beta1) * gpts
    v = beta2 * v + (1 - beta2) * (gpts ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    pts_new = pts - lr * m_hat / (np.sqrt(v_hat) + eps)
    return pts_new, m, v


def construct():
    """
    Place 50 equal, non-overlapping circles inside the unit circle.
    Returns an array of shape (N, 2) with the centres.
    """
    best_pts = None
    best_score = -np.inf
    
    # Multiple restarts
    for restart in range(6):
        if restart == 0:
            pts = _hex_lattice_init()
        else:
            pts = _hex_lattice_init()
            # Perturb
            pts = pts + np.random.normal(0, 0.05, size=(N, 2))
            # Project inside disk
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 0.95
            if np.any(mask):
                pts[mask] *= 0.95 / r[mask, None]
        
        # Adam hyperparameters
        lr = 0.003
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = np.zeros_like(pts)
        v = np.zeros_like(pts)
        
        # Animate temperature: start high (smooth) -> low (sharp)
        n_iter = 3000
        for t in range(1, n_iter + 1):
            # Schedule temperature
            frac = t / n_iter
            T = 0.04 * (1.0 - frac) + 0.005 * frac
            
            # Recompute gradient with current T (inline for efficiency)
            dx = pts[:, 0:1] - pts[0:1, 0]
            dy = pts[:, 1:2] - pts[0:2, 1]
            d = np.sqrt(dx ** 2 + dy ** 2 + 1e-12)
            np.fill_diagonal(d, 1e10)
            
            iu = np.triu_indices(N, k=1)
            pair_vals = d[iu] / 2.0
            r = np.hypot(pts[:, 0], pts[:, 1])
            bound_vals = 1.0 - r
            all_vals = np.concatenate([pair_vals, bound_vals])
            
            neg = -all_vals / T
            neg = np.clip(neg, -80, 80)
            exp_neg = np.exp(neg)
            logsum = np.log(np.sum(exp_neg) + 1e-30)
            soft_min = -T * logsum
            obj = -soft_min
            
            w = exp_neg / (np.sum(exp_neg) + 1e-30)
            gw = -w
            
            gpts = np.zeros_like(pts)
            n_pair = len(iu)
            for idx in range(n_pair):
                i, j = iu[0][idx], iu[1][idx]
                dij = d[i, j]
                if dij < 1e-8:
                    continue
                coeff = gw[idx]
                diff = pts[i] - pts[j]
                gpts[i] += coeff * 0.5 * diff / dij
                gpts[j] += coeff * 0.5 * (-diff) / dij
            
            for i in range(N):
                ri = r[i]
                if ri < 1e-8:
                    continue
                gpts[i] += gw[n_pair + i] * (-pts[i] / ri)
            
            # Adam update
            m = beta1 * m + (1 - beta1) * gpts
            v = beta2 * v + (1 - beta2) * (gpts ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            lr_t = lr * (0.5 + 0.5 * (1 - frac))  # decay learning rate
            pts = pts - lr_t * m_hat / (np.sqrt(v_hat) + eps)
            
            # Project inside unit disk
            r_new = np.hypot(pts[:, 0], pts[:, 1])
            mask = r_new > 0.999
            if np.any(mask):
                pts[mask] *= 0.999 / r_new[mask, None]
        
        # Evaluate
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _hex_lattice_init()
    
    # Final safety: ensure inside disk
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] *= 0.9999 / r[mask, None]
    
    return best_pts
