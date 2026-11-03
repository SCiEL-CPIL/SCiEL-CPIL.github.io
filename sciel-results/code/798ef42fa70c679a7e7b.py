"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_init(n, r):
    """Hexagonal lattice init for n points with circle radius r in unit disk."""
    dx = 2.0 * r
    dy = dx * np.sqrt(3.0) / 2.0
    k = int(1.0 / dx) + 3
    xs, ys = [], []
    for j in range(-k, k + 1):
        y = j * dy
        if abs(y) > 1.0 - r:
            continue
        offset = 0.5 * dx if (j % 2) else 0.0
        for i in range(-k, k + 1):
            x = i * dx + offset
            if x * x + y * y <= (1.0 - r) ** 2:
                xs.append(x)
                ys.append(y)
    if len(xs) < n:
        return None
    pts = np.column_stack([xs, ys])
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d, kind="stable")[:n]
    return pts[idx]


def _compute_radius(p):
    """Compute the common radius: min over pair distances / 2 and container constraints."""
    # Pairwise distances
    n = p.shape[0]
    # Use chunked computation to avoid memory issues for large n
    min_pair = np.inf
    chunk = 200
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = p[start:end, :][:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        # Zero out diagonal
        np.fill_diagonal(dist, np.inf)
        min_pair = min(min_pair, np.min(dist))
    # Container constraint
    norms = np.linalg.norm(p, axis=1)
    min_container = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_container)


def _soft_min(values, beta):
    """Soft-min using log-sum-exp."""
    min_val = np.min(values)
    return min_val - np.log(np.sum(np.exp(-beta * (values - min_val)))) / beta


def _soft_radius(p, beta):
    """Soft-min surrogate for the radius objective."""
    n = p.shape[0]
    # Pairwise soft-min
    min_pair = np.inf
    chunk = 200
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = p[start:end, :][:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_soft = _soft_min(dist / 2.0, beta)
        min_pair = min(min_pair, pair_soft)
    # Container soft-min
    norms = np.linalg.norm(p, axis=1)
    container_soft = _soft_min(1.0 - norms, beta)
    return min(min_pair, container_soft)


def _soft_radius_grad(p, beta):
    """Gradient of the soft-radius objective."""
    n = p.shape[0]
    grad = np.zeros_like(p)
    
    # Pairwise contribution
    chunk = 200
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = p[start:end, :][:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Soft-min weights
        min_val = np.min(dist / 2.0)
        exp_terms = np.exp(-beta * (dist / 2.0 - min_val))
        exp_terms[~np.isfinite(exp_terms)] = 0
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of soft-min of dist/2
        # d(soft_min)/d(p_i) = sum_j w_j * d(dist_ij/2)/d(p_i)
        # d(dist_ij)/d(p_i) = (p_i - p_j) / dist_ij
        for i in range(start, end):
            local_i = i - start
            for j in range(n):
                if i == j:
                    continue
                w = weights[local_i, j]
                if w < 1e-15:
                    continue
                d = dist[local_i, j]
                if d < 1e-12:
                    continue
                grad[i] += w * (p[i] - p[j]) / d
    
    # Container contribution
    norms = np.linalg.norm(p, axis=1)
    min_c = np.min(1.0 - norms)
    exp_c = np.exp(-beta * (1.0 - norms - min_c))
    exp_c[~np.isfinite(exp_c)] = 0
    weights_c = exp_c / np.sum(exp_c)
    for i in range(n):
        if norms[i] < 1e-12:
            continue
        grad[i] -= weights_c[i] * p[i] / norms[i]
    
    return grad


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin
    
    # Initial hexagonal packing
    r_init = 0.053
    p = _hex_init(N, r_init)
    if p is None:
        r_init = 0.05
        p = _hex_init(N, r_init)
    if p is None:
        # Fallback: random placement
        p = np.random.randn(N, 2) * 0.8
        return p
    
    best_p = p.copy()
    best_r = _compute_radius(best_p)
    
    # Phase 1: Gradient ascent with soft-min surrogate
    beta = 50.0
    lr = 0.001
    step = 0
    while time.time() < deadline - 30.0:
        r_soft = _soft_radius(p, beta)
        if r_soft > best_r:
            best_r = r_soft
            best_p = p.copy()
        
        grad = _soft_radius_grad(p, beta)
        norm = np.linalg.norm(grad)
        if norm < 1e-10:
            break
        
        # Adaptive learning rate
        p_new = p + lr * grad / norm
        r_new = _soft_radius(p_new, beta)
        
        if r_new > r_soft:
            p = p_new
            lr *= 1.05
        else:
            lr *= 0.9
            if lr < 1e-6:
                break
        
        step += 1
        if step % 50 == 0:
            # Update best
            r_exact = _compute_radius(p)
            if r_exact > best_r:
                best_r = r_exact
                best_p = p.copy()
    
    # Phase 2: Polish with sharper soft-min
    p = best_p.copy()
    beta2 = 200.0
    lr2 = 0.0005
    for _ in range(200):
        if time.time() > deadline - 10.0:
            break
        r_soft = _soft_radius(p, beta2)
        grad = _soft_radius_grad(p, beta2)
        norm = np.linalg.norm(grad)
        if norm < 1e-10:
            break
        p_new = p + lr2 * grad / norm
        r_new = _soft_radius(p_new, beta2)
        if r_new > r_soft:
            p = p_new
        else:
            lr2 *= 0.8
    
    # Final exact radius
    r_final = _compute_radius(p)
    if r_final > best_r:
        best_p = p.copy()
        best_r = r_final
    
    # Ensure validity: clip to container if needed
    norms = np.linalg.norm(best_p, axis=1)
    mask = norms > 1.0 - best_r
    if np.any(mask):
        best_p[mask] *= (1.0 - best_r) / norms[mask]
    
    return best_p
# EVOLVE-BLOCK-END
