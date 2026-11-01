"""Circle packing: 50 equal circles in unit disk. Smooth surrogate gradient ascent + contact graph local search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 135.0  # safety margin before 150s kill

def _compute_score(pts):
    """Return the packing radius (min of pairwise/2 and boundary distances)."""
    n = len(pts)
    if n < 2:
        return 0.0
    # boundary distances
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    if np.any(boundary < -1e-12):
        return 0.0  # invalid: outside container
    # pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(boundary))

def _softmin(vals, beta):
    """Soft-min approximation of min(vals) using log-sum-exp."""
    neg = -vals
    m = np.max(neg)
    if m > 1e12:
        return m
    exp_vals = np.exp(neg - m)
    log_sum = np.log(np.sum(exp_vals)) + m
    return -log_sum / beta

def _softmax(vals, beta):
    """Soft-max approximation of max(vals) using log-sum-exp."""
    m = np.max(vals)
    if m > 1e12:
        return m
    exp_vals = np.exp(vals - m)
    log_sum = np.log(np.sum(exp_vals)) + m
    return log_sum / beta

def _objective_and_grad(pts, beta):
    """
    Objective: maximize the soft-min of (pairwise distances / 2, boundary clearances).
    We minimize the negative of this.
    
    Returns (value, gradient) where value is -softmin and gradient is -d(softmin)/dx.
    """
    n = len(pts)
    pts = pts.reshape(n, 2)
    
    # Compute all pairwise distances and boundary clearances
    # Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_half = dists / 2.0
    
    # Boundary
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    # Combine into one vector for soft-min
    # We want soft-min of [pair_half[i,j] for i<j, boundary[i]]
    # But to compute gradient efficiently, we handle them separately and combine weights
    
    # Soft-min of pair_half
    # soft-min(A) = -soft-max(-A)
    # Let's compute soft-max of -pair_half (upper triangle only to avoid double counting)
    # Actually, let's just use all values and the soft-min will naturally weight the smallest
    
    # For efficiency and correctness, let's compute the soft-min over the union of:
    # 1. All pairwise half-distances (i < j)
    # 2. All boundary clearances
    
    # Get indices for upper triangle
    iu, ju = np.triu_indices(n, k=1)
    pair_vals = pair_half[iu, ju]
    
    # All values for soft-min
    all_vals = np.concatenate([pair_vals, boundary])
    
    # Compute soft-min
    # soft-min = -1/beta * log(sum(exp(-beta * val)))
    neg_beta_vals = -beta * all_vals
    m = np.max(neg_beta_vals)
    if m > 1e12:
        sm = m
    else:
        exp_vals = np.exp(neg_beta_vals - m)
        log_sum = np.log(np.sum(exp_vals)) + m
        sm = -log_sum / beta
    
    # Gradient of soft-min w.r.t. pts
    # d(sm)/d(val_k) = -exp(-beta * val_k) / sum(exp(-beta * val_j))
    # We need d(pair_vals)/d(pts) and d(boundary)/d(pts)
    
    # Weights for each value in all_vals
    weights = np.exp(neg_beta_vals - m)
    weights /= np.sum(weights)
    
    # Gradient of pair_half[i,j] w.r.t. pts[i] and pts[j]
    # pair_half[i,j] = 0.5 * ||pts[i] - pts[j]||
    # d/d(pts[i]) = 0.5 * (pts[i] - pts[j]) / ||pts[i] - pts[j]||
    # d/d(pts[j]) = 0.5 * (pts[j] - pts[i]) / ||pts[i] - pts[j]||
    
    grad = np.zeros_like(pts)
    
    # Pairwise contributions
    # For each (i, j) in upper triangle, weight is weights[k] where k is index in pair_vals
    for k in range(len(pair_vals)):
        i, j = iu[k], ju[k]
        w = weights[k]
        d = pts[i] - pts[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            continue
        # d(pair_half)/d(pts[i]) = 0.5 * d / dist
        # d(pair_half)/d(pts[j]) = 0.5 * (-d) / dist
        # d(sm)/d(pts[i]) = -w * 0.5 * d / dist
        # d(sm)/d(pts[j]) = -w * (-0.5 * d / dist) = w * 0.5 * d / dist
        factor = 0.5 * w / dist
        grad[i] -= factor * d
        grad[j] += factor * d
    
    # Boundary contributions
    # boundary[i] = 1 - ||pts[i]||
    # d(boundary[i])/d(pts[i]) = -pts[i] / ||pts[i]||
    for i in range(n):
        w = weights[len(pair_vals) + i]
        r = norms[i]
        if r < 1e-12:
            continue
        # d(sm)/d(pts[i]) = -w * (-pts[i] / r) = w * pts[i] / r
        factor = w / r
        grad[i] += factor * pts[i]
    
    # We are minimizing -sm, so objective = -sm, gradient = -grad_sm
    value = -sm
    grad_obj = -grad
    
    return value, grad_obj

def _gradient_ascent(pts, beta_start=10.0, beta_end=10000.0, steps=5000, lr_start=0.01, lr_end=0.0001):
    """
    Gradient ascent on the soft-min objective.
    Uses annealing of beta (sharpness) and learning rate.
    """
    pts = pts.copy().reshape(N, 2)
    n_steps = steps
    
    for step in range(n_steps):
        if time.time() > _DEADLINE - 5:
            break
        
        # Anneal beta and lr
        t = step / max(n_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        lr = lr_start * (lr_end / lr_start) ** t
        
        # Compute objective and gradient
        val, grad = _objective_and_grad(pts, beta)
        
        # Update: gradient ascent on sm = gradient descent on -sm
        # pts = pts + lr * grad (since grad is d(sm)/d(pts))
        # Wait, _objective_and_grad returns gradient of -sm.
        # So grad_obj = -d(sm)/d(pts).
        # We want to maximize sm, so move in direction +d(sm)/d(pts) = -grad_obj.
        pts -= lr * grad  # grad is grad_obj = -d(sm)/d(pts), so -lr*grad = lr*d(sm)/d(pts)
        
        # Project back to unit disk
        norms = np.linalg.norm(pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] /= norms[mask]
    
    return pts

def _contact_graph_local_search(pts, max_iter=2000, tol=1e-8):
    """
    Local search on the contact graph.
    Identifies the tightest constraints (contacts) and optimizes only those.
    """
    pts = pts.copy().reshape(N, 2)
    best_score = _compute_score(pts)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        # Compute current score
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
        
        # Find the tightest constraints
        # Boundary
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        
        # Pairwise
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_half = dists / 2.0
        
        # Find the minimum few
        # We'll pick the 5 tightest pairwise and 3 tightest boundary
        iu, ju = np.triu_indices(N, k=1)
        pair_vals = pair_half[iu, ju]
        
        # Get indices of smallest pair values
        pair_idx = np.argsort(pair_vals)[:5]
        tight_pairs = [(iu[p], ju[p]) for p in pair_idx]
        
        # Get indices of smallest boundary values
        bound_idx = np.argsort(boundary)[:3]
        tight_bounds = list(bound_idx)
        
        # If we've converged, break
        min_pair = np.min(pair_vals)
        min_bound = np.min(boundary)
        min_val = min(min_pair, min_bound)
        
        if min_val < tol:
            break
        
        # Perturb points involved in tight constraints
        # Try small random moves to break symmetry
        moved = False
        for i, j in tight_pairs:
            # Try moving i and j apart
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(2) * 1e-10
                dist = np.linalg.norm(d)
            dir_i = d / dist
            dir_j = -d / dist
            
            # Small step
            step = 0.001
            new_i = pts[i] + step * dir_i
            new_j = pts[j] + step * dir_j
            
            # Check if valid
            ni_norm = np.linalg.norm(new_i)
            nj_norm = np.linalg.norm(new_j)
            if ni_norm <= 1.0 and nj_norm <= 1.0:
                # Check if it improves local score
                old_local = min(pair_half[i, j], boundary[i], boundary[j])
                new_d = np.linalg.norm(new_i - new_j) / 2.0
                new_bi = 1.0 - ni_norm
                new_bj = 1.0 - nj_norm
                new_local = min(new_d, new_bi, new_bj)
                
                if new_local > old_local:
                    pts[i] = new_i
                    pts[j] = new_j
                    moved = True
        
        for i in tight_bounds:
            # Move inward
            r = norms[i]
            if r > 1e-10:
                dir_in = -pts[i] / r
                new_p = pts[i] + 0.001 * dir_in
                new_r = np.linalg.norm(new_p)
                if new_r <= 1.0:
                    old_b = boundary[i]
                    new_b = 1.0 - new_r
                    if new_b > old_b:
                        pts[i] = new_p
                        moved = True
        
        if not moved:
            break
    
    return pts.reshape(N, 2)

def _hex_lattice_init(n):
    """Generate a hexagonal lattice point cloud and pick n points closest to center."""
    best_pts = None
    for spacing in np.linspace(0.12, 0.25, 40):
        k = int(2.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            r = np.linalg.norm(pts, axis=1)
            idx = np.argsort(r, kind="stable")[:n]
            best_pts = pts[idx]
            break
    if best_pts is None:
        # fallback: random
        theta = np.random.uniform(0, 2 * np.pi, n)
        rho = np.sqrt(np.random.uniform(0, 1, n))
        best_pts = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
    return best_pts

def _golden_angle_init(n):
    """Generate points using golden angle spiral."""
    pts = np.zeros((n, 2))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        r = np.sqrt((i + 0.5) / n) * 0.95
        theta = i * golden_angle
        pts[i, 0] = r * np.cos(theta)
        pts[i, 1] = r * np.sin(theta)
    return pts

def _ring_init(n):
    """Generate points in concentric rings."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    remaining = n - 1
    ring = 1
    while remaining > 0:
        # Approximate number of points on ring
        circumference = 2 * np.pi * (ring * 0.15)
        count = min(remaining, max(1, int(circumference / 0.15)))
        for i in range(count):
            theta = 2 * np.pi * i / count
            r = ring * 0.15
            pts.append([r * np.cos(theta), r * np.sin(theta)])
            remaining -= 1
            if remaining <= 0:
                break
        ring += 1
    return np.array(pts[:n])

def construct():
    t_start = time.time()
    deadline = t_start + 135.0  # 15s safety margin
    
    # Try multiple initializations
    initializers = [
        _hex_lattice_init,
        _golden_angle_init,
        _ring_init
    ]
    
    best_pts = None
    best_score = -np.inf
    
    for init_func in initializers:
        if time.time() > deadline - 20:
            break
        
        pts = init_func(N)
        
        # Gradient ascent with annealing
        if time.time() < deadline - 10:
            pts = _gradient_ascent(pts, beta_start=10.0, beta_end=10000.0, steps=3000, lr_start=0.01, lr_end=0.0001)
        
        # Contact graph local search
        if time.time() < deadline - 5:
            pts = _contact_graph_local_search(pts, max_iter=1000)
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Final polish with Nelder-Mead on the best result
    if time.time() < deadline - 2 and best_pts is not None:
        x0 = best_pts.flatten()
        
        def obj(x):
            p = x.reshape(N, 2)
            # Check validity
            norms = np.linalg.norm(p, axis=1)
            if np.any(norms > 1.0):
                return 10.0
            diff = p[:, None, :] - p[None, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            min_bound = np.min(1.0 - norms)
            return -min(min_pair, min_bound)
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < -best_score:
                candidate = result.x.reshape(N, 2)
                # Verify validity
                norms = np.linalg.norm(candidate, axis=1)
                if np.all(norms <= 1.0):
                    best_pts = candidate
                    best_score = -result.fun
        except Exception:
            pass
    
    # Ensure validity
    if best_pts is not None:
        r = np.linalg.norm(best_pts, axis=1)
        if np.any(r > 1.0):
            best_pts[r > 1.0] *= 1.0 / r[r > 1.0]
    
    if best_pts is None:
        best_pts = _hex_lattice_init(N)
    
    return best_pts
