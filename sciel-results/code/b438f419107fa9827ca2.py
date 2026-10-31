"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3.0)
H = S3 / 2.0

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Time budget
DEADLINE = 140.0  # seconds, with 10s safety margin


def _barycentric_to_cartesian(b):
    """Convert barycentric coords (u, v) with u+v<=1 to Cartesian."""
    # b = (u, v), point = u*V1 + v*V2
    x = b[:, 0] * 1.0 + b[:, 1] * 0.5
    y = b[:, 1] * H
    return np.column_stack([x, y])


def _cartesian_to_barycentric(pts):
    """Convert Cartesian to barycentric (u, v) where point = u*V1 + v*V2."""
    v = pts[:, 1] / H
    u = pts[:, 0] - 0.5 * v
    return np.column_stack([u, v])


def _all_triangle_areas_barycentric(b):
    """
    Compute areas of all triangles formed by triples of points, in barycentric coords.
    Points are (u, v) with u+v<=1. The area of triangle (i,j,k) in Cartesian is
    |det([p_j - p_i, p_k - p_i])| / 2 where p = u*V1 + v*V2.
    
    Since V1=(1,0), V2=(0.5,H), the mapping from (u,v) to (x,y) is linear:
    x = u + 0.5*v, y = H*v
    det = (u_j + 0.5*v_j - u_i - 0.5*v_i) * (H*(v_k - v_i)) - (H*(v_j - v_i)) * (u_k + 0.5*v_k - u_i - 0.5*v_i)
    
    Let du_j = u_j - u_i, dv_j = v_j - v_i, etc.
    x_diff = du + 0.5*dv
    y_diff = H*dv
    det = (du_j + 0.5*dv_j)*(H*dv_k) - (H*dv_j)*(du_k + 0.5*dv_k)
         = H * [dv_j*dv_k + 0.5*du_j*dv_k - 0.5*dv_j*du_k - 0.25*dv_j*dv_k]
         = H * [0.75*dv_j*dv_k + 0.5*du_j*dv_k - 0.5*dv_j*du_k]
         = H * [0.5*(du_j*dv_k - dv_j*du_k) + 0.75*dv_j*dv_k]
    
    Area = |det| / 2
    """
    n = len(b)
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(i + 1, n):
            du_j = b[j, 0] - b[i, 0]
            dv_j = b[j, 1] - b[i, 1]
            for k in range(j + 1, n):
                du_k = b[k, 0] - b[i, 0]
                dv_k = b[k, 1] - b[i, 1]
                det = H * (0.5 * (du_j * dv_k - dv_j * du_k) + 0.75 * dv_j * dv_k)
                a = abs(det) / 2.0
                areas[i, j, k] = a
                areas[j, i, k] = a
                areas[i, k, j] = a
                areas[j, k, i] = a
                areas[k, i, j] = a
                areas[k, j, i] = a
    return areas


def _min_area(b):
    """Compute minimum triangle area for barycentric points."""
    n = len(b)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            du_j = b[j, 0] - b[i, 0]
            dv_j = b[j, 1] - b[i, 1]
            for k in range(j + 1, n):
                du_k = b[k, 0] - b[i, 0]
                dv_k = b[k, 1] - b[i, 1]
                det = H * (0.5 * (du_j * dv_k - dv_j * du_k) + 0.75 * dv_j * dv_k)
                a = abs(det) / 2.0
                if a < min_a:
                    min_a = a
    return min_a


def _soft_min_area(b, beta=50.0):
    """Soft-min surrogate using log-sum-exp for differentiability."""
    n = len(b)
    log_sum = np.zeros(())
    max_log = -np.inf
    # Collect all log areas
    logs = []
    for i in range(n):
        for j in range(i + 1, n):
            du_j = b[j, 0] - b[i, 0]
            dv_j = b[j, 1] - b[i, 1]
            for k in range(j + 1, n):
                du_k = b[k, 0] - b[i, 0]
                dv_k = b[k, 1] - b[i, 1]
                det = H * (0.5 * (du_j * dv_k - dv_j * du_k) + 0.75 * dv_j * dv_k)
                a = max(abs(det) / 2.0, 1e-12)
                logs.append(np.log(a))
    logs = np.array(logs)
    max_log = np.max(logs)
    # log-sum-exp: max + log(sum(exp(logs - max)))
    lse = max_log + np.log(np.sum(np.exp(logs - max_log)))
    # soft min = -1/beta * log(sum(exp(-beta * a))) = -1/beta * log(sum(exp(-beta * exp(log_a))))
    # Actually: soft_min(a, beta) = -1/beta * log(sum(exp(-beta * a)))
    # = -1/beta * log(sum(exp(-beta * exp(log_a))))
    # Let's compute directly
    exp_neg_beta_a = np.exp(-beta * np.exp(logs))
    soft_min = -np.log(np.sum(exp_neg_beta_a)) / beta
    return soft_min


def _soft_min_area_vec(b, beta=50.0):
    """Vectorized soft-min area computation."""
    n = len(b)
    # Build all differences
    # For each triple (i,j,k), compute area
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            du_j = b[j, 0] - b[i, 0]
            dv_j = b[j, 1] - b[i, 1]
            for k in range(j + 1, n):
                du_k = b[k, 0] - b[i, 0]
                dv_k = b[k, 1] - b[i, 1]
                det = H * (0.5 * (du_j * dv_k - dv_j * du_k) + 0.75 * dv_j * dv_k)
                a = abs(det) / 2.0
                areas.append(a)
    areas = np.array(areas)
    areas = np.maximum(areas, 1e-12)
    # soft min
    neg_beta_a = -beta * areas
    max_val = np.max(neg_beta_a)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_a - max_val)))
    soft_min = -log_sum / beta
    return soft_min


def _soft_min_grad(b, beta=50.0):
    """Compute soft-min area and its gradient w.r.t. barycentric coords."""
    n = len(b)
    areas = []
    # Store (i, j, k, area, det_sign, du_j, dv_j, du_k, dv_k)
    triples = []
    for i in range(n):
        for j in range(i + 1, n):
            du_j = b[j, 0] - b[i, 0]
            dv_j = b[j, 1] - b[i, 1]
            for k in range(j + 1, n):
                du_k = b[k, 0] - b[i, 0]
                dv_k = b[k, 1] - b[i, 1]
                det = H * (0.5 * (du_j * dv_k - dv_j * du_k) + 0.75 * dv_j * dv_k)
                a = abs(det) / 2.0
                triples.append((i, j, k, a, det, du_j, dv_j, du_k, dv_k))
    
    areas_arr = np.array([t[3] for t in triples])
    areas_arr = np.maximum(areas_arr, 1e-12)
    
    # weights for soft min: w_m = exp(-beta * a_m) / sum(exp(-beta * a))
    neg_beta_a = -beta * areas_arr
    max_val = np.max(neg_beta_a)
    exp_vals = np.exp(neg_beta_a - max_val)
    weights = exp_vals / np.sum(exp_vals)
    
    soft_min = -np.log(np.sum(exp_vals)) / beta - max_val / beta
    # Actually: -1/beta * log(sum(exp(-beta*a))) = -1/beta * (max_val + log(sum(exp(neg_beta_a - max_val))))
    soft_min = -(max_val + np.log(np.sum(exp_vals))) / beta
    
    # Gradient: d(soft_min)/d(b) = -1/beta * sum_m w_m * (-beta * d(a_m)/d(b)) = sum_m w_m * d(a_m)/d(b)
    # Wait: soft_min = -1/beta * log(sum(exp(-beta * a_m)))
    # d(soft_min)/d(x) = -1/beta * (1/sum) * sum(exp(-beta*a_m) * (-beta) * d(a_m)/d(x))
    # = sum(exp(-beta*a_m) * d(a_m)/d(x)) / sum(exp(-beta*a_m))
    # = sum(w_m * d(a_m)/d(x))
    
    grad = np.zeros_like(b)
    for idx, (i, j, k, a, det, du_j, dv_j, du_k, dv_k) in enumerate(triples):
        w = weights[idx]
        # a = |det|/2, det = H*(0.5*(du_j*dv_k - dv_j*du_k) + 0.75*dv_j*dv_k)
        # da/d(b) = sign(det)/2 * d(det)/d(b)
        sign = 1.0 if det >= 0 else -1.0
        # d(det)/d(b[i,0]) = 0 (i is reference)
        # d(det)/d(b[j,0]): du_j depends on b[j,0], dv_j doesn't
        # det = H*(0.5*(du_j*dv_k - dv_j*du_k) + 0.75*dv_j*dv_k)
        # d(det)/du_j = H*0.5*dv_k
        # d(det)/dv_j = H*(-0.5*du_k + 0.75*dv_k)
        # d(det)/du_k = H*(-0.5*dv_j)
        # d(det)/dv_k = H*(0.5*du_j + 0.75*dv_j)
        
        dad_uj = H * 0.5 * dv_k
        dad_vj = H * (-0.5 * du_k + 0.75 * dv_k)
        dad_uk = H * (-0.5 * dv_j)
        dad_vk = H * (0.5 * du_j + 0.75 * dv_j)
        
        # da/d(b[j,0]) = sign/2 * dad_uj
        # da/d(b[j,1]) = sign/2 * dad_vj
        # da/d(b[k,0]) = sign/2 * dad_uk
        # da/d(b[k,1]) = sign/2 * dad_vk
        # da/d(b[i,0]) = -sign/2 * (dad_uj + dad_uk)  [since du_j, du_k depend on b[i,0] with -1]
        # da/d(b[i,1]) = -sign/2 * (dad_vj + dad_vk)
        
        coeff = sign / 2.0 * w
        grad[j, 0] += coeff * dad_uj
        grad[j, 1] += coeff * dad_vj
        grad[k, 0] += coeff * dad_uk
        grad[k, 1] += coeff * dad_vk
        grad[i, 0] -= coeff * (dad_uj + dad_uk)
        grad[i, 1] -= coeff * (dad_vj + dad_vk)
    
    return soft_min, grad


def _project_to_feasible(b):
    """Project barycentric coords to feasible region: u>=0, v>=0, u+v<=1."""
    b = b.copy()
    b[:, 0] = np.maximum(b[:, 0], 0.0)
    b[:, 1] = np.maximum(b[:, 1], 0.0)
    # Project onto u+v <= 1
    s = b[:, 0] + b[:, 1]
    mask = s > 1.0
    if np.any(mask):
        excess = s[mask] - 1.0
        # Distribute excess proportionally
        u = b[mask, 0]
        v = b[mask, 1]
        total = u + v
        b[mask, 0] = u - excess * (u / total)
        b[mask, 1] = v - excess * (v / total)
        b[mask, 0] = np.maximum(b[mask, 0], 0.0)
        b[mask, 1] = np.maximum(b[mask, 1], 0.0)
        # Re-check
        s2 = b[:, 0] + b[:, 1]
        mask2 = s2 > 1.0
        if np.any(mask2):
            b[mask2, 0] = np.maximum(b[mask2, 0], 0.0)
            b[mask2, 1] = 1.0 - b[mask2, 0]
    return b


def _sa_global(b_init, t_start=0.05, t_end=0.001, n_iter=5000, seed=42):
    """Simulated annealing in barycentric space."""
    rng = np.random.default_rng(seed)
    b = b_init.copy()
    b = _project_to_feasible(b)
    best_b = b.copy()
    best_val = _min_area(b)
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / n_iter)
    
    for it in range(n_iter):
        # Random perturbation
        delta = rng.normal(0, 1, size=b.shape) * (0.02 + 0.03 * t / t_start)
        b_new = b + delta
        b_new = _project_to_feasible(b_new)
        
        val_new = _min_area(b_new)
        val_old = _min_area(b)
        
        if val_new > val_old or rng.random() < np.exp((val_new - val_old) / t):
            b = b_new
            if val_new > best_val:
                best_val = val_new
                best_b = b.copy()
        
        t *= cooling
    
    return best_b, best_val


def _gradient_polish(b, max_iter=100, beta=50.0):
    """Gradient-based polishing with projected gradient ascent on soft-min area."""
    b = b.copy()
    for it in range(max_iter):
        val, grad = _soft_min_grad(b, beta)
        # Gradient ascent: move in direction of increasing soft-min area
        # Use adaptive step size
        step = 0.01
        for _ in range(10):
            b_new = b + step * grad
            b_new = _project_to_feasible(b_new)
            val_new = _min_area(b_new)
            if val_new > _min_area(b):
                b = b_new
                break
            step *= 0.5
        if step < 1e-8:
            break
    return b, _min_area(b)


def _multi_start_sa(n_starts=8, seed=0):
    """Run SA from multiple random starts, return best result."""
    best_b = None
    best_val = -1
    rng = np.random.default_rng(seed)
    
    for s in range(n_starts):
        # Random start
        b = np.random.default_rng(seed + s).uniform(0, 1, size=(N, 2))
        b = _project_to_feasible(b)
        # Scale down to be more interior
        b = b * 0.8
        
        b_res, val_res = _sa_global(b, t_start=0.05, t_end=0.001, n_iter=3000, seed=seed + s)
        # Polish
        b_pol, val_pol = _gradient_polish(b_res, max_iter=50, beta=50.0)
        if val_pol > best_val:
            best_val = val_pol
            best_b = b_pol.copy()
    
    return best_b, best_val


def construct():
    start_time = time.time()
    
    # Phase 1: Multi-start SA
    b_best, val_best = _multi_start_sa(n_starts=6, seed=12345)
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 30:
        return _barycentric_to_cartesian(b_best)
    
    # Phase 2: More SA iterations from best
    b_best, val_best = _sa_global(b_best, t_start=0.03, t_end=0.0005, n_iter=5000, seed=99999)
    b_best, val_best = _gradient_polish(b_best, max_iter=100, beta=50.0)
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 15:
        return _barycentric_to_cartesian(b_best)
    
    # Phase 3: Re-explore with SA from perturbed best
    rng = np.random.default_rng(777)
    for _ in range(3):
        b_pert = b_best + rng.normal(0, 0.02, size=b_best.shape)
        b_pert = _project_to_feasible(b_pert)
        b_res, val_res = _sa_global(b_pert, t_start=0.02, t_end=0.0005, n_iter=3000, seed=int(rng.integers(1e9)))
        b_res, val_res = _gradient_polish(b_res, max_iter=80, beta=50.0)
        if val_res > val_best:
            val_best = val_res
            b_best = b_res.copy()
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
    
    # Phase 4: Final high-precision polish
    b_best, val_best = _gradient_polish(b_best, max_iter=200, beta=100.0)
    
    return _barycentric_to_cartesian(b_best)
