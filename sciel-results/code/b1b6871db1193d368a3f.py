"""Iterative radius increase with smooth energy minimization for sphere packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers, dtype=float)
    if len(centers) < 2:
        return float(np.min(0.5 - np.abs(centers)))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    min_wall_dist = np.min(0.5 - np.abs(centers))
    return float(min(min_pair_dist / 2.0, min_wall_dist))

def _energy_and_gradient(centers, target_r, lam=50.0):
    """
    Smooth energy: penalize violations of target radius.
    E = lam * sum_{i<j} max(0, target_r - d_ij/2)^2 + lam * sum_i max(0, target_r - w_i)^2
    where d_ij = |c_i - c_j|, w_i = min_k(0.5 - |c_ik|)
    
    We use a soft version: replace max(0,x)^2 with x^2 * sigmoid(beta*x) for smoothness.
    """
    centers = np.asarray(centers, dtype=float)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pair constraints: want d_ij/2 >= target_r, i.e., d_ij >= 2*target_r
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    dists = np.sqrt(sq_dists)
    
    # For each pair, violation = target_r - d_ij/2 = (2*target_r - d_ij)/2
    # We penalize when d_ij < 2*target_r
    # Smooth penalty: use softplus-like: f(x) = log(1+exp(beta*x))/beta for x>0
    # Or simpler: use quadratic with smooth gate
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    # violation amount: how much we need to increase distance
    # v = 2*target_r - d_ij (positive when violated)
    violations = 2.0 * target_r - pair_dists
    
    # Smooth: only penalize when violated, use soft gate
    beta_gate = 20.0
    # gate = sigmoid(beta_gate * violations) ≈ 1 when violated, 0 when not
    # For numerical stability
    gate = np.zeros_like(violations)
    pos_mask = violations > -5.0 / beta_gate
    gate[pos_mask] = 1.0 / (1.0 + np.exp(-beta_gate * violations[pos_mask]))
    neg_mask = ~pos_mask
    # For very negative violations, gate ≈ 0, gradient ≈ 0
    
    # Energy contribution: lam * sum(violations^2 * gate)
    # dE/d(c_i) from pair (i,j): 
    #   v = 2r - d_ij, d(v)/d(c_i) = -(c_i - c_j)/d_ij
    #   E_pair = lam * v^2 * gate(v)
    #   dE_pair/d(c_i) = lam * [2*v*gate(v) * d(v)/d(c_i) + v^2 * gate'(v) * d(v)/d(c_i)]
    #   gate'(v) = beta_gate * gate(v) * (1-gate(v))
    
    # Simplified: just use lam * v^2 for violated pairs (hard-ish), smooth with small epsilon
    # Actually let's use a cleaner approach:
    # E = lam * sum_{pairs} (max(0, 2r - d_ij))^2 + lam * sum_i (max(0, r - w_i))^2
    # Gradient is piecewise, but we can smooth with a small quadratic
    
    # Let me use a different smooth formulation:
    # For pairs: penalty = lam * (2r - d_ij)^2 when d_ij < 2r, else 0
    # Smooth approximation: lam * (2r - d_ij)^2 * sigmoid(k*(2r - d_ij))
    
    k = 15.0
    # Compute smooth penalties and gradients vectorized
    v = violations  # 2r - d_ij
    # Smooth gate: s(v) = 1/(1+exp(-k*v))
    # For stability
    s = np.zeros_like(v)
    s[v > -10/k] = 1.0 / (1.0 + np.exp(-k * v[v > -10/k]))
    # s'[v] = k * s * (1-s)
    sp = k * s * (1.0 - s)
    
    # E_pairs = lam * sum(v^2 * s)
    energy += lam * np.sum(v**2 * s)
    
    # dE/d(c_i) for pair (i,j):
    # d(v)/d(c_i) = -(c_i - c_j) / d_ij
    # d(v)/d(c_j) = (c_i - c_j) / d_ij
    # dE/d(c_i) = lam * [2*v*s + v^2*sp] * d(v)/d(c_i)
    #           = lam * [2*v*s + v^2*sp] * (-(c_i-c_j)/d_ij)
    
    coeff = lam * (2.0 * v * s + v**2 * sp)  # dE/dv
    
    for idx in range(len(i_idx)):
        i = i_idx[idx]
        j = j_idx[idx]
        c_val = coeff[idx]
        if abs(c_val) < 1e-12:
            continue
        d = pair_dists[idx]
        if d < 1e-10:
            continue
        vec = centers[i] - centers[j]
        # d(v)/d(c_i) = -vec/d, d(v)/d(c_j) = vec/d
        grad[i] += c_val * (-vec / d)
        grad[j] += c_val * (vec / d)
    
    # Wall constraints: want w_i >= target_r, i.e., 0.5 - |c_ik| >= target_r for all k
    # violation = target_r - w_i where w_i = min_k(0.5 - |c_ik|)
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists, axis=1)
    wall_violations = target_r - min_wall  # positive when violated
    
    # Smooth penalty
    wv = wall_violations
    sw = np.zeros_like(wv)
    sw[wv > -10/k] = 1.0 / (1.0 + np.exp(-k * wv[wv > -10/k]))
    spw = k * sw * (1.0 - sw)
    
    energy += lam * np.sum(wv**2 * sw)
    
    # Gradient: d(wv_i)/d(c_ik) = -d(min_wall_i)/d(c_ik)
    # min_wall_i = min_k(0.5 - |c_ik|)
    # d(min_wall_i)/d(c_ik) = -sign(c_ik) for the minimizing k
    # So d(wv_i)/d(c_ik) = sign(c_ik) for minimizing k
    
    coeff_w = lam * (2.0 * wv * sw + wv**2 * spw)
    
    for i in range(n):
        c_val = coeff_w[i]
        if abs(c_val) < 1e-12:
            continue
        min_k = np.argmin(wall_dists[i])
        sign = np.sign(centers[i, min_k])
        if sign == 0:
            sign = 1.0
        # d(wv_i)/d(c_ik) = sign(c_ik) for the minimizing coordinate
        grad[i, min_k] += c_val * sign
    
    return energy, grad

def _minimize_energy(centers, target_r, lam=50.0, time_limit=10.0):
    """Minimize energy for given target radius."""
    start = time.time()
    
    def obj_and_grad(x):
        if time.time() - start > time_limit:
            return 1e10, np.zeros_like(x)
        c = x.reshape(N, 3)
        e, g = _energy_and_gradient(c, target_r, lam)
        return e, g.flatten()
    
    x0 = centers.flatten()
    bounds = [(-0.4999999, 0.4999999)] * (N * 3)
    
    try:
        result = minimize(
            obj_and_grad, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50, 'maxfun': 3000}
        )
        return result.x.reshape(N, 3)
    except Exception:
        return centers

def _grid_init(n):
    """Grid initialization."""
    m = int(np.ceil(n ** (1.0/3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n]

def construct():
    """Construct optimal packing of N spheres in unit cube."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Start with grid initialization
    centers = _grid_init(N)
    current_r = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_r
    
    # Iterative radius increase strategy
    # Start from current radius, gradually increase target
    target_r = current_r * 0.95  # Start slightly below current
    max_target = 0.115  # Upper bound based on known best
    
    # Phase 1: Gradually increase target radius
    n_increments = 30
    for step in range(n_increments):
        if time.time() > deadline - 30.0:
            break
        
        # Linearly interpolate target
        progress = step / n_increments
        target_r = current_r + (max_target - current_r) * progress
        target_r = min(target_r, max_target)
        
        # Minimize energy for this target
        remaining = deadline - time.time() - 25.0
        if remaining < 3.0:
            break
        
        # Use adaptive lambda: increase as we go
        lam = 30.0 + 70.0 * progress
        
        centers = _minimize_energy(centers, target_r, lam=lam, time_limit=min(remaining, 8.0))
        centers = np.clip(centers, -0.4999999, 0.4999999)
        
        r_new = _compute_radius(centers)
        if r_new > best_radius:
            best_radius = r_new
            best_centers = centers.copy()
        
        # Update current_r to be the achieved radius
        current_r = max(current_r, r_new)
    
    # Phase 2: Fine-tune with high lambda and current best radius
    if time.time() < deadline - 15.0:
        remaining = deadline - time.time() - 10.0
        if remaining > 3.0:
            # Try to push slightly above best
            target_r2 = best_radius * 1.001
            centers2 = _minimize_energy(best_centers, target_r2, lam=100.0, time_limit=min(remaining, 8.0))
            centers2 = np.clip(centers2, -0.4999999, 0.4999999)
            r2 = _compute_radius(centers2)
            if r2 > best_radius:
                best_radius = r2
                best_centers = centers2.copy()
    
    # Phase 3: Final polish - minimize energy at best_radius with high lambda
    if time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 2.0:
            centers3 = _minimize_energy(best_centers, best_radius, lam=200.0, time_limit=min(remaining, 6.0))
            centers3 = np.clip(centers3, -0.4999999, 0.4999999)
            r3 = _compute_radius(centers3)
            if r3 > best_radius:
                best_radius = r3
                best_centers = centers3.copy()
    
    best_centers = np.clip(best_centers, -0.4999999, 0.4999999)
    return best_centers
