import numpy as np
import time
from scipy.optimize import minimize

N = 100
_IU = np.triu_indices(N, k=1)
_PAIRS = _IU[0].size  # number of pairwise constraints
_WALLS = N * 3        # number of wall constraints

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)

    # Pairwise distances (only upper triangle)
    i, j = _IU
    diff = centers[i] - centers[j]
    dists_sq = np.einsum('ij,ij->i', diff, diff)
    min_pair_dist = np.sqrt(np.min(dists_sq))

    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2 (upper triangle only)
    i, j = iu
    diff = centers[i] - centers[j]
    dists_sq = np.einsum('ij,ij->i', diff, diff)
    pair_vals = np.sqrt(dists_sq) / 2.0
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj


def _soft_min_objective_and_grad(centers_flat, beta, iu):
    """
    Objective and gradient for the soft-min approximation.
    Returns (obj, grad).
    """
    centers = centers_flat.reshape(-1, 3)
    n = centers.shape[0]
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2 (upper triangle only)
    i, j = iu
    diff = centers[i] - centers[j]
    dists_sq = np.einsum('ij,ij->i', diff, diff)
    pair_dists = np.sqrt(dists_sq)
    pair_vals = pair_dists / 2.0
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    m = all_vals.size
    
    # Soft-min: -1/beta * log(sum(exp(-beta * x)))
    # = 1/beta * log(sum(exp(-beta * x))) with sign flip... 
    # Actually we want to MINIMIZE the radius, so we maximize soft-min.
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize this, so minimize -soft_min = 1/beta * log(sum(exp(-beta*x)))
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_shifted = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_shifted)
    log_sum = max_val + np.log(sum_exp)
    obj = log_sum / beta
    
    # Gradient of obj w.r.t. all_vals:
    # d/dx_i [1/beta * log(sum(exp(-beta*x)))] = 1/beta * (-beta * exp(-beta*x_i)) / sum(exp(-beta*x))
    # = -exp(-beta*x_i) / sum(exp(-beta*x)) = -exp_shifted_i / sum_exp
    weights = -exp_shifted / sum_exp  # shape (m,)
    
    # Now backprop to centers_flat
    grad = np.zeros_like(centers_flat)
    
    # Wall part: wall_vals[k] = 0.5 - |centers[k//3, k%3]|
    # d(wall_vals[k])/d(centers[c]) = -sign(centers[c])
    wall_grad = weights[:N*3].reshape(N, 3)
    grad -= wall_grad * np.sign(centers)
    
    # Pair part: pair_vals[p] = |centers[i_p] - centers[j_p]| / 2
    # d(pair_vals[p])/d(centers[i_p]) = (centers[i_p] - centers[j_p]) / (2 * dist_p)
    # d(pair_vals[p])/d(centers[j_p]) = (centers[j_p] - centers[i_p]) / (2 * dist_p)
    pair_weights = weights[N*3:]
    # For each pair p:
    # grad[i_p] += pair_weights[p] * (diff_p) / (2 * dist_p)
    # grad[j_p] -= pair_weights[p] * (diff_p) / (2 * dist_p)
    
    # Avoid division by zero
    safe_dists = np.where(pair_dists > 1e-15, pair_dists, 1e-15)
    pair_unit = diff / safe_dists[:, np.newaxis]  # unit vectors
    
    # Accumulate gradients
    np.add.at(grad, i, pair_weights[:, np.newaxis] * pair_unit)
    np.add.at(grad, j, -pair_weights[:, np.newaxis] * pair_unit)
    
    return obj, grad

def _build_initial_grid(n=100, noise_scale=0.003, seed=42):
    """
    Build a well-distributed grid initialization.
    Uses a 5x5x4 grid which gives exactly 100 points.
    """
    nx, ny, nz = 5, 5, 4
    # Use full extent with small margin to allow optimization to push to boundaries
    margin = 0.005
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers[:n]


def _build_fcc_init(n=100, noise_scale=0.002, seed=42):
    """
    Build an FCC-like initialization. FCC has the highest density for equal spheres.
    We place points at FCC lattice sites within the unit cube.
    """
    rng = np.random.default_rng(seed)
    centers = []
    
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) + integer shifts
    # Scale to fit in [-0.5, 0.5]^3
    s = 1.0 / 5.0  # lattice spacing
    
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                x0 = -0.5 + ix * s
                y0 = -0.5 + iy * s
                z0 = -0.5 + iz * s
                # FCC basis
                for dx, dy, dz in [(0, 0, 0), (0, 0.5*s, 0.5*s), (0.5*s, 0, 0.5*s), (0.5*s, 0.5*s, 0)]:
                    x = x0 + dx
                    y = y0 + dy
                    z = z0 + dz
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        centers.append([x, y, z])
    
    # Take first n or pad
    if len(centers) >= n:
        centers = np.array(centers[:n])
    else:
        centers = np.array(centers)
        extra = rng.uniform(-0.49, 0.49, (n - len(centers), 3))
        centers = np.vstack([centers, extra])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(n=100, noise_scale=0.003, seed=42):
    """
    Build a body-centered cubic initialization.
    BCC has better packing density than simple cubic.
    For a unit cube, place points at corners and face/body centers.
    """
    rng = np.random.default_rng(seed)
    centers = []
    
    # Simple cubic grid with spacing ~1/5
    s = 1.0 / 5.0
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                x = -0.5 + ix * s
                y = -0.5 + iy * s
                z = -0.5 + iz * s
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    centers.append([x, y, z])
    
    # Add body centers of each sub-cell
    for ix in range(5):
        for iy in range(5):
            for iz in range(5):
                x = -0.5 + (ix + 0.5) * s
                y = -0.5 + (iy + 0.5) * s
                z = -0.5 + (iz + 0.5) * s
                centers.append([x, y, z])
    
    centers = np.array(centers[:n])
    if len(centers) < n:
        # Pad with random points if needed
        extra = rng.uniform(-0.49, 0.49, (n - len(centers), 3))
        centers = np.vstack([centers, extra])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers


def _build_random_init(n=100, seed=42):
    """
    Random initialization with minimum separation enforced via rejection.
    """
    rng = np.random.default_rng(seed)
    centers = np.empty((n, 3))
    min_sep = 0.15  # target minimum separation
    
    for i in range(n):
        attempts = 0
        while attempts < 1000:
            c = rng.uniform(-0.48, 0.48, 3)
            if i == 0:
                centers[i] = c
                break
            # Check distance to all previous
            dists = np.sqrt(np.sum((centers[:i] - c)**2, axis=1))
            if np.min(dists) >= min_sep * 0.7:  # allow some overlap initially
                centers[i] = c
                break
            attempts += 1
        else:
            # Give up and place randomly
            centers[i] = rng.uniform(-0.48, 0.48, 3)
    
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _polish(centers, iu, deadline):
    """
    Final high-precision polish pass using very high beta soft-min
    with tight tolerances and many iterations, plus exact local refinement.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Try several high-beta values with increasing precision
    betas = [2000.0, 5000.0, 10000.0, 20000.0, 40000.0]
    
    for beta in betas:
        if time.time() > deadline - 2.0:
            break
        
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 500}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Exact local refinement: identify active constraints and solve QP-like problem
    if time.time() < deadline - 0.5:
        try:
            for iteration in range(5):
                if time.time() > deadline - 0.3:
                    break
                    
                new_centers = best_centers.copy()
                r = best_radius
                
                # Identify tight constraints (within tolerance of current radius)
                tol = 1e-10
                dists_to_walls = 0.5 - np.abs(best_centers)
                wall_tight = np.where(dists_to_walls < r + tol)[0]
                
                i, j = iu
                diff = best_centers[i] - best_centers[j]
                dists_sq = np.einsum('ij,ij->i', diff, diff)
                pair_dists = np.sqrt(dists_sq) / 2.0
                pair_tight = np.where(pair_dists < r + tol)[0]
                
                # For each tight pair, nudge apart
                for tp in pair_tight:
                    ii, jj = int(i[tp]), int(j[tp])
                    d = best_centers[ii] - best_centers[jj]
                    dist = np.sqrt(np.sum(d**2))
                    if dist > 1e-15:
                        unit = d / dist
                        # Move each by half the deficit
                        deficit = 2*r - dist
                        if deficit > 0:
                            new_centers[ii] += unit * deficit * 0.5
                            new_centers[jj] -= unit * deficit * 0.5
                
                # For each tight wall, push inward
                for wi in wall_tight:
                    for k in range(3):
                        coord = new_centers[wi, k]
                        if coord > 0:
                            wall_dist = 0.5 - coord
                            if wall_dist < r:
                                new_centers[wi, k] = 0.5 - r
                        else:
                            wall_dist = 0.5 + coord
                            if wall_dist < r:
                                new_centers[wi, k] = -0.5 + r
                
                new_centers = np.clip(new_centers, -0.5, 0.5)
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
                else:
                    break  # No improvement, stop
        except Exception:
            pass
    
    return best_centers, best_radius

def _anneal(centers, iu, deadline, initial_temp=0.004, cooling_rate=0.9997):
    """
    Targeted simulated annealing that focuses on improving the minimum constraint.
    Uses bottleneck identification to guide perturbations with adaptive step sizes.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    temp = initial_temp
    
    rng = np.random.default_rng(123)
    stuck_count = 0
    max_stuck = 500
    total_iters = 0
    
    while time.time() < deadline and temp > 0.00003:
        total_iters += 1
        
        # Compute all constraints to identify bottlenecks
        dists_to_walls = 0.5 - np.abs(best_centers)
        wall_min = np.min(dists_to_walls)
        
        i, j = iu
        diff = best_centers[i] - best_centers[j]
        dists_sq = np.einsum('ij,ij->i', diff, diff)
        pair_dists = np.sqrt(dists_sq) / 2.0
        pair_min = np.min(pair_dists)
        
        # Determine which constraint is the bottleneck
        # Find indices of spheres involved in tight constraints
        candidates = set()
        
        # Wall-bottleneck spheres (use tighter threshold as we progress)
        wall_thresh = wall_min + 0.01
        wall_close = np.where(dists_to_walls < wall_thresh)[0]
        candidates.update(wall_close.tolist())
        
        # Pair-bottleneck spheres
        pair_thresh = pair_min + 0.01
        tight_pairs = np.where(pair_dists < pair_thresh)[0]
        for tp in tight_pairs[:15]:  # Limit to avoid too many
            candidates.add(int(i[tp]))
            candidates.add(int(j[tp]))
        
        if len(candidates) == 0:
            candidates = set(rng.choice(N, size=8, replace=False).tolist())
        
        # Perturb 1-4 spheres from the bottleneck set
        n_perturb = int(rng.integers(1, 5))
        cand_list = list(candidates)
        idx = rng.choice(cand_list, size=min(n_perturb, len(cand_list)), replace=False)
        
        new_centers = best_centers.copy()
        # Adaptive step size: larger when stuck, smaller when making progress
        base_step = temp * 0.3
        if stuck_count > max_stuck // 2:
            step_size = base_step * 1.5
        else:
            step_size = base_step
        new_centers[idx] += rng.normal(0, step_size, (len(idx), 3))
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centers)
        
        delta = new_radius - current_radius
        if delta > 0:
            best_centers = new_centers
            best_radius = new_radius
            current_radius = new_radius
            stuck_count = 0
        else:
            prob = np.exp(delta / max(temp, 1e-10))
            if rng.random() < prob:
                best_centers = new_centers
                current_radius = new_radius
                stuck_count += 1
            else:
                stuck_count += 1
        
        temp *= cooling_rate
        
        if stuck_count > max_stuck:
            # Restart with perturbed best, reset temperature partially
            stuck_count = 0
            temp = max(temp * 0.4, 0.0008)
            perturbation = rng.normal(0, 0.003, best_centers.shape)
            best_centers = best_centers + perturbation
            best_centers = np.clip(best_centers, -0.5, 0.5)
            current_radius = _compute_radius(best_centers)
            # Update best if this perturbation happened to be better
            if current_radius > best_radius:
                best_radius = current_radius
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # Safety margin for 150s limit
    
    # Precompute upper triangle indices
    iu = _IU
    
    # Initial guess: Multiple diverse initializations
    best_centers = None
    best_radius = 0.0
    
    inits = []
    
    # Grid inits (5x5x4 = 100 points)
    inits.append(_build_initial_grid(seed=42))
    inits.append(_build_initial_grid(seed=123))
    inits.append(_build_initial_grid(seed=456))
    
    # BCC init
    inits.append(_build_bcc_init(seed=42))
    inits.append(_build_bcc_init(seed=789))
    
    # FCC init
    inits.append(_build_fcc_init(seed=42))
    inits.append(_build_fcc_init(seed=999))
    
    # Halton init
    inits.append(_build_halton_init(seed=42))
    inits.append(_build_halton_init(seed=789))
    
    # Random init with separation
    inits.append(_build_random_init(seed=42))
    inits.append(_build_random_init(seed=137))
    
    for centers in inits:
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        best_centers = _build_initial_grid()
        best_radius = _compute_radius(best_centers)
    
    # Phase 1: Soft-min optimization with adaptive beta and restarts
    # Use gradient-augmented L-BFGS-B for faster convergence
    beta = 10.0
    max_beta = 50000.0
    no_improve_count = 0
    max_no_improve = 3
    
    while time.time() < deadline - 50.0 and beta < max_beta:
        x0 = best_centers.flatten()
        r_before = best_radius
        
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
                no_improve_count = 0
            else:
                no_improve_count += 1
        except Exception:
            no_improve_count += 1
        
        # Increase beta adaptively
        if no_improve_count >= max_no_improve:
            # Try a restart: perturb and try different beta
            no_improve_count = 0
            rng = np.random.default_rng(int(time.time() * 1000) % 100000)
            perturbed = best_centers + rng.normal(0, 0.002, best_centers.shape)
            perturbed = np.clip(perturbed, -0.5, 0.5)
            r_pert = _compute_radius(perturbed)
            if r_pert > best_radius:
                best_radius = r_pert
                best_centers = perturbed
            beta = max(beta * 0.8, 5.0)  # Reset beta lower for exploration
        else:
            beta *= 1.35
    
    # Phase 2: Annealing to escape local minima
    if time.time() < deadline - 30.0:
        try:
            sa_centers, sa_radius = _anneal(
                best_centers, 
                iu,
                deadline - 30.0,
                initial_temp=0.004,
                cooling_rate=0.9997
            )
            if sa_radius > best_radius:
                best_radius = sa_radius
                best_centers = sa_centers
        except Exception:
            pass
            
    # Phase 3: Another round of optimization after annealing
    if time.time() < deadline - 15.0:
        beta = 5000.0
        for _ in range(3):
            if time.time() > deadline - 15.5:
                break
            try:
                res = minimize(
                    _soft_min_objective_and_grad,
                    best_centers.flatten(),
                    args=(beta, iu),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 300}
                )
                
                new_centers = res.x.reshape(-1, 3)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                r = _compute_radius(new_centers)
                if r > best_radius:
                    best_radius = r
                    best_centers = new_centers
            except Exception:
                pass
            beta *= 1.5
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            polished, p_radius = _polish(
                best_centers,
                iu,
                deadline - 3.0
            )
            if p_radius > best_radius:
                best_radius = p_radius
                best_centers = polished
        except Exception:
            pass
    
    return best_centers

# EVOLVE-BLOCK-END
