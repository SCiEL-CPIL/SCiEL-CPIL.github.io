import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

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

    # Pairwise distances using cdist for efficiency
    dists = cdist(centers, centers, metric='euclidean')
    
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)

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
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    # Using log-sum-exp trick for numerical stability
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _soft_min_objective_grad(centers_flat, beta, iu):
    """
    Gradient of the soft-min objective for L-BFGS-B.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_shifted = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_shifted)
    
    # Soft-min value
    obj = (max_val + np.log(sum_exp)) / beta
    
    # Gradient: d(obj)/d(x_i) = -exp(-beta*x_i) / sum(exp(-beta*x_j))
    # = -softmax(-beta * all_vals)
    weights = exp_shifted / sum_exp  # shape (N*3 + N*(N-1)/2,)
    
    # Build gradient w.r.t. centers_flat
    grad = np.zeros_like(centers_flat)
    
    # Wall part: d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    wall_grad = -np.sign(centers)  # shape (N, 3)
    grad += wall_grad * weights[:N*3].reshape(-1, 3)
    
    # Pair part: d(|c_i - c_j|/2)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # We need to accumulate contributions from each pair
    pair_weights = weights[N*3:]  # shape (N*(N-1)/2,)
    
    # For each pair (i,j) with i<j, contribution to c_i: w_ij * (c_i - c_j)/|c_i-c_j|
    # and to c_j: -w_ij * (c_i - c_j)/|c_i-c_j|
    
    # Use the upper triangle indices
    i_idx, j_idx = iu
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = pair_weights[k]
        diff = centers[i] - centers[j]
        d = np.linalg.norm(diff)
        if d > 1e-12:
            direction = diff / d
            grad[i] += w * direction
            grad[j] -= w * direction
    
    return obj, grad

def _build_fcc_init(seed=42):
    """
    Build a Face-Centered Cubic (FCC) initialization.
    FCC is denser than BCC.
    """
    rng = np.random.default_rng(seed)
    points = []
    
    # FCC lattice points in a 4x4x4 supercell
    # Basis: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # Scale to fit in [-0.5, 0.5]
    # Let's use a grid of unit cells and add the FCC basis vectors.
    
    nx, ny, nz = 4, 4, 4
    margin = 0.02
    x_min, x_max = -0.5 + margin, 0.5 - margin
    dx = (x_max - x_min) / (nx - 1) if nx > 1 else 0
    
    basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ]
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x_base = x_min + i * dx
                y_base = x_min + j * dx
                z_base = x_min + k * dx
                
                for b in basis:
                    x = x_base + b[0] * dx
                    y = y_base + b[1] * dx
                    z = z_base + b[2] * dx
                    if -0.499 <= x <= 0.499 and -0.499 <= y <= 0.499 and -0.499 <= z <= 0.499:
                        points.append([x, y, z])
    
    centers = np.array(points)
    
    # If we have too many or too few, adjust
    if len(centers) > N:
        # Randomly select N points
        indices = rng.choice(len(centers), N, replace=False)
        centers = centers[indices]
    elif len(centers) < N:
        # Add random points
        missing = N - len(centers)
        new_points = rng.uniform(-0.49, 0.49, (missing, 3))
        centers = np.vstack([centers, new_points])
    
    # Add small noise to break symmetry
    centers += rng.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(seed=42):
    """
    Build a body-centered cubic (BCC) initialization.
    """
    rng = np.random.default_rng(seed)
    
    nx, ny, nz = 4, 4, 4
    points = []
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = -0.45 + i * (0.9 / (nx - 1)) if nx > 1 else 0.0
                y = -0.45 + j * (0.9 / (ny - 1)) if ny > 1 else 0.0
                z = -0.45 + k * (0.9 / (nz - 1)) if nz > 1 else 0.0
                points.append([x, y, z])
                points.append([x + 0.9/(2*(nx-1)), y + 0.9/(2*(ny-1)), z + 0.9/(2*(nz-1))])
    
    centers = np.array(points[:N])
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n_dims=3, seed=42):
    """
    Build a Halton sequence initialization.
    """
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        point = [halton(i, base) for base in [2, 3, 5]]
        point = [p * 0.94 - 0.47 for p in point]
        points.append(point)
    
    centers = np.array(points)
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.002, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _force_directed_refinement(centers, deadline, iu, n_steps=2000):
    """
    Refine centers using force-directed repulsion.
    This helps escape local minima of the soft-min objective.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = centers.copy()
    current_radius = best_radius
    
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    
    # Adaptive step size
    step_size = 0.03
    learning_rate = 0.98
    patience = 0
    max_patience = 50
    
    for step in range(n_steps):
        if time.time() > deadline - 25.0:
            break
        
        # Compute forces using broadcasting (single pass)
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists_mat, np.inf)
        
        # Threshold for repulsion
        threshold = 2.2 * best_radius
        
        # Mask for repulsion
        mask = dists_mat < threshold
        # Safe distances
        safe_dists_mat = np.where(mask, dists_mat, 1.0)
        
        # Force magnitude: 1/d^2
        force_mag = np.where(mask, 1.0 / (safe_dists_mat ** 2), 0.0)
        
        # Direction: diff / dist
        dir_mat = diff / safe_dists_mat[:, :, np.newaxis]
        
        # Total force
        forces = np.sum(force_mag[:, :, np.newaxis] * dir_mat, axis=1)
        
        # Wall forces: Push away from walls
        wall_dist = 0.5 - np.abs(current_centers)
        wall_mask = wall_dist < best_radius * 1.2
        # Force towards center
        wall_forces = np.where(wall_mask, -current_centers / (np.abs(current_centers) + 1e-6), 0.0)
        # Scale wall force
        forces += 0.8 * wall_forces
        
        # Update positions
        current_centers += step_size * forces
        current_centers = np.clip(current_centers, -0.499, 0.499)
        
        new_radius = _compute_radius(current_centers)
        
        if new_radius > current_radius:
            current_radius = new_radius
            patience = 0
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = current_centers.copy()
            # Reduce step size if improving
            step_size *= 0.97
        else:
            # Keep current position but reduce step
            patience += 1
            step_size *= learning_rate
            if patience >= max_patience:
                # Reset to best and try again with smaller step
                current_centers = best_centers.copy()
                current_radius = best_radius
                step_size = max(0.005, step_size * 0.5)
                patience = 0
                if step_size < 0.001:
                    break
            
    return best_centers, best_radius

def _optimize_sequence(centers, deadline, iu, beta_start=10.0, beta_end=20000.0, n_steps=10):
    """
    Run L-BFGS-B optimization with increasing beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for i_beta, beta in enumerate(betas):
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        x0 = best_centers.flatten()
        
        # Allocate more iterations for later (higher beta) steps
        frac = i_beta / max(1, n_steps - 1)
        maxiter = int(200 + 400 * frac)
        ftol = 1e-12 * (1.0 - 0.5 * frac)
        gtol = 1e-8 * (1.0 - 0.5 * frac)
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
    return best_centers, best_radius

def _final_polish(centers, deadline, iu):
    """
    High-precision polish pass with multiple beta values and tighter tolerances.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    high_betas = [5000, 10000, 20000, 40000, 80000]
    
    for beta in high_betas:
        if time.time() > deadline - 3.0:
            break
            
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
            
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 147.0
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Generate diverse initializations
    inits = []
    
    # FCC Initialization (Densest)
    fcc_centers = _build_fcc_init(seed=42)
    inits.append(fcc_centers)
    
    # BCC Initialization
    bcc_centers = _build_bcc_init(seed=42)
    inits.append(bcc_centers)
    
    # Halton Initialization
    halton_centers = _build_halton_init(seed=42)
    inits.append(halton_centers)
    
    # Additional grid initializations for diversity
    for seed in [7, 13]:
        rng = np.random.default_rng(seed)
        grid = rng.uniform(-0.48, 0.48, (N, 3))
        inits.append(grid)
    
    # Phase 2: For each initialization, run optimization
    for init_idx, init_centers in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 35.0:
            break
        
        r_init = _compute_radius(init_centers)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init_centers.copy()
        
        # Optimization 1: Soft-Min L-BFGS
        opt_centers, opt_radius = _optimize_sequence(
            init_centers, 
            deadline, 
            iu,
            beta_start=10.0, 
            beta_end=5000.0, 
            n_steps=8
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers.copy()
            
        # Optimization 2: Force-Directed Refinement
        if time.time() < deadline - 20.0:
            fd_centers, fd_radius = _force_directed_refinement(
                best_centers, 
                deadline, 
                iu,
                n_steps=1500
            )
            if fd_radius > best_radius:
                best_radius = fd_radius
                best_centers = fd_centers.copy()
                
        # Optimization 3: Final Polish
        if time.time() < deadline - 5.0:
            polished_centers, polished_radius = _final_polish(best_centers, deadline, iu)
            if polished_radius > best_radius:
                best_radius = polished_radius
                best_centers = polished_centers.copy()

    if best_centers is None:
        best_centers = _build_fcc_init(seed=42)
        best_radius = _compute_radius(best_centers)
        
    return best_centers

# EVOLVE-BLOCK-END
