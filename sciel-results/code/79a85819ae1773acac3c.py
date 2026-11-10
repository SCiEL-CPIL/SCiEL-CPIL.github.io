"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(c):
    """Compute the largest common radius for given centers."""
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _compute_radius_fast(c):
    """Faster radius computation using vectorized ops, no fill_diagonal."""
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Set diagonal to inf by masking
    min_pair_dist = np.min(dists[~np.eye(n, dtype=bool)])
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    return 0.5 * min(min_pair_dist, min_wall_dist)

def _forces(c, beta=30.0):
    """
    Compute force-directed forces with smooth soft-min surrogate for the
    nonsmooth min objective. Returns forces array of shape (n, 3).
    """
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    forces = np.zeros_like(c)
    
    # Pairwise repulsion using inverse-distance weighting (smooth surrogate)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Soft weights: w_ij = exp(-beta * d_ij), normalized per row
    neg_beta_dists = -beta * dists
    max_nb = np.max(neg_beta_dists, axis=1, keepdims=True)
    exp_nb = np.exp(neg_beta_dists - max_nb)
    np.fill_diagonal(exp_nb, 0.0)
    sum_exp = np.sum(exp_nb, axis=1, keepdims=True)
    sum_exp = np.maximum(sum_exp, 1e-30)
    weights = exp_nb / sum_exp  # shape (n, n)
    
    # Force on i from j: weights[i,j] * (c_i - c_j) / d_ij
    # Vectorized: for each i, sum_j weights[i,j] * (c_i - c_j) / d_ij
    inv_dists = np.zeros_like(dists)
    mask = dists < np.inf
    inv_dists[mask] = 1.0 / dists[mask]
    
    # direction[i,j] = (c_i - c_j) / d_ij
    direction = diff * inv_dists[:, :, np.newaxis]
    np.fill_diagonal(direction, 0.0)
    
    # force_pair[i] = sum_j weights[i,j] * direction[i,j]
    force_pair = np.sum(weights[:, :, np.newaxis] * direction, axis=1)
    forces += force_pair
    
    # Wall forces: push away from walls where close
    wall_dists = 0.5 - np.abs(c)
    # Soft-min over all wall distances per component
    for k in range(3):
        neg_beta_walls = -beta * wall_dists[:, k]
        max_nbw = np.max(neg_beta_walls)
        exp_nbw = np.exp(neg_beta_walls - max_nbw)
        sum_expw = np.sum(exp_nbw)
        sum_expw = max(sum_expw, 1e-30)
        weights_w = exp_nbw / sum_expw
        # Gradient of soft-min wall dist w.r.t. c_ik: -sign(c_ik) * weight
        # Force should push away from wall: +sign(c_ik) * weight
        forces[:, k] += weights_w * np.sign(c[:, k])
    
    return forces

def _refine_force_directed(c, time_limit=100.0):
    """
    Force-directed refinement using smooth surrogate forces.
    Accepts moves that improve the actual (nonsmooth) radius.
    """
    c = np.array(c, dtype=np.float64)
    n = len(c)
    best_c = c.copy()
    best_r = _compute_radius_fast(c)
    
    start_time = time.time()
    step_size = 0.003
    beta = 30.0
    max_iter = 20000
    patience = 2000
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = _forces(c, beta=beta)
        
        # Normalize forces to unit magnitude per sphere
        norm_f = np.linalg.norm(forces, axis=1, keepdims=True)
        norm_f = np.maximum(norm_f, 1e-15)
        forces = forces / norm_f
        
        # Move
        new_c = c + step_size * forces
        
        # Clamp to cube interior
        new_c = np.clip(new_c, -0.5 + 1e-12, 0.5 - 1e-12)
        
        # Evaluate
        new_r = _compute_radius_fast(new_c)
        
        if new_r > best_r + 1e-15:
            c = new_c
            best_c = c.copy()
            best_r = new_r
            no_improve = 0
            # Slowly increase step size
            step_size = min(step_size * 1.01, 0.015)
            # Slowly decrease beta (more focused on closest pairs)
            beta = max(beta * 0.999, 10.0)
        else:
            no_improve += 1
            step_size *= 0.98
            if step_size < 1e-7:
                break
            if no_improve > patience:
                break
    
    return best_c, best_r

def _jittered_grid(scale=1.0, seed=42):
    """Create a jittered grid initial configuration."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    # Add jitter
    c += rng.randn(N, 3) * 0.01 * scale
    # Clamp
    c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
    return c

def _body_centered_init(seed=0):
    """
    Alternative initialization: body-centered cubic-like arrangement.
    Splits the cube into sub-cubes and places spheres at centers and body centers.
    """
    rng = np.random.RandomState(seed)
    # Use 5x5x4 = 100 arrangement
    # Place on a 5x5x4 grid with some offset
    nx, ny, nz = 5, 5, 4
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    c = grid[:N].copy()
    # Small random perturbation to break symmetry
    c += rng.randn(N, 3) * 0.005
    c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
    return c

def _face_centered_init(seed=0):
    """
    Another initialization strategy: layered arrangement.
    """
    rng = np.random.RandomState(seed)
    # 10 layers of 10 spheres each
    n_layers = 10
    per_layer = N // n_layers
    z_positions = (np.arange(n_layers) + 0.5) / n_layers - 0.5
    c_list = []
    for li, z in enumerate(z_positions):
        # Alternate between two 5x5 patterns offset by half
        if li % 2 == 0:
            xs = (np.arange(5) + 0.5) / 5 - 0.5
            ys = (np.arange(5) + 0.5) / 5 - 0.5
        else:
            xs = (np.arange(5) + 0.0) / 5 - 0.5
            ys = (np.arange(5) + 0.0) / 5 - 0.5
        for x in xs:
            for y in ys:
                c_list.append([x, y, z])
    c = np.array(c_list[:N])
    c += rng.randn(N, 3) * 0.003
    c = np.clip(c, -0.5 + 1e-10, 0.5 - 1e-10)
    return c

def construct():
    """
    Construct centers for 100 equal spheres in unit cube.
    Uses force-directed optimization with smooth surrogate forces
    and multiple initialization strategies.
    """
    total_time_limit = 115.0  # Safety margin before 120s kill
    start_time = time.time()
    
    # Try multiple initialization strategies
    inits = [
        ("grid", _jittered_grid(1.0, seed=42)),
        ("grid2", _jittered_grid(2.0, seed=123)),
        ("bcc", _body_centered_init(seed=0)),
        ("fcc", _face_centered_init(seed=0)),
        ("grid3", _jittered_grid(0.5, seed=777)),
    ]
    
    best_c = None
    best_r = 0.0
    
    n_inits = len(inits)
    # Allocate time per initialization
    time_per_init = (total_time_limit - 5.0) / n_inits  # 5s buffer
    
    for idx, (name, c_init) in enumerate(inits):
        if time.time() - start_time > total_time_limit - 3.0:
            break
        
        r_init = _compute_radius_fast(c_init)
        if r_init > best_r:
            best_c = c_init.copy()
            best_r = r_init
        
        # Refine
        remaining = min(time_per_init, total_time_limit - (time.time() - start_time) - 2.0)
        if remaining > 5.0:
            c_ref, r_ref = _refine_force_directed(c_init, time_limit=remaining)
            if r_ref > best_r:
                best_c = c_ref.copy()
                best_r = r_ref
    
    # Final polish on the best configuration with remaining time
    if best_c is not None:
        elapsed = time.time() - start_time
        remaining = total_time_limit - elapsed - 1.0
        if remaining > 5.0:
            c_polish, r_polish = _refine_force_directed(best_c, time_limit=remaining)
            if r_polish > best_r:
                best_c = c_polish.copy()
                best_r = r_polish
    
    # Ensure valid output
    if best_c is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    # Final validity check and clamp
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify finite values
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_c
# EVOLVE-BLOCK-END
