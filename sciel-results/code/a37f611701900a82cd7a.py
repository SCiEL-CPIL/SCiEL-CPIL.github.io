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
    
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_shifted = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_shifted)
    log_sum = max_val + np.log(sum_exp)
    obj = log_sum / beta
    
    weights = -exp_shifted / sum_exp
    
    grad = np.zeros_like(centers_flat)
    
    wall_grad = weights[:N*3].reshape(N, 3)
    grad -= wall_grad * np.sign(centers)
    
    pair_weights = weights[N*3:]
    
    safe_dists = np.where(pair_dists > 1e-15, pair_dists, 1e-15)
    pair_unit = diff / safe_dists[:, np.newaxis]
    
    np.add.at(grad, i, pair_weights[:, np.newaxis] * pair_unit)
    np.add.at(grad, j, -pair_weights[:, np.newaxis] * pair_unit)
    
    return obj, grad

def _build_initial_grid(n=100, noise_scale=0.003, seed=42):
    nx, ny, nz = 5, 5, 4
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
    rng = np.random.default_rng(seed)
    centers = []
    s = 1.0 / 5.0
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                x0 = -0.5 + ix * s
                y0 = -0.5 + iy * s
                z0 = -0.5 + iz * s
                for dx, dy, dz in [(0, 0, 0), (0, 0.5*s, 0.5*s), (0.5*s, 0, 0.5*s), (0.5*s, 0.5*s, 0)]:
                    x = x0 + dx
                    y = y0 + dy
                    z = z0 + dz
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        centers.append([x, y, z])
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
    rng = np.random.default_rng(seed)
    centers = []
    s = 1.0 / 5.0
    for ix in range(6):
        for iy in range(6):
            for iz in range(6):
                x = -0.5 + ix * s
                y = -0.5 + iy * s
                z = -0.5 + iz * s
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    centers.append([x, y, z])
    for ix in range(5):
        for iy in range(5):
            for iz in range(5):
                x = -0.5 + (ix + 0.5) * s
                y = -0.5 + (iy + 0.5) * s
                z = -0.5 + (iz + 0.5) * s
                centers.append([x, y, z])
    centers = np.array(centers[:n])
    if len(centers) < n:
        extra = rng.uniform(-0.49, 0.49, (n - len(centers), 3))
        centers = np.vstack([centers, extra])
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _build_halton_init(n=100, seed=42):
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
    rng = np.random.default_rng(seed)
    centers = np.empty((n, 3))
    min_sep = 0.15
    for i in range(n):
        attempts = 0
        while attempts < 1000:
            c = rng.uniform(-0.48, 0.48, 3)
            if i == 0:
                centers[i] = c
                break
            dists = np.sqrt(np.sum((centers[:i] - c)**2, axis=1))
            if np.min(dists) >= min_sep * 0.7:
                centers[i] = c
                break
            attempts += 1
        else:
            centers[i] = rng.uniform(-0.48, 0.48, 3)
    centers = np.clip(centers, -0.499, 0.499)
    return centers

def _anneal(centers, iu, deadline, initial_temp=0.004, cooling_rate=0.9997):
    # Placeholder to keep function signature
    return centers, _compute_radius(centers)

def _optimize_from_initial(centers, iu, time_limit, seed=0):
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_centers = best_centers.copy()
    current_radius = best_radius
    
    rng = np.random.default_rng(seed)
    start_time = time.time()
    deadline = start_time + time_limit
    
    beta = 10.0
    max_beta = 100000.0
    no_improve_count = 0
    max_no_improve = 5
    
    while time.time() < deadline and beta < max_beta:
        if time.time() > deadline - 2.0:
            break
            
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                current_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            
            if r > current_radius + 1e-12:
                current_radius = r
                current_centers = new_centers
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
                
        except Exception:
            no_improve_count += 1
            
        if no_improve_count >= max_no_improve:
            kick_scale = 0.005
            current_centers = current_centers + rng.normal(0, kick_scale, current_centers.shape)
            current_centers = np.clip(current_centers, -0.5, 0.5)
            current_radius = _compute_radius(current_centers)
            beta = max(beta * 0.5, 10.0)
            no_improve_count = 0
        else:
            beta *= 1.5
            
    return best_centers, best_radius

def _polish(centers, iu, deadline):
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    betas = [10000.0, 50000.0, 100000.0]
    for beta in betas:
        if time.time() > deadline - 1.0:
            break
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                best_centers.flatten(),
                args=(beta, iu),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 500}
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
    deadline = start_time + 145.0
    iu = _IU
    
    inits = []
    inits.append(_build_initial_grid(seed=42))
    inits.append(_build_initial_grid(seed=123))
    inits.append(_build_bcc_init(seed=42))
    inits.append(_build_bcc_init(seed=789))
    inits.append(_build_fcc_init(seed=42))
    inits.append(_build_fcc_init(seed=999))
    inits.append(_build_halton_init(seed=42))
    inits.append(_build_halton_init(seed=789))
    inits.append(_build_random_init(seed=42))
    inits.append(_build_random_init(seed=137))
    
    scored_inits = []
    for c in inits:
        r = _compute_radius(c)
        scored_inits.append((r, c))
    
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    best_centers = None
    best_radius = 0.0
    
    polish_time = 10.0
    time_for_search = deadline - 10.0 - start_time
    time_per_init = time_for_search / 4.0
    
    for idx in range(min(4, len(scored_inits))):
        r_init, c_init = scored_inits[idx]
        
        if time.time() > deadline - polish_time - 2.0:
            break
            
        opt_deadline = time.time() + time_per_init
        if opt_deadline > deadline - polish_time - 2.0:
            opt_deadline = deadline - polish_time - 2.0
            
        try:
            opt_centers, opt_radius = _optimize_from_initial(c_init, iu, opt_deadline - time.time(), seed=idx+100)
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = opt_centers
        except Exception:
            pass
            
    if best_centers is not None and time.time() < deadline - 1.0:
        try:
            polished, p_radius = _polish(best_centers, iu, deadline - 1.0)
            if p_radius > best_radius:
                best_radius = p_radius
                best_centers = polished
        except Exception:
            pass
            
    if best_centers is None:
        best_centers = scored_inits[0][1]
        
    return best_centers

# EVOLVE-BLOCK-END
