"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _compute_radius(centers):
    """Compute the radius: min of half pairwise distances and wall distances."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    if min_wall < 0:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_wall, min_pair)


def _soft_radius(centers, D, beta):
    """
    Smoothed soft-min radius.
    softmin of (d_ij/2) and wall distances, all divided by 2 for pair.
    We use: r_soft = -1/beta * log( sum exp(-beta * val) )  over all constraints.
    Constraints: d_ij/2 for i<j, and (0.5 - |c_ik|) for each i,k.
    """
    # Pair softmin: -1/beta * log(sum_{i<j} exp(-beta * d_ij/2))
    # But we only have D (distances). We need to be careful with the sum.
    # For efficiency, we approximate: the softmin is dominated by the smallest few.
    # Use a truncated sum: only consider pairs with d < 2.5 * current_min + some margin
    # Actually for SA we need a smooth value. Let's use the full sum but it's O(N^2) per eval.
    # To keep it O(N) per step, we maintain D incrementally and recompute softmin from D each step.
    # Recomputing softmin from D is O(N^2) but with small constant. For N=50, N^2=2500, fine.
    
    # Wall values
    wall = 0.5 - np.abs(centers)  # shape (N, 3)
    
    # Pair half-distances
    D_half = D * 0.5
    np.fill_diagonal(D_half, np.inf)
    
    # Combine all constraint values into one array
    # To avoid allocating a huge array, compute log-sum-exp in parts
    # vals = [d_ij/2 for i<j] concatenated with [0.5-|c_ik| for all i,k]
    
    # Log-sum-exp for pairs
    # m = min of D_half (off-diagonal)
    D_flat = D_half[np.triu_indices(N, 1)]  # upper triangle
    m_pair = np.min(D_flat)
    sum_exp_pair = np.sum(np.exp(-beta * (D_flat - m_pair)))
    log_soft_pair = -m_pair - np.log(sum_exp_pair) / beta
    
    # Log-sum-exp for walls
    w_flat = wall.flatten()
    m_wall = np.min(w_flat)
    sum_exp_wall = np.sum(np.exp(-beta * (w_flat - m_wall)))
    log_soft_wall = -m_wall - np.log(sum_exp_wall) / beta
    
    # Combine: softmin of two softmins = -1/beta * log(exp(-beta*log_soft_pair) + exp(-beta*log_soft_wall))
    # = -1/beta * log(exp(-beta*log_soft_pair) + exp(-beta*log_soft_wall))
    # Let a = -beta * log_soft_pair, b = -beta * log_soft_wall
    # result = -1/beta * log(exp(a) + exp(b)) = -max(a,b) - 1/beta * log(1 + exp(-|a-b|))
    a = -beta * log_soft_pair
    b = -beta * log_soft_wall
    m_ab = max(a, b)
    result = -m_ab - np.log(1.0 + np.exp(-abs(a - b))) / beta
    
    return result


def _sa_optimize(centers, time_limit=20.0):
    """
    Simulated annealing with smooth soft-min objective.
    Uses distance-identity trick for O(N) distance updates per step.
    """
    start = time.time()
    centers = centers.copy().astype(np.float64)
    rng = np.random.RandomState(42)
    
    # Precompute distance matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(D, np.inf)
    
    # Initial radius
    r_init = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = r_init
    
    # SA parameters
    T_start = 0.01
    T_end = 1e-6
    n_steps = 50000
    # Adaptive: fewer steps if time is tight
    est_time_per_step = 0.0005  # rough estimate in seconds
    n_steps = min(n_steps, int((time_limit * 0.8) / max(est_time_per_step, 1e-4)))
    n_steps = max(n_steps, 5000)
    
    # Beta for soft-min: start moderate, increase over time
    beta_start = 20.0
    beta_end = 200.0
    
    # Temperature schedule: geometric cooling
    if n_steps > 1:
        cool_rate = (T_end / T_start) ** (1.0 / n_steps)
    else:
        cool_rate = 1.0
    
    T = T_start
    step_size = 0.01  # perturbation scale
    
    # Track best
    for step in range(n_steps):
        if time.time() - start > time_limit * 0.95:
            break
        
        beta = beta_start + (beta_end - beta_start) * (step / max(n_steps - 1, 1))
        
        # Current soft radius
        r_cur = _soft_radius(centers, D, beta)
        
        # Choose a random sphere and perturb it
        i = rng.randint(0, N)
        old_pos = centers[i].copy()
        
        # Perturbation: random direction, magnitude scaled by T and step_size
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction /= norm
        mag = step_size * (0.5 + T / T_start * 0.5)
        new_pos = old_pos + direction * mag
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        # Update distances for sphere i using distance identity:
        # |a - b|^2 = |a - c|^2 + |b - c|^2 - 2(a-c)·(b-c)
        # We update D[i, j] and D[j, i] for all j != i
        c = centers  # current centers
        # For each j != i:
        # new_D[i,j]^2 = |new_pos - c[j]|^2
        # We can compute directly: (new_pos - c[j]) for all j at once
        diffs = new_pos - c  # shape (N, 3), diffs[i] = new_pos - new_pos = 0
        new_D_i = np.sqrt(np.sum(diffs ** 2, axis=1))
        new_D_i[i] = np.inf
        
        # Old D row for i
        old_D_i = D[i].copy()
        
        # Apply new distances
        D[i, :] = new_D_i
        D[:, i] = new_D_i
        
        # Update center
        centers[i] = new_pos
        
        # New soft radius
        r_new = _soft_radius(centers, D, beta)
        
        # Accept?
        delta = r_new - r_cur  # we want to maximize r, so delta > 0 is good
        if delta >= 0 or rng.random() < np.exp(delta / T):
            # Accept
            if r_new > best_radius:
                best_radius = r_new
                best_centers = centers.copy()
                # Also update the hard radius for best
                hard_r = _compute_radius(best_centers)
                if hard_r > best_radius:
                    best_radius = hard_r
        else:
            # Reject: undo
            centers[i] = old_pos
            D[i, :] = old_D_i
            D[:, i] = old_D_i
        
        # Cool
        T *= cool_rate
        
        # Adapt step size: if acceptance rate is too low, increase; too high, decrease
        # (simplified: just decay step size slowly)
        step_size *= 0.9998
    
    # Final: use best_centers found during SA
    # Do a few final hard-radius evaluations
    final_radius = _compute_radius(best_centers)
    if final_radius < best_radius - 1e-12:
        # The soft radius may have overestimated; use hard radius
        pass
    return best_centers, max(best_radius, final_radius)


def _make_bcc_initial(seed):
    """BCC-inspired initialization: body-centered cubic lattice points."""
    rng = np.random.RandomState(seed)
    # BCC: corners + body centers of sub-cubes
    # Divide cube into 3x3x3 sub-cubes, take body centers (27) + some corners
    m = 3
    sub_size = 1.0 / m
    centers = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Body center of sub-cube
                bx = -0.5 + (ix + 0.5) * sub_size
                by = -0.5 + (iy + 0.5) * sub_size
                bz = -0.5 + (iz + 0.5) * sub_size
                centers.append([bx, by, bz])
    centers = np.array(centers)  # 27 points
    # Add face centers of the main cube (6) and edge midpoints to get to 50
    # Face centers:
    face_centers = [
        [0.0, 0.0, 0.5], [0.0, 0.0, -0.5],
        [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0], [0.0, -0.5, 0.0],
    ]
    centers = np.vstack([centers, np.array(face_centers)])  # 33
    # Add edge midpoints (12 edges)
    edge_midpoints = []
    for axis in range(3):
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                pos = [0.0, 0.0, 0.0]
                pos[axis] = -0.5 + (i + 0.5) * sub_size
                # The other two coords at ±0.5
                # This is getting complex; let's just add random points near existing
    # Simpler: add 17 random points near the existing 33, jittered
    remaining = N - len(centers)
    extra = rng.uniform(-0.45, 0.45, size=(remaining, 3))
    centers = np.vstack([centers, extra])
    centers = centers[:N]
    centers += rng.uniform(-0.002, 0.002, size=centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _make_fcc_initial(seed):
    """FCC-inspired initialization."""
    rng = np.random.RandomState(seed)
    # FCC: face-centered cubic. In a cube, FCC has points at corners and face centers.
    # Scale to fit: FCC lattice constant a, radius = a*sqrt(2)/4 for touching.
    # Use a 2x2x2 FCC: 8 corners + 12 face centers = 20, plus interior
    # Let's use a 3x3x3 grid of FCC unit cells
    m = 3
    a = 1.0 / m  # lattice parameter
    centers = []
    for ix in range(m + 1):
        for iy in range(m + 1):
            for iz in range(m + 1):
                # Corner
                x = -0.5 + ix * a
                y = -0.5 + iy * a
                z = -0.5 + iz * a
                # Only include if within bounds (corners of the big cube)
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                    centers.append([x, y, z])
    # Face centers of each sub-cube
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                x0 = -0.5 + ix * a
                y0 = -0.5 + iy * a
                z0 = -0.5 + iz * a
                # Face centers: (x0+a/2, y0+a/2, z0), etc.
                centers.append([x0 + a/2, y0 + a/2, z0])
                centers.append([x0 + a/2, y0 + a/2, z0 + a])
                centers.append([x0 + a/2, y0, y0 + a/2, z0][:3] if False else [x0 + a/2, y0, z0 + a/2])
                centers.append([x0 + a/2, y0 + a, z0 + a/2])
                centers.append([x0, y0 + a/2, z0 + a/2])
                centers.append([x0 + a, y0 + a/2, z0 + a/2])
    centers = np.array(centers)
    # Remove duplicates and out-of-bounds
    valid = []
    for c in centers:
        if np.all(c >= -0.5) and np.all(c <= 0.5):
            # Check if close to existing
            if not valid:
                valid.append(c)
            else:
                v = np.array(valid)
                if np.min(np.linalg.norm(v - c, axis=1)) > 1e-6:
                    valid.append(c)
    centers = np.array(valid) if valid else np.zeros((N, 3))
    if len(centers) < N:
        extra = rng.uniform(-0.45, 0.45, size=(N - len(centers), 3))
        centers = np.vstack([centers, extra])
    centers = centers[:N]
    centers += rng.uniform(-0.002, 0.002, size=centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _make_grid_initial(seed):
    """Jittered grid initialization."""
    rng = np.random.RandomState(seed)
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += rng.uniform(-0.005, 0.005, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _make_random_initial(seed):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.49, 0.49, size=(N, 3))


def _quick_refine(centers, time_limit=5.0):
    """Quick local refinement using a few SA steps with high beta."""
    start = time.time()
    centers = centers.copy().astype(np.float64)
    rng = np.random.RandomState(99)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(D, np.inf)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(centers)
    
    T = 1e-4
    n_steps = min(5000, int((time_limit * 0.8) / 0.0003))
    n_steps = max(n_steps, 500)
    beta = 100.0
    step_size = 0.002
    
    for step in range(n_steps):
        if time.time() - start > time_limit * 0.95:
            break
        
        r_cur = _soft_radius(centers, D, beta)
        
        i = rng.randint(0, N)
        old_pos = centers[i].copy()
        old_D_i = D[i].copy()
        
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction /= norm
        mag = step_size * (0.5 + T / 1e-4 * 0.5)
        new_pos = old_pos + direction * mag
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        diffs = new_pos - centers
        new_D_i = np.sqrt(np.sum(diffs ** 2, axis=1))
        new_D_i[i] = np.inf
        
        D[i, :] = new_D_i
        D[:, i] = new_D_i
        centers[i] = new_pos
        
        r_new = _soft_radius(centers, D, beta)
        delta = r_new - r_cur
        if delta >= 0 or rng.random() < np.exp(delta / T):
            hard_r = _compute_radius(centers)
            if hard_r > best_radius:
                best_radius = hard_r
                best_centers = centers.copy()
        else:
            centers[i] = old_pos
            D[i, :] = old_D_i
            D[:, i] = old_D_i
        
        T *= 0.999
    
    return best_centers, best_radius


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
    best_centers = None
    best_radius = 0.0
    
    # Generate diverse initializations
    initializations = []
    
    # BCC-inspired
    for seed in [42, 123]:
        initializations.append(_make_bcc_initial(seed))
    
    # FCC-inspired
    for seed in [7, 8]:
        initializations.append(_make_fcc_initial(seed))
    
    # Grid
    for seed in [1, 2]:
        initializations.append(_make_grid_initial(seed))
    
    # Random
    for seed in [10, 20]:
        initializations.append(_make_random_initial(seed))
    
    # Optimize each initialization with SA
    for init_idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        remaining = deadline - time.time()
        n_remaining = len(initializations) - init_idx
        time_per_init = min(18.0, remaining / (n_remaining + 1))
        
        if time_per_init < 5.0:
            break
        
        try:
            optimized, radius = _sa_optimize(init_centers, time_limit=time_per_init)
        except Exception:
            continue
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()
    
    # Final refinement on the best solution
    if best_centers is not None and time.time() < deadline - 3.0:
        time_left = deadline - time.time() - 3.0
        if time_left > 3.0:
            try:
                final_centers, final_radius = _quick_refine(best_centers, time_limit=time_left)
                if final_radius > best_radius:
                    best_radius = final_radius
                    best_centers = final_centers
            except Exception:
                pass
    
    if best_centers is None:
        # Fallback: simple grid
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    return best_centers
# EVOLVE-BLOCK-END
