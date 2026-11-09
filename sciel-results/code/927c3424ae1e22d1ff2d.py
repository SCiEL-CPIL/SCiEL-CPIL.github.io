# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _soft_score(points, beta):
    """
    Compute a smooth surrogate for R using log-sum-exp for min and max.
    R_soft = (soft_min / soft_max)^2
    soft_min = -1/beta * log(sum(exp(-beta * d_i)))
    soft_max = 1/beta * log(sum(exp(beta * d_i)))
    """
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    # Numerical stability for log-sum-exp
    # soft_min
    neg_d = -beta * dists
    max_neg_d = np.max(neg_d)
    log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
    soft_min = -log_sum_exp_neg / beta
    
    # soft_max
    pos_d = beta * dists
    max_pos_d = np.max(pos_d)
    log_sum_exp_pos = max_pos_d + np.log(np.sum(np.exp(pos_d - max_pos_d)))
    soft_max = log_sum_exp_pos / beta
    
    if soft_max < 1e-15:
        return 0.0
        
    ratio = soft_min / soft_max
    return ratio * ratio

def _generate_triangular_16():
    """Generate a 16-point triangular lattice configuration (1-2-3-4-3-2-1)."""
    rows = [1, 2, 3, 4, 3, 2, 1]
    points = []
    s = 1.0
    dy = s * np.sqrt(3) / 2
    for i, count in enumerate(rows):
        y = -i * dy
        for j in range(count):
            x = (j - (count - 1) / 2.0) * s
            points.append([x, y])
    pts = np.array(points)
    return pts

def _generate_hexagonal_16():
    """
    Generate a hexagonal packing configuration.
    Center (1) + Ring 1 (6) + Ring 2 (9) = 16.
    Ring 2 is arranged to fit in the gaps of Ring 1 extended, or just a larger ring.
    For 16 points, a common tight packing is 1 + 6 + 9? 
    Actually, 1+6=7. Next shell is 12. 7+12=19.
    So we take 1 center, 6 neighbors, and 9 from the next shell.
    """
    points = [[0.0, 0.0]]
    r1 = 1.0
    for i in range(6):
        angle = np.pi / 3 * i
        points.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    
    # Need 9 more. The next shell in hex packing has radius sqrt(3) * r1? No, distance between centers is 1.
    # Ring 2 vertices are at distance sqrt(3) from center? No.
    # In hex lattice, neighbors of neighbors are at distance sqrt(3) (if angle 60 deg) or 1 (if collinear).
    # Let's just place 9 points on a circle of radius sqrt(3) + epsilon?
    # Or better, use the actual hex lattice points.
    # Lattice points: (i + j/2, j*sqrt(3)/2)
    # Let's generate a large grid of hex points and pick the 16 closest to origin.
    hex_pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Sort by distance to origin
    dists = np.linalg.norm(hex_pts, axis=1)
    idx = np.argsort(dists)
    # Take first 16
    pts = hex_pts[idx[:16]]
    return pts

def _simulated_annealing(points, time_budget, rng, use_soft=False):
    """
    Simulated Annealing local search.
    If use_soft is True, uses soft surrogate for acceptance in early phases?
    Actually, let's just use exact score but maybe soft for exploration?
    The directive suggests testing smooth surrogate.
    Let's use soft score for the *acceptance criterion* in the early high-temp phase,
    and exact score for tracking the best.
    """
    n, d = points.shape
    current = points.copy()
    
    # Normalize to unit scale for numerical stability
    norm = np.max(np.linalg.norm(current, axis=1))
    if norm > 1e-10:
        current = current / norm
        
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 1e-2
    T_min = 1e-6
    cooling_rate = 0.999
    step_scale = 0.1
    
    # Softness parameter beta. Start low (sharp), increase? Or start high (smooth)?
    # High beta -> sharp (exact). Low beta -> smooth.
    # Let's start with a moderate beta for smoothing, e.g., beta=10.
    # As T drops, we can increase beta to sharpen the landscape.
    beta = 5.0 
    beta_max = 100.0
    
    while time.time() - start_time < time_budget:
        # Pick a point to move
        i = rng.integers(n)
        
        # Random perturbation
        step_size = step_scale * (0.5 + 0.5 * (T / 1e-2)) 
        delta = rng.standard_normal(d) * step_size
        
        candidate = current.copy()
        candidate[i] += delta
        
        # Use soft score for acceptance to navigate flat regions/plateaus
        if use_soft and T > 1e-4:
             cand_score_soft = _soft_score(candidate, beta)
             cur_score_soft = _soft_score(current, beta)
             delta_E = cand_score_soft - cur_score_soft
        else:
             cand_score = _score(candidate)
             cur_score = current_score
             delta_E = cand_score - cur_score
        
        # Accept if better, or with probability exp(delta_E / T)
        if delta_E > 0 or (T > 1e-15 and rng.random() < np.exp(delta_E / T)):
            current = candidate
            # Update exact score for best tracking
            current_score = _score(current)
            
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
                
        T *= cooling_rate
        if T < T_min:
            # Reheat
            T = 1e-2
            # Perturb current state to escape local minimum
            current += rng.standard_normal((n, d)) * 0.2
            # Re-normalize
            norm = np.max(np.linalg.norm(current, axis=1))
            if norm > 1e-10:
                current = current / norm
            current_score = _score(current)
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        # Increase beta (sharpen) as T decreases
        if beta < beta_max:
            beta *= 1.001

    return best, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2

    # Hard deadline: 630s wall-clock, leave 30s margin
    deadline = time.time() + 600

    best_points = None
    best_score = -1.0

    rng = np.random.default_rng(42)

    initial_configs = []

    # Config 1: Triangular Lattice (High Quality Start)
    initial_configs.append(_generate_triangular_16())

    # Config 2: Hexagonal Packing (1 + 6 + 9)
    initial_configs.append(_generate_hexagonal_16())

    # Config 3: Two concentric regular octagons
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r_outer = 1.0
    r_inner = 0.5
    pts1 = np.vstack([
        np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)]),
        np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    ])
    initial_configs.append(pts1)

    # Config 4: Random
    initial_configs.append(rng.standard_normal((n, d)))

    # Config 5: Square grid 4x4 with jitter
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts4 = np.column_stack([gx.ravel(), gy.ravel()])
    pts4 += rng.standard_normal((n, d)) * 0.05
    initial_configs.append(pts4)

    # Config 6: Circle with 16 points
    angles_c = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts5 = np.column_stack([np.cos(angles_c), np.sin(angles_c)])
    initial_configs.append(pts5)

    # Evaluate and improve each initial config
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 60:
            break

        # Normalize to unit scale
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale

        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        # Local search (Simulated Annealing)
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_budget = min(remaining - 5, 120)  # Cap per-config time
        improved_pts, improved_score = _simulated_annealing(init, time_budget, rng, use_soft=True)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    # --- Phase 2: Refine the best solution found so far ---
    if best_points is not None and time.time() < deadline - 30:
        remaining = deadline - time.time()
        if remaining > 10:
            # Run SA again on the best point, maybe with different parameters or just more time
            # Use use_soft=False here to focus on exact improvement? Or keep soft?
            # Let's try a mix: run one more pass.
            time_budget = min(remaining - 10, 180)
            improved_pts, improved_score = _simulated_annealing(best_points, time_budget, rng, use_soft=False)
            if improved_score > best_score:
                best_score = improved_score
                best_points = improved_pts.copy()

    # --- Phase 3: Additional random restarts if time permits ---
    while time.time() < deadline - 20:
        init = rng.standard_normal((n, d))
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale

        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        remaining = deadline - time.time()
        if remaining < 15:
            break
        time_budget = min(remaining - 15, 40)
        improved_pts, improved_score = _simulated_annealing(init, time_budget, rng, use_soft=True)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    if best_points is None:
        return np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
