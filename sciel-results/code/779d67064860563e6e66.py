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

def _simulated_annealing(points, time_budget, rng):
    """
    Simulated Annealing local search.
    Returns (best_points, best_score).
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
    T = 1e-3
    T_min = 1e-6
    cooling_rate = 0.9995
    step_scale = 0.05
    
    while time.time() - start_time < time_budget:
        # Pick a point to move
        i = rng.integers(n)
        
        # Random perturbation
        # Step size decreases as T decreases? Or stays constant?
        # Usually, step size is constant or slightly decreasing.
        step_size = step_scale * (0.5 + 0.5 * (T / 1e-3)) 
        delta = rng.standard_normal(d) * step_size
        
        candidate = current.copy()
        candidate[i] += delta
        
        cand_score = _score(candidate)
        
        delta_E = cand_score - current_score
        
        # Accept if better, or with probability exp(delta_E / T)
        # Note: delta_E is negative for worse moves.
        # If delta_E is very negative, exp(delta_E/T) is ~0.
        if delta_E > 0 or (T > 1e-15 and rng.random() < np.exp(delta_E / T)):
            current = candidate
            current_score = cand_score
            
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
                
        T *= cooling_rate
        if T < T_min:
            # Reheat
            T = 1e-3
            # Perturb current state to escape local minimum
            # Move each point by a small random amount
            current += rng.standard_normal((n, d)) * 0.1
            # Re-normalize?
            norm = np.max(np.linalg.norm(current, axis=1))
            if norm > 1e-10:
                current = current / norm
            current_score = _score(current)
            if current_score > best_score:
                best = current.copy()
                best_score = current_score

    return best, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2

    # Hard deadline: 600s wall-clock, leave 30s margin
    deadline = time.time() + 570

    best_points = None
    best_score = -1.0

    rng = np.random.default_rng(42)

    initial_configs = []

    # Config 1: Triangular Lattice (High Quality Start)
    initial_configs.append(_generate_triangular_16())

    # Config 2: Two concentric regular octagons
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r_outer = 1.0
    r_inner = 0.5
    pts1 = np.vstack([
        np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)]),
        np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    ])
    initial_configs.append(pts1)

    # Config 3: Random
    initial_configs.append(rng.standard_normal((n, d)))

    # Config 4: Square grid 4x4 with jitter
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts4 = np.column_stack([gx.ravel(), gy.ravel()])
    pts4 += rng.standard_normal((n, d)) * 0.05
    initial_configs.append(pts4)

    # Config 5: Circle with 16 points
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
        time_budget = min(remaining - 5, 150)  # Cap per-config time
        improved_pts, improved_score = _simulated_annealing(init, time_budget, rng)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    # --- Phase 2: Additional random restarts if time permits ---
    while time.time() < deadline - 30:
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
        if remaining < 10:
            break
        time_budget = min(remaining - 10, 60)
        improved_pts, improved_score = _simulated_annealing(init, time_budget, rng)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    if best_points is None:
        return np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
