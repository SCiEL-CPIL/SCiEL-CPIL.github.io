import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)

def _objective(x):
    """
    Objective function to minimize for the final polish.
    We want to maximize the minimum distance.
    We approximate min(d_ij) using a soft-min.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Soft min of distances
    k = 50.0
    y = -k * d
    y_flat = y.flatten()
    y_valid = y_flat[np.isfinite(y_flat)]
    
    if len(y_valid) == 0:
        return 0.0
    
    max_y = np.max(y_valid)
    exp_sum = np.sum(np.exp(y_valid - max_y))
    log_sum_exp = max_y + np.log(exp_sum)
    
    d_min_soft = -log_sum_exp / k
    
    # Boundary constraint
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_violation = np.maximum(0, norms - 1.0)
    boundary_penalty = np.sum(boundary_violation ** 2)
    
    return -d_min_soft + 100.0 * boundary_penalty

def _gradient(x):
    """
    Analytical gradient for the soft-min objective.
    """
    pts = x.reshape(-1, 2)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    k = 50.0
    y = -k * d
    y_flat = y.flatten()
    y_valid = y_flat[np.isfinite(y_flat)]
    
    if len(y_valid) == 0:
        return np.zeros_like(x)
    
    max_y = np.max(y_valid)
    exp_terms = np.exp(y - max_y)
    # Zero out diagonal
    np.fill_diagonal(exp_terms, 0.0)
    
    S = np.sum(exp_terms)
    
    if S == 0:
        return np.zeros_like(x)
    
    grad = np.zeros_like(pts)
    
    # Pairwise contribution
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    safe_d = np.where(d > 1e-8, d, 1.0)
    term = exp_terms[:, :, np.newaxis] * diff / safe_d[:, :, np.newaxis]
    grad_pair = np.sum(term, axis=1)
    grad_pair /= S
    
    # Boundary contribution
    norms = np.hypot(pts[:, 0], pts[:, 1])
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    mask = norms > 1.0
    grad_boundary = np.zeros_like(pts)
    if np.any(mask):
        grad_boundary[mask] = 200.0 * (norms[mask, np.newaxis] - 1.0) * pts[mask] / safe_norms[mask, np.newaxis]
    
    total_grad = grad_pair + grad_boundary
    return total_grad.flatten()

def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select N closest to origin
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _simulated_annealing(start_points, time_limit):
    """
    Simulated Annealing to escape local minima.
    Move: pick random point, jitter it.
    Acceptance: Metropolis criterion.
    """
    current = start_points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.05 # Initial temperature
    T_min = 0.001
    cooling = 0.9995
    
    rng = np.random.default_rng(42)
    
    while time.time() < time_limit:
        # Pick a random point to move
        i = rng.integers(0, N)
        old_pos = current[i].copy()
        
        # Jitter
        noise = rng.normal(0, T, 2)
        new_pos = old_pos + noise
        
        # Ensure new pos is inside circle (or penalize heavily)
        # For SA, we can just allow it but the score will drop, or clip?
        # Clipping distorts the move. Let's allow it, score will handle it.
        # Actually, if it goes outside, score becomes 0. That's a huge penalty.
        # Better to reject moves that go outside immediately? No, let score decide.
        
        current[i] = new_pos
        new_score = _score(current)
        
        # Metropolis acceptance
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        else:
            # Accept worse with probability exp((new - current) / T)
            delta = new_score - current_score
            if delta < 0:
                p = np.exp(delta / T)
                if rng.random() < p:
                    current_score = new_score
                else:
                    # Reject: revert
                    current[i] = old_pos
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best, best_score

def _local_optimize(points):
    """
    Run L-BFGS-B on the points.
    """
    x0 = points.flatten()
    try:
        res = minimize(
            _objective, 
            x0, 
            method='L-BFGS-B', 
            jac=_gradient,
            options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        optimized_points = res.x.reshape(-1, 2)
        
        # Check validity
        norms = np.hypot(optimized_points[:, 0], optimized_points[:, 1])
        if np.any(norms > 1.0):
            # Clip if slightly outside
            mask = norms > 1.0
            optimized_points[mask] = optimized_points[mask] / norms[mask, np.newaxis] * 0.999
            
        return optimized_points
    except Exception:
        return points

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    initializations = []
    
    # Hexagonal
    for spacing in [0.20, 0.22, 0.24, 0.26, 0.28]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            initializations.append(pts)
            
    # Ring
    def _ring_initial():
        points = []
        points.append([0.0, 0.0])
        radii = [0.3, 0.55, 0.75, 0.9]
        counts = [6, 12, 18, 13]
        for r, n in zip(radii, counts):
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            for a in angles:
                points.append([r * np.cos(a), r * np.sin(a)])
        return np.array(points[:N])
    initializations.append(_ring_initial())
    
    # Random
    np.random.seed(42)
    for _ in range(3):
        points = []
        for _ in range(N):
            while True:
                x = np.random.uniform(-0.9, 0.9)
                y = np.random.uniform(-0.9, 0.9)
                if x * x + y * y <= 0.81:
                    points.append([x, y])
                    break
        initializations.append(np.array(points))
        
    # Strategy:
    # 1. For each init, do a quick local optimize to get a good local min.
    # 2. Then run Simulated Annealing starting from that local min to try to escape.
    # 3. Keep track of the global best.
    
    # Time budget allocation
    # We have ~145s.
    # Let's spend 60% on SA and 40% on polishing the best found.
    # Or interleave?
    # Let's do a few SA runs.
    
    sa_time_per_run = 10.0 # seconds per SA run
    max_sa_runs = int((TIME_LIMIT - 20) / sa_time_per_run) # reserve 20s for polish
    
    for i, init_points in enumerate(initializations):
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        # Step 1: Quick local optimize
        opt_points = _local_optimize(init_points)
        score_opt = _score(opt_points)
        
        if score_opt > best_score:
            best_score = score_opt
            best_points = opt_points.copy()
            
        # Step 2: Simulated Annealing from this local min
        # Only if we have time
        if time.time() - START_TIME > TIME_LIMIT - 20:
            break
            
        sa_end_time = time.time() + sa_time_per_run
        sa_points, sa_score = _simulated_annealing(opt_points, sa_end_time)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
            
        # If we found a significantly better solution, we might want to polish it immediately?
        # Or just keep it for the final polish phase.
        
    # Final Polish Phase
    # Use the remaining time to polish the best_points found so far.
    if best_points is not None:
        remaining_time = TIME_LIMIT - (time.time() - START_TIME)
        if remaining_time > 5:
            # Run a more aggressive local optimization
            # We can try multiple restarts of L-BFGS-B from slightly perturbed best_points
            # Or just run L-BFGS-B with higher iterations.
            
            # Let's try a few perturbation + polish cycles
            perturbations = [0.01, 0.005, 0.001]
            for p in perturbations:
                if time.time() - START_TIME > TIME_LIMIT - 2:
                    break
                perturbed = best_points + np.random.normal(0, p, (N, 2))
                # Ensure inside
                norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
                mask = norms > 1.0
                if np.any(mask):
                    perturbed[mask] = perturbed[mask] / norms[mask, np.newaxis] * 0.999
                    
                polished = _local_optimize(perturbed)
                score_p = _score(polished)
                if score_p > best_score:
                    best_score = score_p
                    best_points = polished.copy()
                    
            # One final high-precision polish
            if time.time() - START_TIME > TIME_LIMIT - 5:
                pass # Skip if no time
            else:
                try:
                    res2 = minimize(
                        _objective, 
                        best_points.flatten(), 
                        method='L-BFGS-B', 
                        jac=_gradient,
                        options={'maxiter': 2000, 'gtol': 1e-10, 'ftol': 1e-12}
                    )
                    polished_points = res2.x.reshape(-1, 2)
                    norms = np.hypot(polished_points[:, 0], polished_points[:, 1])
                    if np.any(norms > 1.0):
                        mask = norms > 1.0
                        polished_points[mask] = polished_points[mask] / norms[mask, np.newaxis] * 0.999
                    score2 = _score(polished_points)
                    if score2 > best_score:
                        best_score = score2
                        best_points = polished_points.copy()
                except Exception:
                    pass

    # Fallback
    if best_points is None:
        best_points = _hexagonal_initial(0.25)
        if best_points is None:
            best_points = np.zeros((N, 2))

    return best_points
