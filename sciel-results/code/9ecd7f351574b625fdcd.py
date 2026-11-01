"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    dist_sq_upper = dist_sq[iu]
    
    if len(dist_sq_upper) == 0:
        return 1e10
    
    min_d_sq = np.min(dist_sq_upper)
    max_d_sq = np.max(dist_sq_upper)
    
    if min_d_sq <= 1e-12:
        return 1e10
    
    return max_d_sq / min_d_sq

def _simulated_annealing(points, iterations=5000, T_start=1.0, T_end=1e-5, alpha=0.9995, kick_interval=1000, kick_scale=0.5):
    """
    Perform Simulated Annealing on the points.
    """
    current_points = points.copy()
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    T = T_start
    rng = np.random.default_rng()
    
    for i in range(iterations):
        # Generate a random perturbation
        # Scale perturbation by temperature to allow larger jumps when hot
        noise = rng.normal(0, T * 0.1, size=current_points.shape)
        new_points = current_points + noise
        
        new_score = _compute_score(new_points)
        
        # Acceptance probability
        if new_score < current_score:
            current_points = new_points
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            delta = new_score - current_score
            # Avoid overflow in exp
            if delta > 1000 * T:
                accept = False
            else:
                accept = np.random.rand() < np.exp(-delta / T)
            
            if accept:
                current_points = new_points
                current_score = new_score
        
        # Cooling
        T = max(T * alpha, T_end)
        
        # Kick strategy: occasionally perturb significantly to escape local minima
        if i > 0 and i % kick_interval == 0 and T > 0.1:
            kick = rng.normal(0, kick_scale, size=current_points.shape)
            current_points = current_points + kick
            current_score = _compute_score(current_points)
            # Reheat slightly after kick
            T = max(T * 2.0, 0.1)

    return best_points, best_score

def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses Simulated Annealing with multiple restarts and a final Nelder-Mead polish.
    """
    best_score_global = 1e10
    best_points_global = None
    
    # Time budget management
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Number of restarts depends on time
    # Each SA run takes ~1-2s depending on iterations
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization on unit sphere
        rng = np.random.default_rng(restart * 100 + 42)
        init_points = rng.normal(0, 1, size=(N, D))
        init_points /= np.linalg.norm(init_points, axis=1, keepdims=True)
        
        # Run Simulated Annealing
        # Parameters tuned for N=14, D=3
        sa_points, sa_score = _simulated_annealing(
            init_points,
            iterations=3000,
            T_start=0.5,
            T_end=1e-6,
            alpha=0.999,
            kick_interval=500,
            kick_scale=0.3
        )
        
        if sa_score < best_score_global:
            best_score_global = sa_score
            best_points_global = sa_points
        
        # Optional: Light polish with Nelder-Mead if time permits
        # We do this only for the current best to save time
        if time.time() - start_time > time_limit - 10:
            continue
            
        try:
            res = minimize(
                lambda x: _compute_score(x.reshape(N, D)),
                best_points_global.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            polish_score = _compute_score(res.x.reshape(N, D))
            if polish_score < best_score_global:
                best_score_global = polish_score
                best_points_global = res.x.reshape(N, D)
        except Exception:
            pass

    # Final heavy polish on the global best
    if best_points_global is not None and time.time() - start_time < time_limit:
        try:
            res = minimize(
                lambda x: _compute_score(x.reshape(N, D)),
                best_points_global.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            final_score = _compute_score(res.x.reshape(N, D))
            if final_score < best_score_global:
                best_points_global = res.x.reshape(N, D)
        except Exception:
            pass

    # Fallback
    if best_points_global is None:
        best_points_global = np.random.randn(N, D)
        
    return best_points_global
# EVOLVE-BLOCK-END
