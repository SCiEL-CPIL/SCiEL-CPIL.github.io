import time
import numpy as np
from scipy.spatial.distance import cdist

N = 14
D = 3
DEADLINE = 145.0  # Safety margin against 150s limit
_t0 = None

def _true_score(pts):
    """Calculate the raw score: (max_dist / min_dist)^2"""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_score(pts, beta=100.0):
    """
    Smooth approximation of (max/min)^2.
    Uses log-sum-exp to approximate max and min.
    Objective to minimize: soft_max - soft_min + regularization for distinctness.
    Actually, we want to minimize ratio.
    log(ratio) = log(max) - log(min).
    min(log(ratio)) is equivalent to min(ratio).
    Let's minimize: (soft_max_dist / soft_min_dist)^2 is still non-smooth if we divide.
    Better: minimize log(soft_max) - log(soft_min).
    This is smooth and convex-ish in some regions.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    # Soft max: (1/beta) * log(sum(exp(beta * d)))
    # Soft min: - (1/beta) * log(sum(exp(-beta * d)))
    # To avoid overflow, subtract max from d for exp
    d_max = d.max()
    d_min = d.min()
    
    # If points are too close, penalize heavily
    if d_min < 1e-8:
        return 1e10

    # Scale d to be in a reasonable range for exp
    # We want to approximate max(d) and min(d)
    # soft_max = d_max + (1/beta) * log(sum(exp(beta*(d - d_max))))
    # soft_min = d_min - (1/beta) * log(sum(exp(-beta*(d - d_min))))
    
    # However, d contains inf on diagonal. We masked it.
    # d is N x N.
    # For soft max:
    exp_arg_max = beta * (d - d_max)
    # Clip to prevent underflow issues in sum, though exp(-large) is 0
    sum_exp_max = np.sum(np.exp(exp_arg_max))
    soft_max = d_max + (1.0 / beta) * np.log(sum_exp_max)
    
    # For soft min:
    exp_arg_min = -beta * (d - d_min)
    sum_exp_min = np.sum(np.exp(exp_arg_min))
    soft_min = d_min - (1.0 / beta) * np.log(sum_exp_min)
    
    # We want to minimize (max/min)^2.
    # Equivalent to minimizing 2 * (log(max) - log(min))
    # Since distances are positive:
    score = 2.0 * (np.log(soft_max) - np.log(soft_min))
    
    # Add a small penalty if points are not distinct
    if d_min < 0.01:
        score += (0.01 - d_min) ** 2 * 1000.0
        
    return score

def _generate_icosahedron_orbits():
    """
    Generate 14 points using the symmetry of the icosahedron.
    The icosahedron has 12 vertices. We need 14.
    A good strategy is to place 12 points near the icosahedron vertices
    and 2 points near the poles (or other high-symmetry positions).
    
    Alternatively, use a known good configuration for n=14.
    Let's start with a Fibonacci sphere and then refine.
    """
    # Fibonacci sphere
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5**0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _generate_random_cluster(rng):
    """Generate random points, potentially clustered to start."""
    return rng.uniform(-1, 1, size=(N, D))

def _simulated_annealing(pts0, rng, start_temp=0.05, end_temp=0.001, steps=2000):
    """
    Run Simulated Annealing on the point configuration.
    """
    pts = pts0.copy()
    current_score = _soft_score(pts, beta=50.0)
    best_pts = pts.copy()
    best_score = current_score
    
    # Adaptive beta: start low (smooth), end high (sharp)
    # Actually, for SA, we usually keep objective fixed or anneal it.
    # Here we anneal the temperature of the moves, and maybe the sharpness of the objective.
    
    T = start_temp
    decay_rate = (end_temp / start_temp) ** (1.0 / steps)
    
    for i in range(steps):
        if time.time() - _t0 > DEADLINE:
            break
            
        # Generate a move: perturb a random subset of points
        # Perturb all points slightly, or random few?
        # Perturbing all points with small noise is like gradient descent with noise.
        # Let's perturb a random subset of points with larger noise to escape local minima.
        num_perturb = rng.integers(1, N + 1)
        indices = rng.choice(N, size=num_perturb, replace=False)
        
        # Noise magnitude scales with T
        noise_mag = T * 0.5
        new_pts = pts.copy()
        new_pts[indices] += rng.normal(0, noise_mag, size=(num_perturb, D))
        
        new_score = _soft_score(new_pts, beta=100.0) # Use sharper objective for evaluation
        
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta < 0 or rng.random() < np.exp(-delta / T):
            pts = new_pts
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_pts = pts.copy()
                
        T *= decay_rate
        
    return best_pts, best_score

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_raw_score = np.inf
    
    def consider(pts):
        nonlocal best_pts, best_raw_score
        if pts is None:
            return
        # Normalize points to unit sphere scale to help with SA steps
        # Actually, SA works in absolute coordinates. 
        # Let's normalize the initial points to have norm ~1.
        norm = np.linalg.norm(pts, axis=1)
        mask = norm > 1e-6
        pts = pts.copy()
        pts[mask] /= norm[mask, None]
        
        raw = _true_score(pts)
        if raw < best_raw_score:
            best_raw_score = raw
            best_pts = pts.copy()

    # Initial configurations
    starts = []
    starts.append(_generate_icosahedron_orbits())
    
    # Add some random starts
    for _ in range(5):
        starts.append(_generate_random_cluster(rng))
    
    # Run SA on each start
    # We have ~140 seconds.
    # Let's allocate time per start.
    num_starts = len(starts)
    time_per_start = (DEADLINE - (time.time() - _t0)) / num_starts
    
    for i, start_pts in enumerate(starts):
        if time.time() - _t0 > DEADLINE:
            break
            
        # Run SA
        # Steps depends on time available
        # Estimate time per step: ~1ms? cdist for 14x14 is fast.
        # Let's run as many steps as possible within time_per_start
        # But we need to reserve time for final polish.
        
        sa_steps = 1500
        # Adjust steps based on time left
        if i == 0:
            # First start gets more time
            sa_steps = 3000
            
        best_sa_pts, _ = _simulated_annealing(start_pts, rng, start_temp=0.05, end_temp=0.001, steps=sa_steps)
        consider(best_sa_pts)
        
        # Also try a second SA run with different params if time permits
        if time.time() - _t0 < DEADLINE - 10:
            # Perturb the best found so far and run a shorter SA
            if best_pts is not None:
                perturbed = best_pts + rng.normal(0, 0.1, size=(N, D))
                best_sa_pts2, _ = _simulated_annealing(perturbed, rng, start_temp=0.02, end_temp=0.001, steps=1000)
                consider(best_sa_pts2)

    # Final Polish: Local optimization using Nelder-Mead on the raw score
    # This helps fix the "soft" artifacts and squeeze out the last bit.
    if best_pts is not None and time.time() - _t0 < DEADLINE - 5:
        from scipy.optimize import minimize
        
        def raw_obj(x):
            pts = x.reshape(N, D)
            return _true_score(pts)
            
        x0 = best_pts.flatten()
        try:
            res = minimize(raw_obj, x0, method='Nelder-Mead', 
                           options=dict(maxfev=500, maxiter=500, xatol=1e-8, fatol=1e-8))
            final_pts = res.x.reshape(N, D)
            consider(final_pts)
        except Exception:
            pass

    if best_pts is None:
        best_pts = _generate_icosahedron_orbits()
        
    return best_pts
