import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 145.0 # Safety margin for 150s kill

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    if len(pts) < 2:
        return np.inf
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_ratio_obj(pts, tau=100.0):
    """
    Soft approximation of the ratio objective for differentiable optimization.
    We want to minimize (max/min)^2.
    log(ratio^2) = 2 * (log(max) - log(min)).
    Soft max/min can be approximated using log-sum-exp.
    Let D be the matrix of squared distances.
    We minimize 2 * (soft_log_max(D) - soft_log_min(D)).
    soft_log_max(D) = (1/tau) * log(sum(exp(tau * D)))
    soft_log_min(D) = -(1/tau) * log(sum(exp(-tau * D)))
    
    However, standard log-sum-exp can overflow. We use the stable version.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    
    # Stable log-sum-exp for max
    max_d = np.max(d)
    log_sum_exp_max = max_d + np.log(np.sum(np.exp(d - max_d)))
    
    # Stable log-sum-exp for min
    min_d = np.min(d)
    # log(sum(exp(-tau*d))) = -min_d + log(sum(exp(-tau*(d-min_d))))
    # soft_min = -(1/tau) * [ -min_d + log(sum(exp(-tau*(d-min_d)))) ]
    #          = min_d/tau - (1/tau)*log(sum(exp(-tau*(d-min_d))))
    # Actually, let's just use the direct formula with clipping to avoid overflow if tau is large.
    # For tau=100, exp(100*10) overflows. So stable version is required.
    
    # soft_max = (1/tau) * log(sum(exp(tau * d)))
    # soft_min = -(1/tau) * log(sum(exp(-tau * d)))
    
    # Using stable forms:
    # A = tau * d
    A = tau * d
    max_A = np.max(A)
    log_sum_exp_A = max_A + np.log(np.sum(np.exp(A - max_A)))
    soft_max = log_sum_exp_A / tau
    
    # B = -tau * d
    B = -tau * d
    max_B = np.max(B)
    log_sum_exp_B = max_B + np.log(np.sum(np.exp(B - max_B)))
    soft_min = -log_sum_exp_B / tau
    
    # Objective: (soft_max / soft_min)^2
    # We minimize log of this to keep numbers stable: 2 * (log(soft_max) - log(soft_min))
    # But soft_min and soft_max are positive distances.
    if soft_min < 1e-9 or soft_max < 1e-9:
        return 1e10
        
    return (soft_max / soft_min) ** 2

def _init_icosahedral():
    """
    Generate 14 points based on icosahedron symmetry.
    Icosahedron has 12 vertices. We need 14.
    Strategy: Take 12 vertices of a regular icosahedron.
    Add 2 points at the origin? No, distinct points.
    Add 2 points slightly off-center or use a different symmetric configuration.
    
    A better initialization for 14 points:
    Use the 12 vertices of the icosahedron.
    Add 2 points on the z-axis (poles) if the icosahedron is aligned that way?
    Standard icosahedron vertices:
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    
    Let's just use 12 icosahedron vertices + 2 random points near the center or poles,
    or better: 14 points as a subset of a larger symmetric set?
    
    Actually, the optimal configuration for 14 points on a sphere is likely close to
    12 icosahedron vertices + 2 additional points inserted into the largest triangular gaps.
    The largest gaps in an icosahedron are the 20 triangular faces.
    We can place 2 points at the centroids of two opposite faces? Or just random perturbations.
    
    For SA, the initialization doesn't need to be perfect, just reasonable.
    Let's use 12 icosahedron vertices + 2 points at (0,0,1.1) and (0,0,-1.1) (stretched poles).
    """
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Add 2 points on the z-axis, slightly outside to avoid collision
    p13 = np.array([0, 0, 1.2])
    p14 = np.array([0, 0, -1.2])
    
    pts = np.vstack([verts, p13, p14])
    # Normalize all to unit sphere to start on surface
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _simulated_annealing(initial_pts, t_start=1.0, t_end=1e-6, alpha=0.999, deadline=None):
    """
    Simulated Annealing to minimize the ratio.
    We allow points to move in 3D space (x,y,z).
    """
    if deadline is None:
        deadline = time.time() + 10.0
        
    pts = initial_pts.copy()
    current_score = _true_score(pts)
    best_pts = pts.copy()
    best_score = current_score
    
    temp = t_start
    step_size = 0.1
    
    # Precompute number of pairs for efficiency if needed, but N=14 is small.
    # 91 pairs.
    
    while temp > t_end:
        if time.time() > deadline:
            break
            
        # Generate a perturbation
        # Move a random point by a random vector
        idx = np.random.randint(N)
        noise = np.random.normal(0, step_size, D)
        new_pts = pts.copy()
        new_pts[idx] += noise
        
        # Optional: keep on sphere? 
        # The problem is in Euclidean space. The optimal is likely on the sphere, 
        # but radial movement might help escape local minima. 
        # Let's allow radial movement but re-center to prevent drift.
        new_pts = new_pts - np.mean(new_pts, axis=0)
        
        new_score = _true_score(new_pts)
        
        # Acceptance probability
        if new_score < current_score:
            pts = new_pts
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_pts = pts.copy()
        else:
            delta = new_score - current_score
            if delta < 0: # Should not happen if new_score > current_score
                pass
            else:
                # P = exp(-delta / temp)
                # Since scores are ratios > 1, delta is small positive usually.
                # If delta is large, exp is 0.
                p = np.exp(-delta / temp)
                if np.random.random() < p:
                    pts = new_pts
                    current_score = new_score
        
        # Update temperature
        temp *= alpha
        # Slightly decrease step size
        step_size *= 0.9999
        
    return best_pts, best_score

def _polish_lbfgs(pts, deadline=None):
    """
    Polish the solution using L-BFGS-B on the soft objective.
    """
    if deadline is None:
        deadline = time.time() + 5.0
        
    # Flatten points for scipy.optimize
    x0 = pts.flatten()
    
    def objective(x):
        p = x.reshape(N, D)
        # Re-center
        p = p - np.mean(p, axis=0)
        return _soft_ratio_obj(p, tau=50.0)
        
    def jac(x):
        # Numerical gradient is fine for N=14
        p = x.reshape(N, D)
        p = p - np.mean(p, axis=0)
        grad = np.zeros_like(x)
        eps = 1e-5
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            p_plus = x_plus.reshape(N, D)
            p_plus = p_plus - np.mean(p_plus, axis=0)
            f_plus = _soft_ratio_obj(p_plus, tau=50.0)
            
            x_minus = x.copy()
            x_minus[i] -= eps
            p_minus = x_minus.reshape(N, D)
            p_minus = p_minus - np.mean(p_minus, axis=0)
            f_minus = _soft_ratio_obj(p_minus, tau=50.0)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad

    # We don't strictly need bounds if we re-center, but let's keep them reasonable
    bounds = [(-10, 10)] * (N * D)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=jac, bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.success or (time.time() > deadline and res.fun < _true_score(pts)):
             # Check if the result is better
             new_pts = res.x.reshape(N, D)
             new_pts = new_pts - np.mean(new_pts, axis=0)
             new_score = _true_score(new_pts)
             if new_score < _true_score(pts):
                 return new_pts, new_score
    except Exception as e:
        pass
        
    return pts, _true_score(pts)

def construct():
    global _t0
    if '_t0' not in globals() or _t0 is None:
        _t0 = time.time()
        
    t0 = time.time()
    deadline = t0 + DEADLINE - 5.0 # 5s safety buffer
    
    best_pts = None
    best_score = np.inf
    
    # Multiple restarts with different initializations
    # 1. Icosahedral + Poles
    # 2. Fibonacci sphere (random subset)
    # 3. Random uniform sphere
    
    initializations = []
    
    # Init 1: Icosahedral
    initializations.append(_init_icosahedral())
    
    # Init 2: Fibonacci Sphere (14 points)
    i = np.arange(N)
    phi_angle = np.arccos(1 - 2 * (i + 0.5) / N)
    theta_angle = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi_angle) * np.cos(theta_angle)
    y = np.sin(phi_angle) * np.sin(theta_angle)
    z = np.cos(phi_angle)
    initializations.append(np.column_stack([x, y, z]))
    
    # Init 3: Random points on sphere
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        initializations.append(pts)
        
    # Run SA for each initialization, allocating time
    num_inits = len(initializations)
    time_per_init = (DEADLINE - 15.0) / num_inits # Reserve 15s for polishing
    
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 10.0: # Stop if close to deadline
            break
            
        # Run SA
        t_start = 1.0
        t_end = 1e-5
        alpha = 0.995 # Slower cooling to explore more
        
        # Estimate steps
        # temp goes from 1 to 1e-5. log(1e-5)/log(0.995) ~ 2300 steps.
        # Each step is fast.
        
        sa_pts, sa_score = _simulated_annealing(init_pts, t_start=t_start, t_end=t_end, 
                                                 alpha=alpha, deadline=time.time() + time_per_init)
        
        if sa_score < best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
            
    # Polish the best found solution
    if best_pts is not None:
        # Use remaining time for polishing
        time_left = deadline - time.time()
        if time_left > 2.0:
            polished_pts, polished_score = _polish_lbfgs(best_pts, deadline=time.time() + time_left - 1.0)
            if polished_score < best_score:
                best_score = polished_score
                best_pts = polished_pts
                
    # Fallback
    if best_pts is None:
        best_pts = initializations[0]
        
    return best_pts

# Global variable for timing
_t0 = None
