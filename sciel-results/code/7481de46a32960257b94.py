import numpy as np
import time
from scipy.spatial.distance import cdist

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _score(points):
    """Compute the raw score: largest common radius."""
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

def _soft_min_score(points, beta=100.0):
    """
    Smooth surrogate for the score.
    Score = min(min_pair, boundary_min)
    We approximate min(x, y) using -log(exp(beta*x) + exp(beta*y))/beta.
    However, we want to maximize the score.
    Let's maximize -soft_min(penalty) or directly approximate the min.
    
    Actually, it's easier to minimize the violation.
    Let r be a variable? No, PSO optimizes positions.
    
    Objective: Maximize r such that all constraints hold.
    Equivalent to: Maximize min( min_pair, boundary_min ).
    
    Let's use the soft-min of the actual values directly.
    f(points) = soft_min(min_pair, boundary_min)
    To maximize this, we can minimize -f(points).
    Or, we can just evaluate the raw score occasionally and use soft-min for the swarm velocity updates?
    
    A better surrogate for maximization:
    Let constraints be g_i(points) >= 0.
    Here, we have implicit constraints.
    
    Let's stick to a direct smooth approximation of the objective function to be maximized.
    min(a, b) approx -1/beta * log( exp(-beta*a) + exp(-beta*b) )  <-- This is soft-min for minimization.
    For maximization of min(a,b), we want a function that is close to min(a,b) but smooth.
    soft_max(a, b) = 1/beta * log( exp(beta*a) + exp(beta*b) ) is soft-max.
    We want soft-min.
    soft_min(a, b) = - soft_max(-a, -b) = -1/beta * log( exp(-beta*a) + exp(-beta*b) ).
    
    So, Objective = -1/beta * log( exp(-beta * min_pair) + exp(-beta * boundary_min) ).
    As beta -> inf, this converges to min(min_pair, boundary_min).
    
    We will maximize this objective.
    """
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_min = 1.0 - np.max(norms)
    
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    # Handle cases where values are very negative to avoid overflow in exp
    # exp(-beta * x). If x is large positive, exp is small. If x is large negative, exp is large.
    # We expect x to be around 0.1.
    a = -beta * min_pair
    b = -beta * boundary_min
    
    # log-sum-exp trick for stability
    m = np.maximum(a, b)
    if m > 0:
        # If m is very large positive, exp(-m) is tiny.
        # log(exp(a) + exp(b)) = m + log(exp(a-m) + exp(b-m))
        log_sum = m + np.log(np.exp(a - m) + np.exp(b - m))
    else:
        log_sum = np.log(np.exp(a) + np.exp(b))
        
    return -log_sum / beta

def _local_optimize(points, r_target, max_iters=500):
    """
    Fast local optimization using force-directed relaxation.
    Used for polishing particle positions.
    """
    points = points.copy()
    n = len(points)
    lr = 0.02
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        d = cdist(points, points)
        np.fill_diagonal(d, np.inf)
        
        mask = d < 2 * r_target
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        
        d_safe = np.where(d > 1e-12, d, 1.0)
        unit_diff = diff / d_safe[:, :, np.newaxis]
        overlap = np.where(mask, 2 * r_target - d, 0.0)
        force_contrib = overlap[:, :, np.newaxis] * unit_diff
        forces = np.sum(force_contrib, axis=1)
        
        # Boundary forces
        norms = np.hypot(points[:, 0], points[:, 1])
        b_mask = norms > (1.0 - r_target)
        b_overlap = np.where(b_mask, norms - (1.0 - r_target), 0.0)
        norms_safe = np.where(norms > 1e-12, norms, 1.0)
        b_forces = -b_overlap[:, np.newaxis] * points / norms_safe[:, np.newaxis]
        forces += b_forces
        
        points += lr * forces
        lr *= 0.99
        
        # Simple convergence check
        if np.max(np.abs(forces)) < 1e-7:
            break
            
    return points

def _random_initial(seed):
    np.random.seed(seed)
    # Generate points in a disk, slightly inset
    r = np.sqrt(np.random.uniform(0, 0.9))
    theta = np.random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _hexagonal_initial(spacing):
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
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_score = 0.0
    best_points = None
    
    # Initialize swarm
    swarm_size = 10
    # Mix of hexagonal and random starts to ensure diversity
    inits = []
    hex_pts = _hexagonal_initial(0.22)
    if hex_pts is not None:
        inits.append(hex_pts)
    for i in range(swarm_size - 1):
        inits.append(_random_initial(i))
        
    # Ensure we have enough inits
    while len(inits) < swarm_size:
        inits.append(_random_initial(len(inits) + 100))
        
    # Particle positions
    X = np.array(inits[:swarm_size])  # (swarm_size, N, 2)
    
    # Velocities
    V = np.zeros_like(X)
    
    # Personal bests
    pbest = X.copy()
    pbest_scores = np.array([_soft_min_score(p, beta=50) for p in X])
    
    # Global best
    gbest_idx = np.argmax(pbest_scores)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]
    
    # PSO Parameters
    w = 0.7  # Inertia
    c1 = 1.5  # Cognitive
    c2 = 1.5  # Social
    max_iters = 400  # Total iterations
    
    # We will periodically check the raw score of gbest and update if better
    # Also, we will use a local polish on gbest every few iterations
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Update velocities
        # V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        r1 = np.random.rand(swarm_size, 1, 1)
        r2 = np.random.rand(swarm_size, 1, 1)
        
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        
        # Clamp velocity to prevent explosion
        # Max step size per dimension
        max_v = 0.05
        V = np.clip(V, -max_v, max_v)
        
        # Update positions
        X += V
        
        # Evaluate new positions
        # Use soft-min for speed and smoothness
        new_scores = np.array([_soft_min_score(x, beta=50) for x in X])
        
        # Update personal bests
        improved = new_scores > pbest_scores
        pbest[improved] = X[improved]
        pbest_scores[improved] = new_scores[improved]
        
        # Update global best
        if np.max(new_scores) > gbest_score:
            gbest_idx_local = np.argmax(new_scores)
            gbest = X[gbest_idx_local].copy()
            gbest_score = new_scores[gbest_idx_local]
            
            # Optional: Local polish on gbest to get a better raw score
            # We estimate the radius from the soft-min score
            # r_est is roughly gbest_score
            r_est = max(0.05, gbest_score)
            
            # Run a quick local optimization to see if we can improve the raw score
            polished = _local_optimize(gbest, r_est, max_iters=100)
            raw_score = _score(polished)
            
            if raw_score > best_score:
                best_score = raw_score
                best_points = polished.copy()
                # Update gbest to the polished version if it's better in raw terms?
                # The soft-min score of polished might be higher too.
                polished_soft = _soft_min_score(polished, beta=50)
                if polished_soft > gbest_score:
                    gbest = polished
                    gbest_score = polished_soft
                else:
                    # Keep the PSO gbest for the swarm dynamics, but record the polished one as best found
                    pass
            
        # Every 20 iterations, try to polish the current gbest with a slightly higher target
        if it % 20 == 19:
            # Estimate current max radius
            # The soft-min score is an approximation of the min radius.
            # Let's try to push a bit harder.
            r_target = best_score * 1.01 if best_score > 0.05 else 0.12
            polished = _local_optimize(gbest, r_target, max_iters=200)
            raw_score = _score(polished)
            if raw_score > best_score:
                best_score = raw_score
                best_points = polished.copy()
                # Update gbest to this new best to guide the swarm
                gbest = polished.copy()
                gbest_score = _soft_min_score(polished, beta=50)

    # Final polish
    if best_points is not None:
        # Use the best found score as target, or slightly higher
        target_r = best_score
        # Try a few different targets around the best score
        for r_try in [target_r, target_r * 1.005, target_r * 1.01]:
            if time.time() - START_TIME > TIME_LIMIT - 2:
                break
            polished = _local_optimize(best_points, r_try, max_iters=1000)
            s = _score(polished)
            if s > best_score:
                best_score = s
                best_points = polished.copy()
    else:
        # Fallback
        best_points = _hexagonal_initial(0.22)
        if best_points is None:
            best_points = np.zeros((N, 2))

    # Ensure validity
    norms = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.999
        
    return best_points

if __name__ == "__main__":
    construct()
