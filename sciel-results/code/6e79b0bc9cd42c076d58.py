import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_soft_max_objective(points, beta=20.0):
    """
    Compute a smooth approximation of the covering radius using soft-max.
    For each point on the sphere (approximated by the points themselves + midpoints),
    we compute the distance to nearest point, then take a soft-max over all sample points.
    
    Returns: (objective_value, gradients)
    """
    # Sample points on the sphere: use the configuration points plus some additional samples
    # For efficiency, we'll use the points themselves and midpoints between close pairs
    # Actually, for the covering radius, we need to evaluate at ALL points on the sphere.
    # A good approximation: use a fine grid or the vertices of the convex hull.
    # But to avoid ConvexHull, let's use a different approach.
    
    # The covering radius is the max over all y on sphere of min_i angle(y, p_i)
    # This equals the max over all facets of the circumradius.
    # Alternative: sample many points on the sphere and compute min distance to each p_i.
    
    # Use Fibonacci sphere with many points as samples
    n_samples = 2000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each sample, compute distance to nearest point
    # dist[sample_idx] = min over i of arccos(dot(sample, p_i))
    # dot product: (n_samples, N)
    dots = samples @ points.T  # (n_samples, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    # Distance to each point
    dists = np.arccos(dots)  # (n_samples, N)
    
    # For each sample, the distance to nearest point
    min_dists = np.min(dists, axis=1)  # (n_samples,)
    
    # Soft-max over min_dists to get a smooth approximation of max
    # soft_max(x, beta) = (1/beta) * log(sum(exp(beta * x)))
    beta_safe = max(beta, 1e-6)
    max_val = np.max(min_dists)
    exp_terms = np.exp(beta_safe * (min_dists - max_val))
    log_sum = np.log(np.sum(exp_terms)) / beta_safe + max_val
    
    # Gradient computation
    # d(log_sum)/d(min_dists[j]) = exp(beta * (min_dists[j] - max_val)) / sum(exp(...))
    weights = exp_terms / np.sum(exp_terms)  # (n_samples,)
    
    # Now we need d(min_dists)/d(points)
    # min_dists[j] = min_i arccos(dot(samples[j], points[i]))
    # The argmin determines which point is closest
    argmins = np.argmin(dists, axis=1)  # (n_samples,)
    
    # Gradient of arccos(x) w.r.t. x is -1/sqrt(1-x^2)
    # d(arccos(dot(s, p_i)))/d(p_i) = -s / sqrt(1 - dot(s, p_i)^2)
    
    grads = np.zeros_like(points)
    for j in range(n_samples):
        i_idx = argmins[j]
        s = samples[j]
        p = points[i_idx]
        dot_val = np.dot(s, p)
        dot_val_clipped = np.clip(dot_val, -1.0 + 1e-10, 1.0 - 1e-10)
        denom = np.sqrt(1.0 - dot_val_clipped * dot_val_clipped)
        if denom < 1e-10:
            continue
        # d(arccos(dot))/d(p) = -s / sqrt(1 - dot^2)
        grads[i_idx] -= weights[j] * s / denom
    
    return log_sum, grads

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random points
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    
    # 3. Fibonacci + small noise
    np.random.seed(123)
    init_fib_noise = init_fib + np.random.randn(N, 3) * 0.1
    init_fib_noise /= np.linalg.norm(init_fib_noise, axis=1, keepdims=True)
    
    initializations = [init_fib, init_rand, init_fib_noise]
    
    # Add more randomized restarts
    max_restarts = 25
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Random perturbation of Fibonacci
            np.random.seed(restart * 100 + 7)
            points = init_fib + np.random.randn(N, 3) * 0.3
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Gradient descent with dynamic temperature
        learning_rate = 0.02
        no_improve = 0
        current_obj = np.pi
        beta = 10.0  # Start with moderate temperature
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
                
            obj, grads = compute_soft_max_objective(points, beta)
            
            # Project gradients onto tangent space (remove radial component)
            for i in range(N):
                radial = np.dot(grads[i], points[i])
                grads[i] -= radial * points[i]
            
            # Update
            points = points + learning_rate * grads
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
            # Track best
            if obj < best_radius:
                best_radius = obj
                best_points = points.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 100:
                    # Decrease learning rate and increase temperature (explore more)
                    learning_rate *= 0.8
                    beta *= 1.5
                    if learning_rate < 1e-5:
                        break
            
            # Dynamic temperature schedule: increase beta over time to sharpen
            beta = 10.0 + step * 0.05
            
            # Occasionally add small random perturbation to escape local minima
            if step % 100 == 0 and step > 0:
                np.random.seed(step + restart)
                perturb = np.random.randn(N, 3) * 0.01
                points = points + perturb
                points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    if best_points is None:
        best_points = init_fib
        
    return best_points
