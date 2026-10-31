import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0
H = np.sqrt(3) / 2

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def point_in_triangle(pt, v0, v1, v2):
    """Check if point is inside triangle using barycentric coordinates."""
    x, y = pt
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2
    
    # Barycentric coordinates
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-15:
        return False
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1.0 - a - b
    
    return a >= -1e-10 and b >= -1e-10 and c >= -1e-10

def project_to_triangle(pt):
    """Project point onto the triangle if it's outside."""
    if point_in_triangle(pt, V0, V1, V2):
        return pt
    
    # Find closest point on triangle boundary
    best_pt = pt
    best_dist = np.inf
    
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for p1, p2 in edges:
        v = p2 - p1
        t = np.dot(pt - p1, v) / np.dot(v, v)
        t = np.clip(t, 0, 1)
        proj = p1 + t * v
        dist = np.linalg.norm(pt - proj)
        if dist < best_dist:
            best_dist = dist
            best_pt = proj
    
    for v in [V0, V1, V2]:
        dist = np.linalg.norm(pt - v)
        if dist < best_dist:
            best_dist = dist
            best_pt = v
    
    return best_pt

def clamp_points_cart(pts):
    """Clamp all points to be inside the triangle."""
    return np.array([project_to_triangle(p) for p in pts])

def compute_min_area_cart(pts):
    """Compute minimum triangle area for points in Cartesian coordinates."""
    pi = pts[tri_idx[:, 0]]
    pj = pts[tri_idx[:, 1]]
    pk = pts[tri_idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def smooth_min_area(pts, k=50.0):
    """
    Compute a smooth approximation of the minimum area using LogSumExp.
    min(A_i) approx -1/k * log(sum(exp(-k * A_i)))
    We want to maximize min area, so we minimize -smooth_min_area.
    """
    pi = pts[tri_idx[:, 0]]
    pj = pts[tri_idx[:, 1]]
    pk = pts[tri_idx[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # To avoid numerical issues with very small areas (which lead to large exp),
    # we can shift or use stable logsumexp.
    # scipy.special.logsumexp is stable, but let's implement a simple stable version.
    # log(sum(exp(-k * A))) = -k * max(A) + log(sum(exp(-k * (A - max(A)))))
    max_a = np.max(areas)
    if max_a == 0:
        return 0.0
        
    shifted = -k * (areas - max_a)
    # Clip shifted to avoid underflow issues in exp if k is large, though logsumexp handles it
    # exp(shifted) will be <= 1.
    sum_exp = np.sum(np.exp(shifted))
    
    log_sum = np.log(sum_exp)
    
    # log(sum(exp(-k*A))) = log(sum(exp(-k*(A - max_A) * exp(-k*max_A))) )
    # = log(sum_exp) - k * max_A
    log_sum_exp = log_sum - k * max_a
    
    smooth_min = -log_sum_exp / k
    
    return smooth_min

def objective_smooth(x, k=50.0):
    """
    Objective for smooth optimization.
    x is flat array of 2*N floats.
    Returns negative of smooth min area.
    """
    pts = x.reshape(N, 2)
    pts = clamp_points_cart(pts)
    val = smooth_min_area(pts, k=k)
    return -val

def pso_optimize(time_limit=60.0, seed=42):
    """
    Particle Swarm Optimization.
    """
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    # Swarm parameters
    n_particles = 50
    w = 0.7  # Inertia
    c1 = 1.5  # Cognitive
    c2 = 1.5  # Social
    max_v = 0.1  # Max velocity step
    
    # Initialize particles
    # Start with some random points inside triangle
    # Simple rejection sampling or barycentric random
    particles = np.zeros((n_particles, N, 2))
    for i in range(n_particles):
        for j in range(N):
            # Random point in triangle
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            pt = V0 + u * (V1 - V0) + v * (V2 - V0)
            particles[i, j] = pt
            
    # Evaluate initial fitness
    pbest = particles.copy()
    pbest_fit = np.zeros(n_particles)
    gbest = particles[0].copy()
    gbest_fit = -np.inf
    
    for i in range(n_particles):
        val = compute_min_area_cart(particles[i])
        pbest_fit[i] = val
        if val > gbest_fit:
            gbest_fit = val
            gbest = particles[i].copy()
            
    velocities = np.zeros_like(particles)
    
    while time.time() - start_time < time_limit:
        # Update velocities
        for i in range(n_particles):
            r1 = rng.random(N, 2)
            r2 = rng.random(N, 2)
            
            velocities[i] = (w * velocities[i] + 
                             c1 * r1 * (pbest[i] - particles[i]) + 
                             c2 * r2 * (gbest - particles[i]))
            
            # Clip velocity
            velocities[i] = np.clip(velocities[i], -max_v, max_v)
            
            # Update positions
            particles[i] += velocities[i]
            particles[i] = clamp_points_cart(particles[i])
            
            # Evaluate fitness
            val = compute_min_area_cart(particles[i])
            
            if val > pbest_fit[i]:
                pbest_fit[i] = val
                pbest[i] = particles[i].copy()
                
            if val > gbest_fit:
                gbest_fit = val
                gbest = particles[i].copy()
                
    return gbest, gbest_fit

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Phase 1: PSO for global exploration
    # Use a significant portion of time for PSO
    pso_time = 60.0
    remaining_after_pso = deadline - (start_time + pso_time)
    
    # If we have very little time, skip PSO and go straight to polish? 
    # But PSO is the main driver here.
    if remaining_after_pso < 10:
         pso_time = max(5.0, deadline - start_time - 10)
         
    try:
        best_pts, best_val = pso_optimize(time_limit=pso_time, seed=42)
    except Exception as e:
        # Fallback to random if PSO fails
        rng = np.random.default_rng(42)
        best_pts = np.zeros((N, 2))
        for j in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            best_pts[j] = V0 + u * (V1 - V0) + v * (V2 - V0)
        best_val = compute_min_area_cart(best_pts)
        
    # Phase 2: Gradient-Based Polish using Smooth Objective
    # Use L-BFGS-B on the smooth approximation
    # We need to be careful with the smooth parameter k.
    # Start with a moderate k and increase it? Or just use a fixed high k.
    # A high k makes the smooth function closer to the true min, but steeper.
    # Let's use k=100 for polish.
    
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        k_polish = 100.0
        
        # Define wrapper for minimize
        def obj_wrap(x):
            return objective_smooth(x, k=k_polish)
            
        # Initial guess is the best from PSO
        x0 = best_pts.flatten()
        
        try:
            # L-BFGS-B is good for box constraints, but we have triangle constraints.
            # We handle constraints via projection inside the objective/gradient?
            # No, minimize doesn't support custom projection easily in L-BFGS-B.
            # However, if we project inside the objective, the gradient is not accurate for the projected space.
            # Alternative: Use SLSQP with constraints? Too slow for 22 vars?
            # Better: Use the smooth objective which is defined on R^2N, and rely on the fact
            # that the optimum is likely interior or on boundary.
            # If we just minimize the smooth function on R^2N without constraints,
            # points might go outside.
            # We must ensure points stay inside.
            
            # Strategy: Use a penalty method or just project after each step?
            # scipy.optimize.minimize doesn't allow hooking projection easily.
            
            # Let's try a different polish: 
            # Use the smooth objective but add a large penalty if points are outside.
            # Or, simpler: Use the smooth objective and assume the PSO result is already inside,
            # and the gradient steps are small enough that projection is a minor correction.
            # Actually, let's just do a few steps of gradient descent manually with projection.
            
            # Manual Gradient Descent with Projection
            lr = 0.01
            current_pts = best_pts.copy()
            current_val = best_val
            
            # Compute gradient of smooth objective
            # Since smooth_min_area is differentiable, we can use finite differences or analytic gradient.
            # Analytic gradient is complex. Let's use finite differences for the gradient step.
            # Or just use `minimize` with `method='L-BFGS-B'` and accept that it might go outside,
            # then project the result. If the function is smooth and the optimum is inside,
            # it should work reasonably well if we start close.
            
            # Let's try `minimize` with L-BFGS-B.
            # To handle the triangle constraint, we can use a transformation?
            # No, let's stick to the previous plan: Project after minimize.
            # Risk: Minimize might find a local min outside the triangle that projects to a worse point.
            # Mitigation: The smooth function is defined for all R^2. The "minimum area" is 0 if collinear.
            # Points far outside might have large areas? No, if points are far apart, area is large.
            # We want to MAXIMIZE min area. So we MINIMIZE -smooth_min_area.
            # If points go far outside, they might form large triangles, so -smooth_min_area becomes very negative.
            # The optimizer will try to minimize this, i.e., make it more negative, i.e., increase area.
            # This might push points to infinity!
            
            # Therefore, we MUST constrain the domain or add a penalty for being outside.
            
            # Let's add a penalty to the objective for being outside the triangle.
            def obj_penalized(x, k=k_polish):
                pts = x.reshape(N, 2)
                # Penalty for being outside
                pen = 0.0
                for p in pts:
                    if not point_in_triangle(p, V0, V1, V2):
                        # Distance to triangle
                        proj = project_to_triangle(p)
                        d = np.linalg.norm(p - proj)
                        pen += 100.0 * d**2
                val = smooth_min_area(pts, k=k)
                return -val + pen
                
            # Use L-BFGS-B
            res = minimize(
                obj_penalized,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if res.success:
                cand_pts = clamp_points_cart(res.x.reshape(N, 2))
                cand_val = compute_min_area_cart(cand_pts)
                if cand_val > best_val:
                    best_pts = cand_pts
                    best_val = cand_val
                    
        except Exception:
            pass
            
    # Phase 3: Final local search with exact objective (Simulated Annealing or Random Hill Climbing)
    # Since we have time left, do a quick random hill climb with exact objective.
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = np.random.default_rng(123)
        current_pts = best_pts.copy()
        current_val = best_val
        T = 0.01
        cooling = 0.99
        
        while time.time() < deadline - 1 and T > 1e-6:
            p_idx = np.random.randint(N)
            delta = np.random.normal(0, T, 2)
            new_pts = current_pts.copy()
            new_pts[p_idx] += delta
            new_pts = clamp_points_cart(new_pts)
            
            new_val = compute_min_area_cart(new_pts)
            
            if new_val > current_val or np.random.random() < np.exp((new_val - current_val) / T):
                current_pts = new_pts
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best_pts = current_pts.copy()
                    
            T *= cooling

    return best_pts
