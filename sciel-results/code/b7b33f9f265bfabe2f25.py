import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def objective_smooth(x, tau=100.0):
    """
    Smoothed objective function using log-sum-exp for max and min.
    Minimizes log(max_sq_dist) - log(min_sq_dist).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max of squared distances
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min of squared distances
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    # Objective is log(max) - log(min)
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def fibonacci_sphere(n):
    """Generate n points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3. - np.sqrt(5.))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        radius = np.sqrt(1 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def generate_icosahedron():
    """Generate 12 icosahedron vertices on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def generate_initial_guesses(rng, n_guesses=10):
    """Generate diverse initial configurations."""
    guesses = []
    
    # 1. Fibonacci sphere
    fib = fibonacci_sphere(N)
    guesses.append(fib.flatten())
    
    # 2. Icosahedron + center + random point
    ico = generate_icosahedron()
    center = np.array([0.0, 0.0, 0.0])
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v)
    p14 = v * 1.0
    ico_plus = np.vstack([ico, center, p14])
    guesses.append(ico_plus.flatten())
    
    # 3. Icosahedron + 2 random points
    for _ in range(2):
        v1 = rng.normal(size=3)
        v1 = v1 / np.linalg.norm(v1)
        v2 = rng.normal(size=3)
        v2 = v2 / np.linalg.norm(v2)
        ico_plus2 = np.vstack([ico, v1 * 0.8, v2 * 0.8])
        guesses.append(ico_plus2.flatten())
    
    # 4. Random points on sphere
    for _ in range(n_guesses - 6):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        guesses.append(pts.flatten())
    
    # 5. Random points in cube
    for _ in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        guesses.append(pts.flatten())
    
    return guesses

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    def time_remaining():
        return timeout - (time.time() - start_time)
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial guesses
    initial_guesses = generate_initial_guesses(rng, n_guesses=15)
    
    # Multi-stage optimization with increasing tau
    tau_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    
    for guess_idx, x0 in enumerate(initial_guesses):
        if time_remaining() < 10:
            break
        
        x_current = x0.copy()
        
        for tau in tau_schedule:
            if time_remaining() < 5:
                break
            
            try:
                res = minimize(
                    objective_smooth,
                    x_current,
                    method='L-BFGS-B',
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    x_current = res.x
                    pts = x_current.reshape(N, D)
                    update_best(pts)
            except Exception:
                break
        
        # SLSQP polish at high tau if time permits
        if time_remaining() > 8 and x_current is not None:
            try:
                res_slsqp = minimize(
                    objective_smooth,
                    x_current,
                    method='SLSQP',
                    args=(10000.0,),
                    options={'maxiter': 100, 'ftol': 1e-12, 'eps': 1e-8}
                )
                if res_slsqp.x is not None:
                    pts_slsqp = res_slsqp.x.reshape(N, D)
                    update_best(pts_slsqp)
                    x_current = res_slsqp.x
            except Exception:
                pass
    
    # Final high-precision polish on the best points found
    if best_points is not None and time_remaining() > 3:
        try:
            res_final = minimize(
                objective_smooth,
                best_points.flatten(),
                method='L-BFGS-B',
                args=(10000.0,),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                update_best(pts_final)
        except Exception:
            pass
    
    # Another SLSQP polish if time permits
    if best_points is not None and time_remaining() > 2:
        try:
            res_final2 = minimize(
                objective_smooth,
                best_points.flatten(),
                method='SLSQP',
                args=(50000.0,),
                options={'maxiter': 50, 'ftol': 1e-14, 'eps': 1e-10}
            )
            if res_final2.x is not None:
                pts_final2 = res_final2.x.reshape(N, D)
                update_best(pts_final2)
        except Exception:
            pass

    if best_points is None:
        # Fallback: return Fibonacci sphere
        best_points = fibonacci_sphere(N)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
