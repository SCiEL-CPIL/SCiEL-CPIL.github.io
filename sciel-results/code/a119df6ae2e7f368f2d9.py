import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    # Using broadcasting for efficiency
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    This removes translational and scale invariance.
    """
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_direct(x):
    """
    Direct non-smooth objective: log(max_sq_dist) - log(min_sq_dist).
    """
    points = x[:N*D].reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))

def objective_smooth(x, tau=100.0):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    """
    points = x[:N*D].reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def objective_epigraph(x, mu=100.0, tau=50.0, beta=10.0):
    """
    Epigraph formulation with smooth penalty.
    x layout: [points (N*D), log_t (1)]
    Minimizes log_t + mu * SoftReLU(log(SoftMax) - log(SoftMin) - log_t)
    """
    points = x[:N*D].reshape(N, D)
    log_t = x[N*D]
    
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    log_ratio = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    violation = log_ratio - log_t
    
    # Smooth ReLU: log(1 + exp(beta * violation)) / beta
    # Using logsumexp for stability: log(0 + exp(beta*violation)) / beta
    # log(1 + exp(z)) = logsumexp([0, z])
    soft_relu = logsumexp([0, beta * violation]) / beta
    
    return log_t + mu * soft_relu

def fibonacci_sphere(n, seed=42):
    """Generate n points on a sphere using Fibonacci lattice."""
    rng = np.random.default_rng(seed)
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((n, D))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i + rng.uniform(0, 2*np.pi) * 0.01
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points

def generate_icosahedron_poles():
    """Generate Icosahedron vertices + 2 poles."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    return pts_sym

def generate_tetrahedron_based():
    """Generate points based on tetrahedron symmetry."""
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    edge_midpoints = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_midpoints.append(mid)
    
    face_centers = []
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                center = (tetra[i] + tetra[j] + tetra[k]) / 3
                center = center / np.linalg.norm(center)
                face_centers.append(center)
    
    # 4 tetra + 6 edge mids + 4 face centers = 14
    tetra_full = np.vstack([tetra, np.array(edge_midpoints), np.array(face_centers[:4])])
    return tetra_full

def generate_cuboctahedron_based():
    """Generate points based on Cuboctahedron vertices + 2 random."""
    rng = np.random.default_rng(123)
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    cubo = perms / np.linalg.norm(perms, axis=1, keepdims=True)
    p1 = rng.normal(size=D)
    p2 = rng.normal(size=D)
    p1 /= np.linalg.norm(p1)
    p2 /= np.linalg.norm(p2)
    pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
    return pts

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin for 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Fibonacci sphere
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 2. Random points on sphere
    for _ in range(15):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + Poles
    ico_pts = generate_icosahedron_poles()
    candidates.append(ico_pts.flatten())
    
    # 4. Tetrahedron-based
    tetra_pts = generate_tetrahedron_based()
    candidates.append(tetra_pts.flatten())
    
    # 5. Cuboctahedron-based
    cubo_pts = generate_cuboctahedron_based()
    candidates.append(cubo_pts.flatten())
    
    # 6. Fibonacci sphere perturbations
    for _ in range(5):
        perturbed = fib_pts + rng.normal(scale=0.1, size=(N, D))
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        candidates.append(perturbed.flatten())

    # Optimization Loop
    # Strategy:
    # 1. Nelder-Mead on direct objective (robust, non-smooth)
    # 2. L-BFGS-B on Epigraph objective (smooth, constrained ratio)
    # 3. L-BFGS-B on Smooth Log-Ratio (final polish)
    
    for i in range(len(candidates)):
        if time.time() - start_time > timeout - 10:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0_flat = x0.flatten()
        
        current_best_x = x0_flat
        current_best_val = objective_direct(x0_flat)
        
        # Stage 1: Nelder-Mead on direct objective
        # This helps find a good basin without getting stuck in gradient noise
        try:
            res_nm = minimize(
                objective_direct, 
                current_best_x, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_nm.x is not None:
                val_nm = objective_direct(res_nm.x)
                if val_nm < current_best_val:
                    current_best_val = val_nm
                    current_best_x = res_nm.x
        except Exception:
            pass
        
        # Prepare for Epigraph optimization
        # x layout: [points, log_t]
        # Initialize log_t to current log ratio
        pts_nm = current_best_x.reshape(N, D)
        sq_dists_nm = get_pairwise_sq_dists(pts_nm)
        if np.min(sq_dists_nm) > 1e-12:
            log_t_init = np.log(np.max(sq_dists_nm) / np.min(sq_dists_nm))
        else:
            log_t_init = 0.0
            
        x_epigraph = np.append(current_best_x, log_t_init)
        
        # Stage 2: L-BFGS-B with Epigraph formulation
        # We anneal mu (penalty) and tau (softness)
        epigraph_params = [
            (10.0, 20.0, 5.0),   # mu, tau, beta
            (50.0, 50.0, 10.0),
            (200.0, 100.0, 20.0),
            (1000.0, 200.0, 50.0)
        ]
        
        for mu, tau, beta in epigraph_params:
            if time.time() - start_time > timeout - 5:
                break
                
            try:
                res_lbfgs = minimize(
                    lambda x: objective_epigraph(x, mu=mu, tau=tau, beta=beta),
                    x_epigraph,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_lbfgs.x is not None:
                    # Update x_epigraph with the new points and log_t
                    x_epigraph = res_lbfgs.x
                    # Check the direct score of the points part
                    pts_check = res_lbfgs.x[:N*D].reshape(N, D)
                    val_check = objective_direct(pts_check.flatten())
                    if val_check < current_best_val:
                        current_best_val = val_check
                        current_best_x = pts_check.flatten()
            except Exception:
                pass
        
        # Stage 3: Final Polish on Smooth Log-Ratio
        if time.time() - start_time < timeout - 2:
            try:
                res_polish = minimize(
                    lambda x: objective_smooth(x, tau=500.0),
                    current_best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res_polish.x is not None:
                    val_polish = objective_direct(res_polish.x)
                    if val_polish < current_best_val:
                        current_best_val = val_polish
                        current_best_x = res_polish.x
            except Exception:
                pass

        # Update global best
        if current_best_x is not None:
            pts_candidate = current_best_x.reshape(N, D)
            update_best(pts_candidate)
            
        # Early stopping if we hit a very good score
        if best_score < 1.02:
            break

    if best_points is None:
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
