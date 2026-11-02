import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigh
from scipy.special import logsumexp

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
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
    """
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_smooth(x, tau=100.0):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    Uses Log-Sum-Exp for soft max and soft min.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max using logsumexp for numerical stability
    log_sum_exp_max = logsumexp(tau * sq_dists) / tau
    
    # Soft min: -logsumexp(-tau * sq_dists) / tau
    log_sum_exp_min = -logsumexp(-tau * sq_dists) / tau
    
    # Minimize log(soft_max) - log(soft_min)
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def objective_epigraph(x, t, tau=100.0):
    """
    Epigraph formulation with penalty method.
    """
    pts = x[:N*D].reshape(N, D)
    t = x[N*D]
    
    if t < 1e-6:
        t = 1e-6
    
    pts = normalize_points(pts)
    sq_dists = get_pairwise_sq_dists(pts)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Penalty for d^2 < 1/t
    viol_low = np.maximum(1.0/t - sq_dists, 0)**2
    # Penalty for d^2 > t
    viol_high = np.maximum(sq_dists - t, 0)**2
    
    lam = 100.0
    return t + lam * (np.sum(viol_low) + np.sum(viol_high))

def objective_direct(x):
    """
    Direct non-smooth objective for SLSQP or other methods that can handle it.
    Minimizes log(max) - log(min) or just ratio.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))

def objective_gram(g_vec, tau=100.0):
    """
    Objective for Gram matrix optimization.
    g_vec contains the upper triangle of the Gram matrix G (size N x N).
    We reconstruct G, check PSD, extract points, and compute the smooth ratio.
    """
    # Reconstruct Gram matrix from upper triangle
    G = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i, N):
            G[i, j] = g_vec[idx]
            G[j, i] = g_vec[idx]
            idx += 1
    
    # Ensure PSD by clipping eigenvalues
    eigvals, eigvecs = eigh(G)
    eigvals_clipped = np.maximum(eigvals, 0)
    
    # Extract points from eigendecomposition
    # Points are rows of eigvecs * sqrt(eigvals_clipped)
    points = eigvecs.T * np.sqrt(eigvals_clipped)
    
    # Keep only first D dimensions (or use all if N > D)
    if points.shape[1] > D:
        points_d = points[:, :D]
    else:
        points_d = points
    
    points_d = normalize_points(points_d)
    
    sq_dists = get_pairwise_sq_dists(points_d)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max and min using logsumexp
    log_sum_exp_max = logsumexp(tau * sq_dists) / tau
    log_sum_exp_min = -logsumexp(-tau * sq_dists) / tau
    
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def points_to_gram_vec(points):
    """Convert points to Gram matrix upper triangle vector."""
    points = normalize_points(points)
    G = points.T @ points
    # Extract upper triangle including diagonal
    iu = np.triu_indices(N)
    return G[iu]

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

def rotation_matrix_zxz(angles):
    """Create rotation matrix from Z-X-Z Euler angles."""
    az, ax, az2 = angles
    cz, sx = np.cos(az), np.sin(az)
    cx, sx_ = np.cos(ax), np.sin(ax)
    cz2, sx2 = np.cos(az2), np.sin(az2)
    R_z = np.array([[cz, -sx, 0], [sx, cz, 0], [0, 0, 1]])
    R_x = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    R_z2 = np.array([[cz2, -sx2, 0], [sx2, cz2, 0], [0, 0, 1]])
    return R_z2 @ R_x @ R_z

def tetrahedron_points():
    """Return vertices of a regular tetrahedron centered at origin."""
    pts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def two_tetra_points(params):
    """
    Generate 14 points from two concentric regular tetrahedra.
    params: [r1, r2, az1, ax1, az2_1, az2, ax2, az2_2]
    r1, r2: radii of the two tetrahedra
    az1, ax1, az2_1: Z-X-Z Euler angles for first tetrahedron
    az2, ax2, az2_2: Z-X-Z Euler angles for second tetrahedron
    """
    r1, r2 = params[0], params[1]
    angles1 = params[2:5]
    angles2 = params[5:8]
    
    tetra = tetrahedron_points()
    R1 = rotation_matrix_zxz(angles1)
    R2 = rotation_matrix_zxz(angles2)
    
    pts1 = r1 * (tetra @ R1.T)
    pts2 = r2 * (tetra @ R2.T)
    
    return np.vstack([pts1, pts2])

def two_tetra_objective(params):
    """Objective for two-tetrahedron parameterization."""
    pts = two_tetra_points(params)
    pts = normalize_points(pts)
    sq_dists = get_pairwise_sq_dists(pts)
    sq_dists = np.maximum(sq_dists, 1e-12)
    return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))

def two_tetra_objective_smooth(params, tau=100.0):
    """Smooth objective for two-tetrahedron parameterization."""
    pts = two_tetra_points(params)
    pts = normalize_points(pts)
    sq_dists = get_pairwise_sq_dists(pts)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    log_sum_exp_max = logsumexp(tau * sq_dists) / tau
    log_sum_exp_min = -logsumexp(-tau * sq_dists) / tau
    
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

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
    
    # 2. Random points on sphere (more diverse)
    for seed in range(20):
        r = np.random.default_rng(seed * 1000 + 42)
        pts = r.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Symmetric candidate: Ico + Poles
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    # 4. Tetrahedron-based initialization
    # Regular tetrahedron vertices
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    # Edge midpoints
    edge_midpoints = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_midpoints.append(mid)
    
    # Face centers
    face_centers = []
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                center = (tetra[i] + tetra[j] + tetra[k]) / 3
                center = center / np.linalg.norm(center)
                face_centers.append(center)
    
    # Combine: 4 tetra + 6 edge mids + 4 face centers = 14
    tetra_full = np.vstack([tetra, np.array(edge_midpoints), np.array(face_centers[:4])])
    candidates.append(tetra_full.flatten())
    
    # 5. More random symmetric candidates
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())

    # Optimization Loop
    n_restarts = len(candidates)
    
    # Phase 1: Smooth Log-Ratio Annealing with better tau schedule
    taus = [10.0, 50.0, 200.0, 1000.0]
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 40:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Phase 1: Standard smooth optimization with annealing
            for tau in taus:
                if time.time() - start_time > timeout - 40:
                    break
                
                res = minimize(
                    objective_smooth, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                if res.x is not None:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
            
            # Phase 2: Direct non-smooth polish with SLSQP
            if time.time() - start_time < timeout - 30:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                
                res = minimize(
                    objective_direct, 
                    pts_current.flatten(), 
                    method='SLSQP',
                    options={'maxiter': 150, 'ftol': 1e-14}
                )
                
                if res.x is not None:
                    pts_direct = res.x.reshape(N, D)
                    update_best(pts_direct)

        except Exception:
            pass
    
    # Phase 2: Two-tetrahedron parameterization search
    if time.time() - start_time < timeout - 60:
        try:
            # Differential evolution for global search in reduced space
            bounds_tetra = [
                (0.1, 2.0),  # r1
                (0.1, 2.0),  # r2
                (-np.pi, np.pi),  # az1
                (-np.pi, np.pi),  # ax1
                (-np.pi, np.pi),  # az2_1
                (-np.pi, np.pi),  # az2
                (-np.pi, np.pi),  # ax2
                (-np.pi, np.pi),  # az2_2
            ]
            
            de_result = differential_evolution(
                two_tetra_objective_smooth,
                bounds_tetra,
                args=(100.0,),
                seed=42,
                maxiter=50,
                tol=1e-8,
                popsize=15,
                polish=True,
                updating='immediate'
            )
            
            if de_result.success or de_result.fun < best_score:
                best_params = de_result.x
                pts_tetra = two_tetra_points(best_params)
                update_best(pts_tetra)
                
                # Local refinement with Nelder-Mead
                if time.time() - start_time < timeout - 30:
                    nm_result = minimize(
                        two_tetra_objective,
                        best_params,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12}
                    )
                    if nm_result.x is not None:
                        pts_nm = two_tetra_points(nm_result.x)
                        update_best(pts_nm)
                        
                        # Further polish with high-precision smooth objective
                        if time.time() - start_time < timeout - 20:
                            res_smooth = minimize(
                                two_tetra_objective_smooth,
                                nm_result.x,
                                args=(1000.0,),
                                method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 1e-14, 'fatol': 1e-14}
                            )
                            if res_smooth.x is not None:
                                pts_smooth = two_tetra_points(res_smooth.x)
                                update_best(pts_smooth)
        except Exception:
            pass
    
    # Final high-precision polish pass on best point
    if best_points is not None and time.time() - start_time < timeout - 20:
        try:
            x_polish = best_points.flatten()
            # Use very high tau for precise max/min
            res = minimize(
                objective_smooth, 
                x_polish, 
                method='L-BFGS-B', 
                args=(5000.0,),
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if res.x is not None:
                pts_polish = res.x.reshape(N, D)
                update_best(pts_polish)
                
            # SLSQP pass on the result
            if time.time() - start_time < timeout - 10:
                res = minimize(
                    objective_direct, 
                    best_points.flatten(), 
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-16}
                )
                if res.x is not None:
                    pts_polish2 = res.x.reshape(N, D)
                    update_best(pts_polish2)
        except Exception:
            pass

    if best_points is None:
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
    print(f"Time used: {time.time() - time.time()}")
