import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Returns flat array of squared pairwise distances (upper triangle)."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Compute (max_dist / min_dist)^2 = max_sq / min_sq."""
    sq_dists = get_pairwise_sq_dists(points)
    mn = np.min(sq_dists)
    mx = np.max(sq_dists)
    if mn < 1e-14:
        return np.inf
    return mx / mn

def objective_smooth(x, tau=20.0, mu=0.02):
    """
    Smooth objective: log(softmax) - log(softmin) + barrier.
    Uses temperature-scaled log-sum-exp for better conditioning.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-10)
    
    # Softmax with temperature: (1/tau) * log(sum(exp(tau * sq)))
    m = np.max(sq_dists)
    log_softmax = (m + np.log(np.sum(np.exp(tau * (sq_dists - m)))) / tau)
    
    # Softmin with temperature: -(1/tau) * log(sum(exp(-tau * sq)))
    m_neg = np.min(sq_dists)
    log_softmin = (m_neg - np.log(np.sum(np.exp(-tau * (sq_dists - m_neg)))) / tau)
    
    if log_softmin <= 1e-10:
        return 1e10
    
    obj = np.log(log_softmax) - np.log(log_softmin)
    
    # Barrier to prevent coincident points
    min_sq = np.min(sq_dists)
    barrier = -mu * np.log(min_sq)
    
    return obj + barrier

def objective_precise(x, tau=100.0):
    """
    High-precision objective with larger tau for sharper min/max approximation.
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Softmax with large tau
    m = np.max(sq_dists)
    log_softmax = (m + np.log(np.sum(np.exp(tau * (sq_dists - m)))) / tau)
    
    # Softmin with large tau
    m_neg = np.min(sq_dists)
    log_softmin = (m_neg - np.log(np.sum(np.exp(-tau * (sq_dists - m_neg)))) / tau)
    
    if log_softmin <= 1e-12:
        return 1e10
    
    return np.log(log_softmax) - np.log(log_softmin)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        s = raw_score(pts)
        if s < best_score:
            best_score = s
            best_points = pts.copy()
    
    rng = np.random.default_rng(12345)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Improved grid-based start: body-centered cubic-like arrangement
    # Use a 3x3x3 grid, select 14 points that maximize minimum distance
    grid_pts = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                grid_pts.append([x, y, z])
    # Select 14 points: all corners + face centers (6) + 2 body centers offset
    corners = [[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]
    face_centers = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    # Pick 2 more from remaining to get 14
    remaining = [p for p in grid_pts if p not in corners and p not in face_centers]
    init_grid = np.array(corners + face_centers + remaining[:2], dtype=float)
    init_grid = init_grid - np.mean(init_grid, axis=0)
    candidates.append(init_grid.flatten())
    
    # 2. Spherical Fibonacci-like distribution
    golden_angle = np.pi * (3 - np.sqrt(5))
    pts_fib = np.zeros((N, D))
    for i in range(N):
        z = 1 - 2 * i / (N - 1)
        radius = np.sqrt(1 - z**2)
        theta = golden_angle * i
        pts_fib[i] = [radius * np.cos(theta), radius * np.sin(theta), z]
    pts_fib = pts_fib - np.mean(pts_fib, axis=0)
    candidates.append(pts_fib.flatten())
    
    # 3. Two concentric shells
    for shell_ratio in [0.5, 0.6, 0.7]:
        pts_shell = np.zeros((N, D))
        n_outer = N // 2
        n_inner = N - n_outer
        for i in range(n_outer):
            theta = 2 * np.pi * i / n_outer
            phi = np.pi / 3
            pts_shell[i] = [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
        for i in range(n_inner):
            theta = 2 * np.pi * i / n_inner + np.pi / n_inner
            phi = 2 * np.pi / 3
            pts_shell[n_outer + i] = shell_ratio * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        pts_shell = pts_shell - np.mean(pts_shell, axis=0)
        candidates.append(pts_shell.flatten())
    
    # 4. Random starts with various distributions
    for trial in range(40):
        if time.time() - start_time > deadline - 15:
            break
        
        if trial % 4 == 0:
            # Uniform in cube
            pts = rng.uniform(-1, 1, size=(N, D))
        elif trial % 4 == 1:
            # Normal, scaled to unit sphere
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-6) * 1.5
        elif trial % 4 == 2:
            # Spherical shell with random radius
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / np.maximum(norms, 1e-6) * (0.8 + 0.4 * rng.uniform(0, 1, size=(N, 1)))
        else:
            # Random with some structure: pairs close together
            pts = rng.uniform(-1, 1, size=(N, D))
            for i in range(0, N, 2):
                if i + 1 < N:
                    pts[i+1] = pts[i] + rng.normal(0, 0.3, size=D)
        
        pts = pts - np.mean(pts, axis=0)
        candidates.append(pts.flatten())
    
    # Optimize each candidate
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > deadline - 10:
            break
        
        x_curr = init_x.copy()
        
        # Phase 1: Coarse optimization with smooth objective
        try:
            res1 = minimize(
                objective_smooth,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts1 = res1.x.reshape(N, D)
            update_best(pts1)
            x_curr = res1.x.copy()
        except Exception:
            pass
        
        # Phase 2: Fine optimization with higher tau
        if time.time() - start_time > deadline - 5:
            continue
            
        try:
            res2 = minimize(
                objective_precise,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pts2 = res2.x.reshape(N, D)
            update_best(pts2)
            if raw_score(pts2) < raw_score(pts1):
                x_curr = res2.x.copy()
        except Exception:
            pass
        
        # Phase 3: Local perturbation refinement
        if time.time() - start_time > deadline - 3:
            continue
            
        for perturb_trial in range(8):
            if time.time() - start_time > deadline - 2:
                break
            
            # Geometric perturbation: move one point slightly
            idx = rng.integers(0, N)
            x_pert = x_curr.copy()
            x_pert[idx*D:(idx+1)*D] += rng.normal(0, 0.02, size=D)
            
            try:
                res_p = minimize(
                    objective_smooth,
                    x_pert,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts_p = res_p.x.reshape(N, D)
                update_best(pts_p)
            except Exception:
                pass
    
    # Final high-precision polish on best solution
    if best_points is not None and time.time() - start_time < deadline - 1:
        try:
            final_res = minimize(
                objective_precise,
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            pts_final = final_res.x.reshape(N, D)
            update_best(pts_final)
        except Exception:
            pass
    
    # Final fallback
    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    sq_dists = get_pairwise_sq_dists(pts)
    score = (np.max(sq_dists) / np.min(sq_dists))
    print(f"Score: {score}")
