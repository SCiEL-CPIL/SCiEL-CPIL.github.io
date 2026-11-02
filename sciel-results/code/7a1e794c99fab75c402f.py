import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

# Precompute pairwise indices
_IU = np.triu_indices(N, k=1)
_PAIR_COUNT = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return sq_dists[_IU]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_sq)

def objective_log_ratio(x, tau=50.0):
    """
    Smooth approximation of the log of the squared max/min pairwise distance ratio.
    Minimizing this is equivalent to minimizing the ratio itself (monotone).
    """
    points = x.reshape(N, D)
    
    # Center to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    # Normalize to fix scale
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-15:
        return 1e10
    points = points / np.sqrt(mean_sq)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max of log(sq_dists)
    logs = np.log(sq_dists)
    m_max = np.max(logs)
    soft_max = m_max + np.log(np.sum(np.exp(tau * (logs - m_max)))) / tau
    
    # Soft min of log(sq_dists)
    m_min = np.min(logs)
    soft_min = m_min - np.log(np.sum(np.exp(-tau * (logs - m_min)))) / tau
    
    return soft_max - soft_min

def objective_direct(x, tau=100.0):
    """
    Direct smooth objective: minimize soft_max(sq_dists) / soft_min(sq_dists)
    This is a direct ratio rather than log-difference.
    """
    points = x.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-15:
        return 1e10
    points = points / np.sqrt(mean_sq)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max
    m_max = np.max(sq_dists)
    soft_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    soft_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    return soft_max / soft_min

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points at various symmetric positions
    for i in range(12):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 2. Fibonacci sphere
    def fibonacci_sphere(n, rng, offset=0.0):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * (i + offset) / (n - 1)
            z = np.clip(z, -1, 1)
            radius = np.sqrt(max(0, 1 - z**2))
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        return pts
    
    for i in range(8):
        offset = rng.uniform(0, 1)
        pts = fibonacci_sphere(N, rng, offset)
        candidates.append(pts.flatten())

    # 3. Random spherical points
    for i in range(8):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 4. Uniform random in cube
    for i in range(4):
        pts = rng.uniform(-1, 1, size=(N, D))
        candidates.append(pts.flatten())

    # 5. Tetrahedron-based construction
    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    # Midpoints of edges
    mids = []
    for i in range(4):
        for j in range(i+1, 4):
            mids.append((tetra[i] + tetra[j]) / 2)
    mids = np.array(mids)
    mids = mids / np.linalg.norm(mids, axis=1, keepdims=True)
    
    for i in range(4):
        pts = np.vstack([tetra, mids])
        extra = rng.normal(size=(4, D))
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        full_pts = np.vstack([pts, extra])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 6. Octahedron + additional points
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    for i in range(4):
        pts = octa.copy()
        extra = rng.normal(size=(8, D))
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        full_pts = np.vstack([pts, extra])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 7. Cube vertices + face centers
    cube = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=float)
    cube = cube / np.linalg.norm(cube, axis=1, keepdims=True)
    face_centers = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]
    ], dtype=float)
    for i in range(3):
        pts = np.vstack([cube, face_centers])
        extra = rng.normal(size=(2, D))
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        full_pts = np.vstack([pts, extra])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # Optimization Loop
    n_restarts = len(candidates)
    
    # Tau schedule: start low for smooth landscape, increase for precision
    tau_schedule = [3.0, 10.0, 50.0, 200.0, 1000.0]
    
    # Phase 1: Quick optimization with low tau on all candidates
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 15:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 30, 'ftol': 1e-7, 'gtol': 1e-5}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Progressive tau increase
    for tau_idx in range(1, len(tau_schedule)):
        if time.time() - start_time > timeout - 10:
            break
        tau = tau_schedule[tau_idx]
        max_iter = 50 + tau_idx * 30
        for i in range(n_restarts):
            if time.time() - start_time > timeout - 5:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_log_ratio, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 3: Nelder-Mead polish on best candidates
    if time.time() - start_time < timeout - 8:
        scored_candidates = []
        for i in range(n_restarts):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored_candidates.append((score, i))
        scored_candidates.sort()
        
        for score, i in scored_candidates[:8]:
            if time.time() - start_time > timeout - 3:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_direct, 
                    x0, 
                    method='Nelder-Mead', 
                    args=(50.0,),
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 4: High-precision L-BFGS polish
    if time.time() - start_time < timeout - 5:
        scored_candidates = []
        for i in range(n_restarts):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored_candidates.append((score, i))
        scored_candidates.sort()
        
        for score, i in scored_candidates[:5]:
            if time.time() - start_time > timeout - 2:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_log_ratio, 
                    x0, 
                    method='L-BFGS-B', 
                    args=(tau_schedule[-1],),
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 5: Additional random restarts if time remains
    while time.time() - start_time < timeout - 4:
        # Generate diverse random initializations
        init_type = rng.integers(0, 3)
        if init_type == 0:
            x0 = rng.normal(size=(N * D))
        elif init_type == 1:
            pts = rng.normal(size=(N, D))
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            x0 = pts.flatten()
        else:
            x0 = rng.uniform(-1, 1, size=(N * D))
        
        try:
            res = minimize(
                objective_log_ratio, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[1],),
                options={'maxiter': 60, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                
                if raw_score(pts) < best_score * 1.15 and time.time() - start_time < timeout - 3:
                    try:
                        res2 = minimize(
                            objective_direct, 
                            res.x, 
                            method='Nelder-Mead', 
                            args=(100.0,),
                            options={'maxiter': 150, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                        )
                        if res2.x is not None:
                            pts2 = res2.x.reshape(N, D)
                            update_best(pts2)
                    except Exception:
                        pass
        except Exception:
            pass

    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
