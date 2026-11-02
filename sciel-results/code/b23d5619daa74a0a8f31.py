import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

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
    points = x.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-15:
        return 1e10
    points = points / np.sqrt(mean_sq)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    logs = np.log(sq_dists)
    m_max = np.max(logs)
    soft_max = m_max + np.log(np.sum(np.exp(tau * (logs - m_max)))) / tau
    m_min = np.min(logs)
    soft_min = m_min - np.log(np.sum(np.exp(-tau * (logs - m_min)))) / tau
    return soft_max - soft_min

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t + penalty for violations.
    t represents the log ratio target.
    """
    n_params = N * D
    points = x[:n_params].reshape(N, D)
    t = x[n_params]
    points = points - np.mean(points, axis=0, keepdims=True)
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-15:
        return 1e10
    points = points / np.sqrt(mean_sq)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    logs = np.log(sq_dists)
    m_max = np.max(logs)
    soft_max = m_max + np.log(np.sum(np.exp(tau * (logs - m_max)))) / tau
    m_min = np.min(logs)
    soft_min = m_min - np.log(np.sum(np.exp(-tau * (logs - m_min)))) / tau
    log_ratio = soft_max - soft_min
    violation = np.maximum(0.0, log_ratio - t)
    penalty = 50.0 * violation ** 2 + 0.01 * t
    return t + penalty

def objective_direct(x, tau=100.0):
    points = x.reshape(N, D)
    points = points - np.mean(points, axis=0, keepdims=True)
    sq_dists = get_pairwise_sq_dists(points)
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-15:
        return 1e10
    points = points / np.sqrt(mean_sq)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    m_max = np.max(sq_dists)
    soft_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    m_min = np.min(sq_dists)
    soft_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    return soft_max / soft_min

def construct():
    start_time = time.time()
    timeout = 145.0
    
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

    # 1. Tetrahedron + edge midpoints + face centers + extra
    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    # 6 edge midpoints
    mids = []
    for i in range(4):
        for j in range(i+1, 4):
            mids.append((tetra[i] + tetra[j]) / 2)
    mids = np.array(mids)
    mids = mids / np.linalg.norm(mids, axis=1, keepdims=True)
    
    # 4 face centers
    face_centers = []
    for i in range(4):
        others = [j for j in range(4) if j != i]
        fc = (tetra[others[0]] + tetra[others[1]] + tetra[others[2]]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    face_centers = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    # That gives us 4 + 6 + 4 = 14 points
    for i in range(6):
        pts = np.vstack([tetra, mids, face_centers])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 2. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    for i in range(8):
        pts = ico.copy()
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 3. Fibonacci sphere
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

    # 4. Random spherical points
    for i in range(6):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 5. Cube + face centers
    cube = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=float)
    cube = cube / np.linalg.norm(cube, axis=1, keepdims=True)
    face_centers_c = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]
    ], dtype=float)
    for i in range(4):
        pts = np.vstack([cube, face_centers_c])
        extra = rng.normal(size=(2, D))
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        full_pts = np.vstack([pts, extra])
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # 6. Two rings
    for i in range(3):
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False)
        ring2_angles = ring1_angles + np.pi/7
        z_offset = rng.uniform(0.3, 0.7)
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.full(7, -z_offset)])
        ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, z_offset)])
        pts = np.vstack([ring1, ring2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 7. Octahedron + extras
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

    n_restarts = len(candidates)
    
    tau_schedule = [5.0, 15.0, 50.0, 150.0, 500.0]
    
    # Phase 1: Quick L-BFGS with low tau
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 20:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_log_ratio, x0, method='L-BFGS-B',
                args=(tau_schedule[0],),
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Progressive tau increase
    for tau_idx in range(1, len(tau_schedule)):
        if time.time() - start_time > timeout - 15:
            break
        tau = tau_schedule[tau_idx]
        max_iter = 60 + tau_idx * 30
        for i in range(n_restarts):
            if time.time() - start_time > timeout - 8:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_log_ratio, x0, method='L-BFGS-B',
                    args=(tau,),
                    options={'maxiter': max_iter, 'ftol': 1e-11, 'gtol': 1e-9}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 3: Epigraph optimization on best candidates
    if time.time() - start_time < timeout - 12:
        scored = []
        for i in range(n_restarts):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored.append((score, i))
        scored.sort()
        
        for score, i in scored[:10]:
            if time.time() - start_time > timeout - 6:
                break
            x0 = candidates[i]
            # Initialize t with current log ratio
            pts = x0.reshape(N, D)
            pts_c = pts - np.mean(pts, axis=0, keepdims=True)
            sq_d = get_pairwise_sq_dists(pts_c)
            if np.min(sq_d) > 1e-15:
                t_init = np.log(np.max(sq_d) / np.min(sq_d))
            else:
                t_init = 1.0
            x_epigraph = np.concatenate([x0, [t_init]])
            try:
                res = minimize(
                    objective_epigraph, x_epigraph, method='L-BFGS-B',
                    args=(tau_schedule[2],),
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x[:N*D].reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x[:N*D]
            except Exception:
                pass

    # Phase 4: Nelder-Mead polish
    if time.time() - start_time < timeout - 8:
        scored = []
        for i in range(n_restarts):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored.append((score, i))
        scored.sort()
        
        for score, i in scored[:8]:
            if time.time() - start_time > timeout - 3:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_direct, x0, method='Nelder-Mead',
                    args=(80.0,),
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 5: High-precision L-BFGS
    if time.time() - start_time < timeout - 5:
        scored = []
        for i in range(n_restarts):
            pts = candidates[i].reshape(N, D)
            score = raw_score(pts)
            scored.append((score, i))
        scored.sort()
        
        for score, i in scored[:6]:
            if time.time() - start_time > timeout - 2:
                break
            x0 = candidates[i]
            try:
                res = minimize(
                    objective_log_ratio, x0, method='L-BFGS-B',
                    args=(tau_schedule[-1],),
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    candidates[i] = res.x
            except Exception:
                pass

    # Phase 6: Additional random restarts
    while time.time() - start_time < timeout - 4:
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
                objective_log_ratio, x0, method='L-BFGS-B',
                args=(tau_schedule[1],),
                options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                if raw_score(pts) < best_score * 1.2 and time.time() - start_time < timeout - 3:
                    try:
                        res2 = minimize(
                            objective_direct, res.x, method='Nelder-Mead',
                            args=(100.0,),
                            options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
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
