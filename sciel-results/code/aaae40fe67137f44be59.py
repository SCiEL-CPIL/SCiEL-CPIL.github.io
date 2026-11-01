"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

# EVOLVE-BLOCK-START

def get_pairwise_sq_dists(points):
    """Compute squared pairwise distances as flat array of length N*(N-1)/2."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Actual score: (max_dist / min_dist)^2 = max_sq_dist / min_sq_dist."""
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    return np.max(sq_dists) / np.min(sq_dists)

def _normalize_points(points):
    """Center and normalize so mean squared distance = 1."""
    pts = points - np.mean(points, axis=0, keepdims=True)
    sq_dists = get_pairwise_sq_dists(pts)
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-12:
        return pts
    return pts / np.sqrt(mean_sq)

def objective_smooth(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist) using log-sum-exp.
    Centering and normalization remove translation/scale invariance.
    """
    points = x.reshape(N, D)
    points = _normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    log_sq = np.log(sq_dists)
    
    # Soft max via log-sum-exp
    m_max = np.max(log_sq)
    soft_max = m_max + np.log(np.sum(np.exp(log_sq - m_max))) / tau
    
    # Soft min via negative log-sum-exp
    m_min = np.min(log_sq)
    soft_min = m_min - np.log(np.sum(np.exp(m_min - log_sq))) / tau
    
    return soft_max - soft_min

def objective_raw(x):
    """Raw objective for final polish."""
    points = x.reshape(N, D)
    points = _normalize_points(points)
    return raw_score(points)

def _generate_initializations(rng):
    """Generate diverse initial configurations."""
    candidates = []
    
    # 1. Fibonacci sphere with 14 points (various rotations)
    golden = (1 + np.sqrt(5)) / 2
    for i in range(5):
        pts = []
        for k in range(N):
            z = 1 - 2 * k / (N - 1)
            r = np.sqrt(1 - z ** 2)
            theta = 2 * np.pi * k / golden
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
        pts = np.array(pts)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts)
    
    # 2. Icosahedron (12 pts) + 2 additional points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    for i in range(4):
        p1 = rng.normal(size=D); p1 /= np.linalg.norm(p1)
        p2 = rng.normal(size=D); p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1, p2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts)
    
    # 3. Two rings of 7 points each
    for i in range(3):
        angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles2 = angles1 + np.pi / 7
        ring1 = np.column_stack([np.cos(angles1), np.sin(angles1), np.zeros(7)])
        ring2 = np.column_stack([np.cos(angles2), np.sin(angles2), np.full(7, 0.4)])
        pts = np.vstack([ring1, ring2])
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts)
    
    # 4. Random spherical points
    for i in range(4):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts)
    
    # 5. Tetrahedron-based (from current program)
    a = 1.0
    verts = np.array([
        [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]
    ], dtype=float)
    verts -= np.mean(verts, axis=0)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    verts /= edge_len
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    mids = np.array([(verts[i]+verts[j])/2 for i,j in edges])
    faces = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
    fcs = np.array([(verts[i]+verts[j]+verts[k])/3 for i,j,k in faces])
    pts = np.vstack([verts, mids, fcs])
    pts -= np.mean(pts, axis=0)
    candidates.append(pts)
    
    return candidates

def construct():
    """Construct 14 points in 3D to minimize squared max/min distance ratio."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    best_points = None
    best_score = np.inf
    
    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    # Generate diverse initializations
    inits = _generate_initializations(rng)
    
    # Tau schedule: progressive annealing from smooth to sharp
    tau_schedule = [5.0, 15.0, 50.0, 150.0, 500.0]
    
    # Phase 1: Quick L-BFGS-B with low tau on all initializations
    for i in range(len(inits)):
        if time.time() > deadline - 20.0:
            break
        x0 = inits[i].flatten()
        try:
            res = minimize(
                objective_smooth, x0,
                method='L-BFGS-B',
                args=(tau_schedule[0],),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                inits[i] = pts  # carry forward
        except Exception:
            pass
    
    # Phase 2: Progressive tau increase, refining each candidate
    for tau_idx in range(1, len(tau_schedule)):
        tau = tau_schedule[tau_idx]
        max_iter = 80 + tau_idx * 50
        for i in range(len(inits)):
            if time.time() > deadline - 15.0:
                break
            x0 = inits[i].flatten()
            try:
                res = minimize(
                    objective_smooth, x0,
                    method='L-BFGS-B',
                    args=(tau,),
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
                    inits[i] = pts
            except Exception:
                pass
    
    # Phase 3: Final polish with raw objective (exact, non-smooth)
    # Take top few candidates and do high-precision Nelder-Mead
    if best_points is not None and time.time() < deadline - 10.0:
        # Use the best point as starting point for final polish
        x0 = best_points.flatten()
        try:
            res = minimize(
                objective_raw, x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
        except Exception:
            pass
        
        # Also try a second polish from a slightly perturbed version
        if time.time() < deadline - 5.0:
            x0 = best_points.flatten() + rng.normal(size=N*D) * 1e-4
            try:
                res = minimize(
                    objective_raw, x0,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
            except Exception:
                pass
    
    # Fallback if nothing worked
    if best_points is None:
        best_points = inits[0].copy()
    
    # Final validation: ensure distinctness
    sq_dists = get_pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-8:
        # Perturb the closest pair
        iu = np.triu_indices(N, k=1)
        sq_flat = sq_dists
        min_idx = np.argmin(sq_flat)
        i, j = iu[0][min_idx], iu[1][min_idx]
        best_points[j] += 1e-5 * rng.normal(size=D)
    
    return best_points

# EVOLVE-BLOCK-END
