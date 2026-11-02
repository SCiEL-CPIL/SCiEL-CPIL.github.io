import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation

N = 14
D = 3

# Precompute pair indices for speed
_pair_i, _pair_j = np.triu_indices(N, k=1)
_NUM_PAIRS = len(_pair_i)

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    return sq_dists[_pair_i, _pair_j]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def objective_smooth(x, tau=100.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Minimizing this minimizes the ratio of max to min distance.
    Uses log-sum-exp for stability.
    """
    points = x.reshape(N, D)
    
    # Compute squared distances
    sq_dists = get_pairwise_sq_dists(points)
    
    # Handle degenerate cases
    if np.any(sq_dists < 1e-15):
        return 1e10
    
    # Log squared distances
    log_sq_dists = np.log(sq_dists)
    
    # Soft max of log(squared distances)
    m_max = np.max(log_sq_dists)
    exp_max_terms = np.exp(tau * (log_sq_dists - m_max))
    soft_max_log = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Soft min of log(squared distances)
    m_min = np.min(log_sq_dists)
    exp_min_terms = np.exp(-tau * (log_sq_dists - m_min))
    soft_min_log = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    # Objective is the difference
    obj = soft_max_log - soft_min_log
    
    return obj

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        # Check for duplicate points
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Icosahedron + 2 points (various placements)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Strategy A: Add on sphere, random angles
    for _ in range(20):
        pts = ico.copy()
        # Generate 2 random points on sphere
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = rng.normal(size=D)
        p2 = p2 / np.linalg.norm(p2)
        full_pts = np.vstack([pts, p1, p2])
        # Apply random rotation to break symmetry
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        candidates.append(full_pts.flatten())

    # Strategy B: Add points slightly inside/outside? No, usually optimal is on convex hull.
    # But let's try perturbing the icosahedron vertices.
    
    # 2. Fibonacci sphere with 14 points
    def fibonacci_sphere(n, rng):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - 2 * i / (n - 1)
            radius = np.sqrt(1 - z**2)
            theta = 2 * np.pi * i / golden_ratio
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            points.append([x, y, z])
        pts = np.array(points)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        return pts
    
    for _ in range(20):
        pts = fibonacci_sphere(N, rng)
        candidates.append(pts.flatten())

    # 3. Random spherical points
    for _ in range(20):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 4. Two rings of 7 points each (structured)
    for i in range(10):
        ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + rng.uniform(0, 2*np.pi)
        ring2_angles = ring1_angles + np.pi/7
        z_offset = rng.uniform(-0.5, 0.5)
        ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.full(7, z_offset)])
        ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, -z_offset)])
        pts = np.vstack([ring1, ring2])
        # Normalize radius to 1
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        rot = Rotation.from_quat(rng.normal(size=4))
        pts = rot.apply(pts)
        candidates.append(pts.flatten())

    # 5. New Strategy: Icosahedron + 2 points at face centers (radial scaling optimization)
    # Find face centers of icosahedron
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(ico)
        face_centers = []
        for simplex in tri.simplices:
            fc = np.mean(ico[simplex], axis=0)
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc)
        face_centers = np.array(face_centers)
        
        # Add 2 points at opposite face centers with varying radii
        for i in range(10):
            # Pick two roughly opposite face centers
            fc1_idx = rng.integers(0, len(face_centers))
            fc2_idx = rng.integers(0, len(face_centers))
            # Ensure they are somewhat opposite
            if np.dot(face_centers[fc1_idx], face_centers[fc2_idx]) > -0.5:
                continue
            r1 = rng.uniform(0.5, 1.5)
            r2 = rng.uniform(0.5, 1.5)
            p1 = r1 * face_centers[fc1_idx]
            p2 = r2 * face_centers[fc2_idx]
            full_pts = np.vstack([ico, p1, p2])
            rot = Rotation.from_quat(rng.normal(size=4))
            full_pts = rot.apply(full_pts)
            candidates.append(full_pts.flatten())
    except:
        pass

    # Optimization Loop
    # We use a multi-stage tau schedule to anneal the objective.
    tau_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0, 10000.0]
    
    n_restarts = len(candidates)
    
    # Phase 1: Coarse optimization
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 20:
            break
        x0 = candidates[i]
        try:
            # Start with low tau
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[0],),
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 2: Medium optimization
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 15:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[1],),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 3: Fine optimization
    # Sort by current score to prioritize good candidates
    scored_candidates = []
    for i in range(n_restarts):
        pts = candidates[i].reshape(N, D)
        score = raw_score(pts)
        scored_candidates.append((score, i))
    scored_candidates.sort()
    
    for score, i in scored_candidates:
        if time.time() - start_time > timeout - 10:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[2],),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 4: Ultra-fine polish on top candidates
    scored_candidates = []
    for i in range(n_restarts):
        pts = candidates[i].reshape(N, D)
        score = raw_score(pts)
        scored_candidates.append((score, i))
    scored_candidates.sort()
    
    for score, i in scored_candidates[:15]:
        if time.time() - start_time > timeout - 5:
            break
        x0 = candidates[i]
        try:
            # Use very high tau to closely approximate the true objective
            res = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[3],),
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 5: SLSQP polish on top candidates (handles non-smoothness better)
    scored_candidates = []
    for i in range(n_restarts):
        pts = candidates[i].reshape(N, D)
        score = raw_score(pts)
        scored_candidates.append((score, i))
    scored_candidates.sort()
    
    for score, i in scored_candidates[:10]:
        if time.time() - start_time > timeout - 3:
            break
        x0 = candidates[i]
        try:
            res = minimize(
                objective_smooth, 
                x0, 
                method='SLSQP', 
                args=(tau_schedule[4],),
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 6: Final high-precision polish with exact objective
    if time.time() - start_time < timeout - 1:
         scored_candidates = []
         for i in range(n_restarts):
             pts = candidates[i].reshape(N, D)
             score = raw_score(pts)
             scored_candidates.append((score, i))
         scored_candidates.sort()
         
         def exact_objective(x):
             pts = x.reshape(N, D)
             sq_dists = get_pairwise_sq_dists(pts)
             min_sq = np.min(sq_dists)
             if min_sq < 1e-12:
                 return 1e10
             return np.max(sq_dists) / min_sq
             
         for score, i in scored_candidates[:8]:
             if time.time() - start_time > timeout - 0.5:
                 break
             x0 = candidates[i]
             try:
                 # Use BFGS for better convergence on the non-smooth objective
                 res = minimize(
                     exact_objective, 
                     x0, 
                     method='BFGS',
                     options={'maxiter': 200, 'gtol': 1e-12}
                 )
                 if res.x is not None:
                     pts = res.x.reshape(N, D)
                     update_best(pts)
             except Exception:
                 pass

    if best_points is None:
        # Fallback
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
