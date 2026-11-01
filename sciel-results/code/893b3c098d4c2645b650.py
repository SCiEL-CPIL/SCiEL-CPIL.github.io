import numpy as np
import scipy.optimize as so
import time
import itertools

N = 14
D = 3

def _pairwise_distances_squared(X):
    """Compute squared pairwise distances for points X (N x D)."""
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    return dist2

def _raw_score(X):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    X is an array of shape (N, D).
    """
    dist2 = _pairwise_distances_squared(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    if len(d2) == 0:
        return 1e10
    d_min = np.min(d2)
    d_max = np.max(d2)
    if d_min < 1e-12:
        return 1e10
    return (d_max / d_min)

def _objective_smooth(x, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    x is a flattened array of length N*D.
    """
    X = x.reshape(N, D)
    
    # Center the points to remove translation invariance
    X = X - np.mean(X, axis=0, keepdims=True)
    
    # Compute squared distances
    dist2 = _pairwise_distances_squared(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    
    # Handle degenerate cases
    if np.any(d2 < 1e-15):
        return 1e10
    
    # Normalize to fix scale: make the mean squared distance = 1
    mean_sq_dist = np.mean(d2)
    if mean_sq_dist < 1e-12:
        return 1e10
        
    X = X / np.sqrt(mean_sq_dist)
    dist2 = _pairwise_distances_squared(X)
    mask = ~np.eye(N, dtype=bool)
    d2 = dist2[mask]
    
    # Log squared distances
    log_sq_dists = np.log(d2)
    
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
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_score = 1e10
    best_X = None
    
    def update_best(X):
        nonlocal best_score, best_X
        if X is None:
            return
        X = np.asarray(X)
        if X.shape != (N, D):
            return
        # Check for duplicate points
        score = _raw_score(X)
        if score < best_score:
            best_score = score
            best_X = X.copy()

    rng = np.random.default_rng(42)
    
    # --- Symmetric T_d Optimization ---
    # Define base points for T_d orbits
    # Orbit 1: 4 vertices of a regular tetrahedron
    # Normalized such that norm is 1
    t1 = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    t1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
    
    # Orbit 2: 6 vertices of a regular octahedron (edge midpoints of dual tetra)
    t2 = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ], dtype=float)
    t2 = t2 / np.linalg.norm(t2, axis=1, keepdims=True)
    
    # Orbit 3: 4 vertices of the dual tetrahedron (face centers of t1)
    # The dual tetrahedron vertices are proportional to (-1, -1, -1), etc?
    # Let's check: Normal to face (1,1,1)-(1,-1,-1)-(-1,1,-1) is (-1, -1, -1) ?
    # Face 1: pts (1,1,1), (1,-1,-1), (-1,1,-1).
    # Vector 1: (0, -2, -2). Vector 2: (-2, 0, -2).
    # Cross: (4, -4, -4) -> (1, -1, -1).
    # Wait, the dual tetra vertices are indeed the negatives of the original vertices if we center them?
    # Original: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1).
    # Negatives: (-1,-1,-1), (-1,1,1), (1,-1,1), (1,1,-1).
    # Let's verify if these form a regular tetra.
    # Dist between (-1,-1,-1) and (-1,1,1) is sqrt(0+4+4)=sqrt(8).
    # Yes, it's a regular tetra.
    t3 = -t1.copy()
    # t3 is just t1 rotated/reflected. It's the same set of directions if we consider the group, 
    # but as a set of points, t3 is the dual.
    # Actually, t1 and -t1 are the same set of lines through origin, but different points.
    # So Orbit 3 is just -t1.
    
    def build_symmetric_X(params):
        r1, r2, r3 = params
        # Ensure positive radii to avoid overlap issues at 0? 
        # We can allow negative values to flip, but radii should be > 0.
        # We will optimize log(r) or just r with bounds.
        pts1 = r1 * t1
        pts2 = r2 * t2
        pts3 = r3 * t3
        return np.vstack([pts1, pts2, pts3])
    
    def obj_symmetric(log_params):
        # Use log parameters to keep them positive
        r1, r2, r3 = np.exp(log_params)
        X = build_symmetric_X([r1, r2, r3])
        return _objective_smooth(X.flatten(), tau=100.0)
    
    # Initial guess for symmetric solution
    # Based on regular tetra geometry: r1=1, r2=0.5, r3=0.333
    # Let's start near there
    log_params0 = np.log([1.0, 0.5, 0.333])
    
    # Optimize the 3 parameters
    try:
        res_sym = so.minimize(
            obj_symmetric, 
            log_params0, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res_sym.success or res_sym.fun < 1e10:
            r1, r2, r3 = np.exp(res_sym.x)
            X_sym = build_symmetric_X([r1, r2, r3])
            update_best(X_sym)
    except Exception as e:
        pass

    # --- General Optimization with Diverse Initializations ---
    
    def generate_tetrahedron_with_midpoints():
        """Generate tetrahedron vertices + edge midpoints + face centers (14 points)."""
        a = 1.0
        tet = np.array([
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a]
        ], dtype=float)
        tet = tet / np.linalg.norm(tet, axis=1, keepdims=True)
        
        edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        mids = []
        for i, j in edges:
            mid = (tet[i] + tet[j]) / 2.0
            mids.append(mid)
        mids = np.array(mids)
        
        faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        face_centers = []
        for f in faces:
            fc = np.mean(tet[list(f)], axis=0)
            face_centers.append(fc)
        face_centers = np.array(face_centers)
        
        all_pts = np.vstack([tet, mids, face_centers])
        
        # Apply random rotation
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        all_pts = rot.apply(all_pts)
        
        all_pts -= np.mean(all_pts, axis=0)
        return all_pts

    def generate_icosahedron_plus():
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        
        p1 = rng.normal(size=D)
        p1 = p1 / np.linalg.norm(p1)
        p2 = -p1 # Antipodal to maintain some symmetry? Or random.
        # Let's try antipodal for D3d symmetry potential
        full_pts = np.vstack([ico, p1, p2])
        
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        full_pts = rot.apply(full_pts)
        
        full_pts -= np.mean(full_pts, axis=0)
        return full_pts

    def generate_fibonacci_sphere():
        N_pts = N
        golden_angle = np.pi * (3. - np.sqrt(5.))
        X = np.zeros((N_pts, 3))
        for i in range(N_pts):
            z = 1. - (i / (N_pts - 1.)) * 2.
            radius = np.sqrt(1. - z**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            X[i] = [x, y, z]
        X -= np.mean(X, axis=0)
        return X

    def generate_random_sphere():
        X = rng.normal(size=(N, D))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(rng.normal(size=4))
        X = rot.apply(X)
        X -= np.mean(X, axis=0)
        return X

    inits = []
    # Use the symmetric solution as a strong init if it worked
    if best_X is not None:
        inits.append(best_X)
    
    for _ in range(5):
        inits.append(generate_tetrahedron_with_midpoints())
    for _ in range(5):
        inits.append(generate_icosahedron_plus())
    for _ in range(5):
        inits.append(generate_fibonacci_sphere())
    for _ in range(5):
        inits.append(generate_random_sphere())
    
    # Optimization Loop
    tau_schedule = [10.0, 50.0, 200.0, 1000.0]
    
    n_restarts = len(inits)
    candidates = [X.flatten() for X in inits]
    
    # Phase 1: Coarse
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        x0 = candidates[i]
        try:
            res = so.minimize(
                _objective_smooth, 
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

    # Phase 2: Medium
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        x0 = candidates[i]
        try:
            res = so.minimize(
                _objective_smooth, 
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

    # Phase 3: Fine
    for i in range(n_restarts):
        if time.time() > deadline - 10:
            break
        x0 = candidates[i]
        try:
            res = so.minimize(
                _objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(tau_schedule[2],),
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts = res.x.reshape(N, D)
                update_best(pts)
                candidates[i] = res.x
        except Exception:
            pass

    # Phase 4: Polish best candidates
    if best_X is not None and time.time() < deadline - 2:
        # Take the top 3 candidates from the list (re-evaluate to sort)
        scored_candidates = []
        for c in candidates:
            if c is None: continue
            X = c.reshape(N, D)
            s = _raw_score(X)
            scored_candidates.append((s, c))
        
        scored_candidates.sort(key=lambda x: x[0])
        
        for s, x0 in scored_candidates[:3]:
            if time.time() > deadline - 1:
                break
            X_pert = x0.reshape(N, D)
            X_pert = X_pert + rng.normal(size=(N, D), scale=0.005)
            X_pert -= np.mean(X_pert, axis=0)
            try:
                res = so.minimize(
                    lambda x: _objective_smooth(x, tau=5000.0),
                    X_pert.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res.x is not None:
                    pts = res.x.reshape(N, D)
                    update_best(pts)
            except Exception:
                pass

    if best_X is None:
        # Fallback
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_X
