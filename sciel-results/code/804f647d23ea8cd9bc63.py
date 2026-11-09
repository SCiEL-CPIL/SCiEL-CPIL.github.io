import numpy as np
from scipy.spatial.distance import pdist
import time


def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s limit
    n = 14

    def score(pts):
        """Compute (d_min/d_max)^2."""
        d = pdist(pts)
        d_min = float(d.min())
        d_max = float(d.max())
        if d_max <= 1e-15:
            return 0.0
        r = d_min / d_max
        return r * r

    def neg_score(pts):
        """Negative score for minimization."""
        return -score(pts)

    def make_tetrahedron_base():
        """
        Create 14 points as 4 tetrahedron vertices + 10 edge midpoints.
        Returns array of shape (14, 3) on unit sphere.
        The 5 symmetry orbits are:
        - Orbit 0: 4 vertices (radius r_v)
        - Orbit 1-4: 5 pairs of edge midpoints? No, there are 6 edges, so 6 midpoints.
        Wait, a tetrahedron has 6 edges. 4 vertices + 6 midpoints = 10 points.
        We need 14 points.
        
        Let's reconsider the symmetry. 
        A regular tetrahedron has symmetry group T_d (order 24).
        14 points... 
        Maybe: 4 vertices + 6 edge midpoints + 4 face centers? That's 14.
        Yes! 4 vertices, 6 edge midpoints, 4 face centers.
        
        Orbits under T_d:
        - Vertices: 1 orbit of 4
        - Edge midpoints: 1 orbit of 6
        - Face centers: 1 orbit of 4
        
        So we have 3 orbits, each with a single radial parameter.
        But the base positions are fixed by symmetry.
        
        Let's define the base positions on the unit sphere.
        Tetrahedron vertices:
        v1 = (1, 1, 1) / sqrt(3)
        v2 = (1, -1, -1) / sqrt(3)
        v3 = (-1, 1, -1) / sqrt(3)
        v4 = (-1, -1, 1) / sqrt(3)
        
        Edge midpoints (average of two vertices):
        m12 = (v1+v2)/2, normalized
        m13 = (v1+v3)/2, normalized
        m14 = (v1+v4)/2, normalized
        m23 = (v2+v3)/2, normalized
        m24 = (v2+v4)/2, normalized
        m34 = (v3+v4)/2, normalized
        
        Face centers (average of three vertices):
        f123 = (v1+v2+v3)/3, normalized
        f124 = (v1+v2+v4)/3, normalized
        f134 = (v1+v3+v4)/3, normalized
        f234 = (v2+v3+v4)/3, normalized
        """
        sqrt3 = np.sqrt(3.0)
        v = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1]
        ], dtype=float) / sqrt3
        
        # 4 vertices
        verts = v.copy()
        
        # 6 edge midpoints
        edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        mids = []
        for i, j in edges:
            m = (v[i] + v[j]) / 2.0
            m = m / np.linalg.norm(m)
            mids.append(m)
        mids = np.array(mids)
        
        # 4 face centers
        faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        faces_arr = []
        for i, j, k in faces:
            f = (v[i] + v[j] + v[k]) / 3.0
            f = f / np.linalg.norm(f)
            faces_arr.append(f)
        faces_arr = np.array(faces_arr)
        
        return np.vstack([verts, mids, faces_arr])

    def make_icosahedron_plus_poles():
        """
        12 icosahedron vertices + 2 poles = 14 points.
        Orbits:
        - Icosahedron vertices: 1 orbit of 12
        - Poles: 1 orbit of 2
        Total 2 orbits.
        """
        t = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [-1, 1, t], [1, 1, t], [-1, -1, t], [1, -1, t],
            [1, t, -1], [1, t, 1], [-1, t, -1], [-1, t, 1],
            [t, -1, -1], [t, 1, -1], [t, -1, 1], [t, 1, 1]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        poles = np.array([[0, 0, 1.0], [0, 0, -1.0]])
        return np.vstack([ico, poles])

    def make_fibonacci_sphere():
        """Fibonacci sphere distribution."""
        k = np.arange(n)
        phi_angle = np.arccos(1 - 2 * (k + 0.5) / n)
        golden = np.pi * (1 + np.sqrt(5))
        theta_angle = golden * k
        theta_angle = np.mod(theta_angle, 2 * np.pi)
        x = np.sin(phi_angle) * np.cos(theta_angle)
        y = np.sin(phi_angle) * np.sin(theta_angle)
        z = np.cos(phi_angle)
        return np.column_stack([x, y, z])

    def random_sphere(seed=42):
        """Random points on sphere."""
        rng = np.random.RandomState(seed)
        theta = rng.uniform(0, np.pi, size=n)
        phi = rng.uniform(0, 2 * np.pi, size=n)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])

    def optimize_radial_orbits(base_pts, orbit_indices, max_iter=200):
        """
        Optimize radial scaling factors for each orbit.
        base_pts: (14, 3) unit sphere points
        orbit_indices: list of lists, each list contains indices of points in that orbit
        Returns optimized points and score.
        """
        n_orbits = len(orbit_indices)
        # Initial radii all 1.0
        radii = np.ones(n_orbits)
        
        def build_pts(radii):
            pts = np.zeros((n, 3))
            for o, idx_list in enumerate(orbit_indices):
                for idx in idx_list:
                    pts[idx] = radii[o] * base_pts[idx]
            return pts
        
        best_pts = build_pts(radii)
        best_score = score(best_pts)
        
        # Coordinate descent on radii
        step_sizes = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005])
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for o in range(n_orbits):
                for ss in step_sizes:
                    for direction in [1, -1]:
                        new_radii = radii.copy()
                        new_radii[o] = radii[o] + direction * ss
                        if new_radii[o] < 0.1:  # Keep radii positive
                            new_radii[o] = 0.1
                        new_pts = build_pts(new_radii)
                        new_score = score(new_pts)
                        if new_score > best_score:
                            best_score = new_score
                            best_pts = new_pts
                            radii = new_radii
                            improved = True
            if not improved:
                break
        
        return best_pts, best_score

    def simulated_annealing(pts, steps=3000, T_start=0.1, T_end=0.001, seed=42):
        """Simulated annealing in 3D Cartesian coordinates with sphere projection."""
        rng = np.random.RandomState(seed)
        current_pts = pts.copy()
        current_score = score(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        for step in range(steps):
            if time.time() > deadline - 20:
                break
            T = T_start * (T_end / T_start) ** (step / max(steps - 1, 1))
            
            # Perturb 1-2 points
            n_perturb = rng.choice([1, 2], p=[0.7, 0.3])
            indices = rng.choice(n, size=n_perturb, replace=False)
            
            new_pts = current_pts.copy()
            for idx in indices:
                noise = rng.randn(3) * 0.1 * (T / T_start + 0.1)
                new_pts[idx] += noise
                # Project back to unit sphere
                norm = np.linalg.norm(new_pts[idx])
                if norm > 1e-12:
                    new_pts[idx] /= norm
            
            new_score = score(new_pts)
            
            if new_score > current_score or rng.rand() < np.exp((new_score - current_score) / max(T, 1e-10)):
                current_pts = new_pts
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        return best_pts, best_score

    def nelder_mead_optimize(pts, max_iter=300):
        """Nelder-Mead optimization in 3D Cartesian with sphere projection."""
        from scipy.optimize import minimize
        
        def wrapped_neg_score(x):
            pts_arr = x.reshape(n, 3)
            # Project to unit sphere
            norms = np.linalg.norm(pts_arr, axis=1, keepdims=True)
            mask = norms < 1e-12
            if np.any(mask):
                norms[mask] = 1.0
                pts_arr[mask, 0] = 1.0
                pts_arr[mask, 1:] = 0.0
            pts_arr = pts_arr / norms
            return neg_score(pts_arr)
        
        try:
            res = minimize(wrapped_neg_score, pts.flatten(), method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
            x_opt = res.x.reshape(n, 3)
            # Project to unit sphere
            norms = np.linalg.norm(x_opt, axis=1, keepdims=True)
            mask = norms < 1e-12
            if np.any(mask):
                norms[mask] = 1.0
                x_opt[mask, 0] = 1.0
                x_opt[mask, 1:] = 0.0
            x_opt = x_opt / norms
            r = -float(res.fun)
            if np.isfinite(r):
                return x_opt, r
        except Exception:
            pass
        return pts, -float('inf')

    # Generate initial candidates
    np.random.seed(12345)
    candidates = [
        make_tetrahedron_base(),
        make_icosahedron_plus_poles(),
        make_fibonacci_sphere(),
        random_sphere(seed=1),
        random_sphere(seed=2),
        random_sphere(seed=3),
    ]

    # Add perturbed versions
    for base in [make_tetrahedron_base(), make_icosahedron_plus_poles(), make_fibonacci_sphere()]:
        for noise_level in [0.05, 0.1, 0.2]:
            perturbed = base + noise_level * np.random.randn(n, 3)
            # Project to unit sphere
            norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
            mask = norms < 1e-12
            if np.any(mask):
                norms[mask] = 1.0
                perturbed[mask, 0] = 1.0
                perturbed[mask, 1:] = 0.0
            perturbed = perturbed / norms
            candidates.append(perturbed)

    best_pts = None
    best_score = -1.0

    # Phase 1: Optimize radial orbits for symmetric templates
    if time.time() < deadline - 300:
        # Tetrahedron template: orbits are [0-3], [4-9], [10-13]
        tetra_base = make_tetrahedron_base()
        tetra_orbits = [list(range(4)), list(range(4, 10)), list(range(10, 14))]
        try:
            opt_pts, opt_score = optimize_radial_orbits(tetra_base, tetra_orbits, max_iter=500)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            pass

        # Icosahedron template: orbits are [0-11], [12-13]
        ico_base = make_icosahedron_plus_poles()
        ico_orbits = [list(range(12)), list(range(12, 14))]
        try:
            opt_pts, opt_score = optimize_radial_orbits(ico_base, ico_orbits, max_iter=500)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts
        except Exception:
            pass

    # Phase 2: Simulated Annealing from multiple starts
    for i, cand in enumerate(candidates[:10]):
        if time.time() > deadline - 150:
            break
        try:
            sa_pts, sa_score = simulated_annealing(cand, steps=2000, T_start=0.05, T_end=0.001, seed=100+i)
            if np.isfinite(sa_score) and sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception:
            pass

    # Phase 3: Nelder-Mead from best SA result and original candidates
    opt_candidates = []
    if best_pts is not None:
        opt_candidates.append(best_pts)
    opt_candidates.extend(candidates[:4])
    
    for i, cand in enumerate(opt_candidates):
        if time.time() > deadline - 80:
            break
        try:
            nm_pts, nm_score = nelder_mead_optimize(cand, max_iter=300)
            if np.isfinite(nm_score) and nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts
        except Exception:
            pass

    # Phase 4: Final polish
    if best_pts is not None and time.time() < deadline - 30:
        try:
            nm_pts, nm_score = nelder_mead_optimize(best_pts, max_iter=400)
            if np.isfinite(nm_score) and nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts
        except Exception:
            pass

    # Phase 5: One more SA round
    if best_pts is not None and time.time() < deadline - 10:
        try:
            sa_pts2, sa_score2 = simulated_annealing(best_pts, steps=1000, T_start=0.02, T_end=0.0005, seed=999)
            if np.isfinite(sa_score2) and sa_score2 > best_score:
                best_score = sa_score2
                best_pts = sa_pts2
        except Exception:
            pass

    if best_pts is None:
        best_pts = make_fibonacci_sphere()

    # Ensure all points are finite and on unit sphere
    best_pts = np.nan_to_num(best_pts, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    mask = norms < 1e-12
    if np.any(mask):
        norms[mask] = 1.0
        best_pts[mask, 0] = 1.0
        best_pts[mask, 1:] = 0.0
    best_pts = best_pts / norms

    return best_pts
