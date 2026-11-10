# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import itertools

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Strategy: Symmetry-Constrained Structural Search.
    
    Instead of treating the problem as a generic 42D optimization, we hypothesize that the optimal
    configuration for 14 points in 3D possesses significant symmetry. We construct a set of
    candidate configurations based on known high-symmetry point sets (Platonic solids, rings,
    prisms) and combinatorial subsets thereof. For each candidate, we perform a local refinement
    in a low-dimensional parameter space (radial scaling of symmetric groups) to maximize the
    (d_min / d_max)^2 metric.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # 10s margin

    def get_score(pts):
        """Calculate (d_min / d_max)^2"""
        if pts.shape != (n, d):
            return 0.0
        # Check for coincident points or invalid values
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        if d_min < 1e-12:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def refine_symmetric(base_pts, group_sizes, deadline):
        """
        Refine a symmetric configuration by adjusting the radial distance of each symmetry group.
        base_pts: (n, 3) array of base directions (normalized).
        group_sizes: list of sizes of symmetry groups. Sum must be n.
        Returns best points and score.
        """
        if time.time() > deadline:
            return base_pts, get_score(base_pts)
        
        # Normalize base directions
        norms = np.linalg.norm(base_pts, axis=1, keepdims=True)
        norms[norms == 0] = 1
        dirs = base_pts / norms
        
        # Parameters: radial scale for each group
        # Initial guess: 1.0 for all
        init_params = np.ones(len(group_sizes))
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            pts = np.zeros((n, 3))
            idx = 0
            for i, size in enumerate(group_sizes):
                # All points in this group have the same radial scale params[i]
                # but potentially different directions (from base_pts)
                pts[idx:idx+size] = dirs[idx:idx+size] * params[i]
                idx += size
            
            # Center of mass correction to remove translation invariance
            # Note: Scaling groups radially might shift COM, so we re-center.
            # However, re-centering changes relative distances.
            # The metric is translation invariant, so we can just center the result.
            pts -= np.mean(pts, axis=0)
            
            score = get_score(pts)
            return -score

        best_score = get_score(base_pts)
        best_pts = base_pts.copy()
        
        try:
            # Use Nelder-Mead on the small parameter space
            res = minimize(
                objective, 
                init_params, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if res.fun < -best_score:
                best_score = -res.fun
                # Reconstruct best points
                params = res.x
                pts = np.zeros((n, 3))
                idx = 0
                for i, size in enumerate(group_sizes):
                    pts[idx:idx+size] = dirs[idx:idx+size] * params[i]
                    idx += size
                pts -= np.mean(pts, axis=0)
                best_pts = pts
        except Exception:
            pass
            
        return best_pts, best_score

    best_score_global = -np.inf
    best_pts_global = None

    # Helper to generate regular polygons
    def make_ring(n_points, radius=1.0, z=0.0):
        pts = np.zeros((n_points, 3))
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        pts[:, 0] = radius * np.cos(angles)
        pts[:, 1] = radius * np.sin(angles)
        pts[:, 2] = z
        return pts

    # Generate Candidate Configurations
    candidates = []

    # 1. Icosahedron (12) + 2 Poles (2)
    # This is a strong candidate. Groups: 12 (icosahedron vertices), 1 (pole), 1 (pole)
    # But poles are distinct. Let's treat as: 12 ico verts, 2 poles.
    phi_golden = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi_golden], [0, -1, phi_golden],
        [0, 1, -phi_golden], [0, -1, -phi_golden],
        [1, phi_golden, 0], [-1, phi_golden, 0],
        [1, -phi_golden, 0], [-1, -phi_golden, 0],
        [phi_golden, 0, 1], [-phi_golden, 0, 1],
        [phi_golden, 0, -1], [-phi_golden, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    poles = np.array([[0, 0, 1], [0, 0, -1]])
    cand_ico_poles = np.vstack([ico_verts, poles])
    candidates.append((cand_ico_poles, [12, 1, 1]))

    # 2. Two Rings of 7? No, 14 is even.
    # 3. Two Rings of 6 + 2 Poles? (6+6+2=14)
    # Groups: 6 (top ring), 6 (bottom ring), 1 (pole), 1 (pole)
    ring1 = make_ring(6, 1.0, 0.5)
    ring2 = make_ring(6, 1.0, -0.5)
    cand_rings_poles = np.vstack([ring1, ring2, poles])
    candidates.append((cand_rings_poles, [6, 6, 1, 1]))

    # 4. Three Rings? 5+5+4? Or 4+5+5?
    # Let's try 4+5+5 with different Z.
    ring_a = make_ring(4, 1.0, 0.6)
    ring_b = make_ring(5, 1.0, 0.0)
    ring_c = make_ring(5, 1.0, -0.6)
    cand_3rings = np.vstack([ring_a, ring_b, ring_c])
    candidates.append((cand_3rings, [4, 5, 5]))

    # 5. Tetrahedron (4) + 6 Edge Midpoints + 4 Face Centers? 
    # Total 14.
    # Tetra verts: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) normalized.
    t_verts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ])
    t_verts /= np.linalg.norm(t_verts, axis=1, keepdims=True)
    
    # Edge midpoints (6 edges)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    edge_mids = []
    for i, j in edges:
        mid = (t_verts[i] + t_verts[j]) / 2
        mid /= np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers (4 faces)
    faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    face_centers = []
    for i, j, k in faces:
        center = (t_verts[i] + t_verts[j] + t_verts[k]) / 3
        center /= np.linalg.norm(center)
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    cand_tetra_sub = np.vstack([t_verts, edge_mids, face_centers])
    candidates.append((cand_tetra_sub, [4, 6, 4]))

    # 6. Cube (8) + Octahedron (6)? Total 14.
    # Cube verts: (±1, ±1, ±1) normalized
    cube_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ])
    cube_verts /= np.linalg.norm(cube_verts, axis=1, keepdims=True)
    
    # Octa verts: (±1, 0, 0), (0, ±1, 0), (0, 0, ±1)
    octa_verts = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    octa_verts /= np.linalg.norm(octa_verts, axis=1, keepdims=True)
    
    cand_cube_octa = np.vstack([cube_verts, octa_verts])
    candidates.append((cand_cube_octa, [8, 6]))

    # 7. Random symmetric perturbations of Icosahedron+Poles
    # Add a few random starts to catch non-symmetric local optima if the structure is slightly broken
    np.random.seed(42)
    for i in range(5):
        perturbed = cand_ico_poles.copy()
        perturbed += np.random.normal(0, 0.1, size=perturbed.shape)
        # Normalize to unit sphere to keep symmetry-ish
        norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
        perturbed /= norms
        candidates.append((perturbed, [12, 1, 1]))

    # Evaluate and Refine Candidates
    for base_pts, group_sizes in candidates:
        if time.time() > deadline - 10:
            break
            
        refined_pts, score = refine_symmetric(base_pts, group_sizes, deadline)
        
        if score > best_score_global:
            best_score_global = score
            best_pts_global = refined_pts

    # Fallback: If no candidate worked well, use the raw Icosahedron+Poles
    if best_pts_global is None or best_score_global < 0.5:
        best_pts_global = cand_ico_poles
        best_score_global = get_score(cand_ico_poles)

    # Final Processing
    points = best_pts_global.copy()
    
    # Center of mass
    points -= np.mean(points, axis=0)
    
    # Normalize scale
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points / d_max

    # Ensure finite
    if not np.all(np.isfinite(points)):
        points = cand_ico_poles
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max

    return points

# EVOLVE-BLOCK-END
