# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def calculate_metric(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_metric(pts):
        return -calculate_metric(pts)

    # Define symmetric shape generators
    # Each generator takes a parameter vector and returns an (14, 3) array
    
    def shape_cube_plus_face(params):
        # 8 cube vertices + 6 face centers
        # Scale the cube by s1, scale face centers by s2? 
        # Or just one scale? The ratio is scale invariant, so relative scale matters.
        # Let cube vertices be at (±1, ±1, ±1) * s1
        # Face centers at (±1, 0, 0) * s2, etc.
        # Actually, face centers of a cube with side 2 are at (±1,0,0).
        # Vertices are at (±1,±1,±1).
        # Distance between face center (1,0,0) and vertex (1,1,1) is sqrt(0+1+1) = sqrt(2).
        # Distance between face center (1,0,0) and opposite face center (-1,0,0) is 2.
        # Distance between vertex (1,1,1) and (-1,-1,-1) is sqrt(9)=3.
        # We can optimize s1 and s2.
        s1 = params[0]
        s2 = params[1]
        
        verts = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
        ]) * s1
        
        faces = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ]) * s2
        
        return np.vstack([verts, faces])

    def shape_ico_poles(params):
        # 12 icosahedron vertices + 2 poles
        # Icosahedron vertices (standard coords)
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        
        # Normalize ico_verts to unit sphere? No, let's scale them.
        # We have 2 parameters: s_ico (scale of ico part) and h (height of poles)
        # Also allow squashing z of ico part: k
        s_ico = params[0]
        h = params[1]
        k = params[2]
        
        ico_verts[:, 2] *= k
        ico_verts *= s_ico
        
        poles = np.array([
            [0, 0, h],
            [0, 0, -h]
        ])
        
        return np.vstack([ico_verts, poles])

    def shape_two_rings(params):
        # 7 points in top ring, 7 in bottom ring
        # Top ring: radius r1, height h, angles offset 0
        # Bottom ring: radius r2, height -h, angles offset pi/7
        r1 = params[0]
        r2 = params[1]
        h = params[2]
        
        angles_top = np.linspace(0, 2*np.pi, 7, endpoint=False)
        angles_bot = angles_top + np.pi/7
        
        top_pts = np.column_stack([
            r1 * np.cos(angles_top),
            r1 * np.sin(angles_top),
            np.full(7, h)
        ])
        
        bot_pts = np.column_stack([
            r2 * np.cos(angles_bot),
            r2 * np.sin(angles_bot),
            np.full(7, -h)
        ])
        
        return np.vstack([top_pts, bot_pts])

    def shape_dodeca_subset(params):
        # 20 dodecahedron vertices. Pick 14?
        # This is harder to parameterize simply.
        # Let's stick to the first 3 which are strong candidates.
        pass

    candidates = []
    
    # Optimize each family
    # Initial guesses for parameters
    # Family 1: Cube + Faces
    # s1 ~ 1, s2 ~ 1
    x0_1 = np.array([1.0, 1.0])
    res_1 = minimize(neg_metric, x0_1, args=(shape_cube_plus_face,), method='Nelder-Mead', 
                     options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    if res_1.success:
        pts_1 = shape_cube_plus_face(res_1.x)
        candidates.append((calculate_metric(pts_1), pts_1))
    
    # Family 2: Icosahedron + Poles
    # s_ico ~ 1, h ~ 1, k ~ 1
    x0_2 = np.array([1.0, 1.5, 1.0])
    res_2 = minimize(neg_metric, x0_2, args=(shape_ico_poles,), method='Nelder-Mead',
                     options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    if res_2.success:
        pts_2 = shape_ico_poles(res_2.x)
        candidates.append((calculate_metric(pts_2), pts_2))
        
    # Also try a few different initial guesses for Family 2
    x0_2b = np.array([1.0, 2.0, 0.5])
    res_2b = minimize(neg_metric, x0_2b, args=(shape_ico_poles,), method='Nelder-Mead',
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    if res_2b.success:
        pts_2b = shape_ico_poles(res_2b.x)
        candidates.append((calculate_metric(pts_2b), pts_2b))

    # Family 3: Two Rings
    # r1 ~ 1, r2 ~ 1, h ~ 1
    x0_3 = np.array([1.0, 1.0, 1.0])
    res_3 = minimize(neg_metric, x0_3, args=(shape_two_rings,), method='Nelder-Mead',
                     options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    if res_3.success:
        pts_3 = shape_two_rings(res_3.x)
        candidates.append((calculate_metric(pts_3), pts_3))

    # Add a generic random restart + local optimization as a fallback/improver
    # This is different from pure SA, it's "Random Init + Local Refine"
    best_generic_score = -np.inf
    best_generic_pts = None
    
    for seed in range(10):
        np.random.seed(seed)
        # Random points in unit ball
        pts = np.random.randn(14, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        pts = pts / norms
        
        # Refine with Nelder-Mead on flattened array
        # This is expensive (42 parameters), but Nelder-Mead might handle it if we are lucky or if we use a coarser metric.
        # Actually, 42D Nelder-Mead is very slow and unreliable.
        # Instead, let's do a few steps of gradient-free local search on the sphere.
        # Or just keep the deterministic families.
        
        # Let's try a "Repulsion" based local refinement using a simple iterative projection.
        # Move each point slightly away from its nearest neighbor.
        current_pts = pts.copy()
        for _ in range(50):
            for i in range(14):
                dists = np.linalg.norm(current_pts[i] - current_pts, axis=1)
                dists[i] = np.inf
                nearest_idx = np.argmin(dists)
                d_nn = dists[nearest_idx]
                if d_nn < 1e-6:
                    continue
                # Move away
                vec = current_pts[i] - current_pts[nearest_idx]
                vec /= np.linalg.norm(vec)
                # Step size proportional to 1/d^2
                step = 0.01 / (d_nn * d_nn + 1e-6)
                current_pts[i] += vec * step
                # Project to sphere
                norm_i = np.linalg.norm(current_pts[i])
                if norm_i > 1e-10:
                    current_pts[i] /= norm_i
        
        score = calculate_metric(current_pts)
        if score > best_generic_score:
            best_generic_score = score
            best_generic_pts = current_pts

    if best_generic_pts is not None:
        candidates.append((best_generic_score, best_generic_pts))

    # Select best candidate
    if not candidates:
        # Fallback
        np.random.seed(0)
        pts = np.random.randn(14, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-10)
        return pts

    best_score, best_pts = max(candidates, key=lambda x: x[0])
    
    # Final check
    if not np.all(np.isfinite(best_pts)):
        np.random.seed(0)
        pts = np.random.randn(14, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-10)
        return pts

    return best_pts

# EVOLVE-BLOCK-END
