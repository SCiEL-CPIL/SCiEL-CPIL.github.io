import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Precompute unit hexagon vertices (centered at origin, side 1)
    # Pointy-top: vertices at 30, 90, 150, 210, 270, 330 degrees?
    # Flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees.
    # The orientation doesn't matter for the inner hexagons as they can rotate.
    # Let's use flat-top for simplicity in standard coordinates.
    angles = np.array([i * math.pi / 3 for i in range(6)])
    verts_rel = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals for SAT (perpendicular to edges)
    # Edge 0 is between v0 and v1. Normal points outwards.
    # For a regular hexagon, normals are at angles 30, 90, 150, 210, 270, 330 degrees (if flat top)
    # Or 0, 60, 120... (if pointy top)
    # Let's just compute normals from edges.
    edges = np.roll(verts_rel, -1, axis=0) - verts_rel
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # The 6 supporting lines of the outer hexagon with rotation alpha
    # Normals are at alpha + k*pi/3
    
    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        return verts_rel @ rot.T + np.array([cx, cy])

    def sat_overlap(v1, v2):
        """
        Calculate maximum overlap depth between two convex polygons v1, v2.
        v1, v2 are (6, 2) arrays.
        Returns positive value if overlapping, negative if separated.
        """
        max_overlap = 0.0
        # Check axes from v1
        for i in range(6):
            n = normals[i]
            p1 = np.dot(v1, n)
            p2 = np.dot(v2, n)
            overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            max_overlap = max(max_overlap, overlap)
            if max_overlap < 0: # Early exit if separated on this axis? No, need to check all to be sure of max overlap?
                # Actually, if they are separated on ANY axis, they don't overlap.
                # But we want the MAXIMUM overlap depth for penalty.
                # If they are separated, overlap is negative. We only care if max_overlap > 0.
                # If we find an axis where they are separated, the true overlap is 0 (or negative).
                # So if we find a separating axis, we can return 0 (or negative) immediately?
                # No, we want the *penalty* which is 0 if no overlap.
                # So if overlap < 0 on any axis, they are disjoint. Return 0.
                pass 
        # Check axes from v2 (same set of normals for regular hexagons? No, relative rotation matters)
        # Wait, the normals array `normals` is for the *unit* hexagon in local coords.
        # The actual normals for v1 and v2 depend on their rotations.
        # My `get_hex_verts` rotates the vertices. The normals must be rotated too.
        
        # Let's re-implement sat_overlap to take vertices and compute normals on the fly or pass rotated normals.
        # Better: pass vertices and compute axes from edges of v1 and v2.
        
        axes = []
        for i in range(6):
            e1 = v1[(i+1)%6] - v1[i]
            n1 = np.array([-e1[1], e1[0]])
            axes.append(n1 / np.linalg.norm(n1))
            
            e2 = v2[(i+1)%6] - v2[i]
            n2 = np.array([-e2[1], e2[0]])
            axes.append(n2 / np.linalg.norm(n2))
            
        max_overlap = 0.0
        for n in axes:
            p1 = np.dot(v1, n)
            p2 = np.dot(v2, n)
            overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
            if overlap < 0:
                return 0.0 # Disjoint
            max_overlap = max(max_overlap, overlap)
            
        return max_overlap

    def compute_S(params, outer_alpha):
        """
        params: (x1, y1, a1, x2, y2, a2, ..., x11, y11, a11)
        outer_alpha: rotation of the enclosing hexagon
        Returns minimal side length S.
        """
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # Collect all vertices
        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)
        
        # Center points for numerical stability
        centroid = np.mean(all_pts, axis=0)
        centered_pts = all_pts - centroid
        
        # Outer hexagon normals
        k_angles = outer_alpha + np.arange(6) * math.pi / 3
        normals_outer = np.column_stack([np.cos(k_angles), np.sin(k_angles)])
        
        # Projections
        dots = centered_pts @ normals_outer.T # (N_pts, 6)
        extents = np.max(dots, axis=0) - np.min(dots, axis=0)
        max_extent = np.max(extents)
        
        # Side length S = max_extent / sqrt(3)
        return max_extent / math.sqrt(3)

    def objective(params):
        if time.time() > deadline - 10.0:
            return 1e10
            
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]
        
        # Calculate overlaps
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                overlap = sat_overlap(verts_list[i], verts_list[j])
                if overlap > 0:
                    penalty += overlap * overlap * 100.0 # Strong penalty
        
        # Compute S with outer angle fixed to 0 for now? 
        # Or optimize outer angle as part of the vector?
        # Let's fix outer angle to 0 during the main optimization to save 1 dimension and complexity.
        # The optimal outer angle for a given packing can be found in 1D afterwards.
        S = compute_S(params, 0.0)
        
        return S + penalty

    best_S = float('inf')
    best_params = None
    
    # Initial configurations
    # 1. Lattice-like
    # 2. Random
    
    # Lattice centers
    lattice_centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = 2 * D * math.cos(angle)
        y = 2 * D * math.sin(angle)
        lattice_centers.append((x, y))
        
    # Add 4 more from next ring
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                spacing = 2 * D
                x = spacing * (q + r / 2.0)
                y = spacing * (D / 2.0) * r
                ring2_centers.append((x, y))
                
    # Pick 4 symmetric ring2 centers
    # Indices 0, 3, 6, 9
    init_lattice = lattice_centers + [ring2_centers[i] for i in [0, 3, 6, 9]]
    
    init_configs = []
    
    # Config 1: Lattice
    p = np.zeros(N * 3)
    for i, (cx, cy) in enumerate(init_lattice):
        p[3*i] = cx
        p[3*i+1] = cy
        p[3*i+2] = 0.0
    init_configs.append(p)
    
    # Config 2: Lattice rotated 30 deg
    p_rot = np.zeros(N * 3)
    for i, (cx, cy) in enumerate(init_lattice):
        p_rot[3*i] = cx
        p_rot[3*i+1] = cy
        p_rot[3*i+2] = math.pi / 6
    init_configs.append(p_rot)
    
    # Configs 3-10: Random
    rng = np.random.default_rng(42)
    for _ in range(8):
        p_rand = np.zeros(N * 3)
        # Scatter in a box of size ~ 15x15
        p_rand[0::3] = rng.uniform(-10, 10, N)
        p_rand[1::3] = rng.uniform(-10, 10, N)
        p_rand[2::3] = rng.uniform(0, 2*math.pi, N)
        init_configs.append(p_rand)
        
    # Optimization
    for init_p in init_configs:
        if time.time() > deadline - 20.0:
            break
            
        try:
            # Use Nelder-Mead with adaptive parameters
            res = minimize(objective, init_p, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
                
        except Exception:
            continue

    # If no valid result, fallback to lattice
    if best_params is None:
        best_params = init_configs[0]
        best_S = objective(best_params)
        
    # Refine outer hexagon angle
    if time.time() < deadline - 10.0:
        def outer_obj(alpha):
            return compute_S(best_params, alpha)
            
        try:
            res_outer = minimize_scalar(outer_obj, bounds=(0, math.pi / 3), method='bounded',
                                        options={'xatol': 1e-10})
            if res_outer.success:
                best_outer_alpha = res_outer.x
                best_S = res_outer.fun
            else:
                best_outer_alpha = 0.0
        except Exception:
            best_outer_alpha = 0.0
    else:
        best_outer_alpha = 0.0

    # Construct output
    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        angle_deg = math.degrees(best_params[3 * i + 2])
        inner.append([x, y, angle_deg])
        
    # Center the outer hexagon at (0,0) for the return value?
    # The problem asks for outer center, side, angle.
    # The score is the side length.
    # The inner coordinates are relative to the outer center?
    # "inner rows (x, y, angle_degrees) of n unit hexagons; outer hexagon center (x, y), side length and rotation"
    # Usually, the evaluator places the outer hexagon at the given center, and the inner hexagons at the given coords.
    # If we return center (0,0), the inner coords should be relative to (0,0).
    # Our optimization assumed outer center at (0,0) and inner coords absolute.
    # So returning center (0,0) is consistent.
    
    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_alpha)
