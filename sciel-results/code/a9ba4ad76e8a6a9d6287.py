import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0
    # Vertices of a unit hexagon with side 1, centered at origin, flat-top (angle 0)
    # Angles: 0, 60, 120, 180, 240, 300
    unit_hex_flat = np.array([
        [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)
    ])
    
    # Vertices for pointy-top (angle 30)
    unit_hex_pointy = np.array([
        [np.cos(np.radians(a + 30)), np.sin(np.radians(a + 30))] for a in range(0, 360, 60)
    ])

    def get_hex_verts(pos, angle, unit_hex):
        # Rotate unit hex
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        verts = unit_hex @ R.T + pos
        return verts

    def get_normals(verts):
        # verts is (6, 2)
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth(v1, n1, v2, n2):
        # v1, v2: (6,2)
        # n1, n2: (6,2)
        # Check all normals from both polygons
        # For each normal, project both polygons
        # Find the minimum overlap
        
        # Efficient check:
        # p1 = v1 @ n.T
        # min1, max1
        # p2 = v2 @ n.T
        # min2, max2
        # overlap = min(max1, max2) - max(min1, min2)
        # if min(overlap) <= 0, no penetration for this axis
        # The penetration depth is the minimum positive overlap? No, it's the minimum overlap across all separating axes?
        # Actually, for two convex polygons, if they are separated, there exists an axis with no overlap.
        # If they are intersecting, the minimum translation to separate them is the minimum overlap depth across all edge normals.
        
        all_normals = np.vstack([n1, n2])
        p1 = v1 @ all_normals.T
        p2 = v2 @ all_normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        if np.min(overlaps) <= 1e-9:
            return 0.0
        
        return np.min(overlaps)

    def outer_violation(inner_verts, S, outer_angle):
        # Outer hexagon centered at (0,0), side S, angle outer_angle
        # We can just check against the 6 half-planes defined by the outer hexagon
        # Outer vertices
        outer_verts = get_hex_verts(np.array([0.0, 0.0]), outer_angle, unit_hex_flat) * S
        outer_normals = get_normals(outer_verts)
        
        # Project inner vertices onto outer normals
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        
        # Violation is how much inner exceeds outer
        viol = np.max(p_inner - p_outer_max, axis=0)
        if viol.max() <= 0:
            return 0.0
        return viol.max()

    def objective(params, outer_angle=0.0):
        if time.time() > deadline:
            return 1e10

        # Unpack params
        # params: [x1, y1, th1, ..., x12, y12, th12, S]
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        S = params[3*N]
        
        if S < 2.0:
            return 1e6

        total_penalty = 0.0
        inner_verts_list = []
        inner_normals_list = []

        # Compute vertices and normals for all inner hexagons
        for i in range(N):
            verts = get_hex_verts(positions[i], angles[i], unit_hex_flat)
            inner_verts_list.append(verts)
            inner_normals_list.append(get_normals(verts))

        # Check outer violation
        for i in range(N):
            v = outer_violation(inner_verts_list[i], S, outer_angle)
            total_penalty += v * 1000.0

        # Check inner-inner violations
        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_normals_list[i], 
                                       inner_verts_list[j], inner_normals_list[j])
                total_penalty += pd * 1000.0

        return S + total_penalty

    def make_honeycomb_init(rows, cols, flat_top):
        pos = []
        ang = []
        if flat_top:
            dx = SQRT3
            dy = 1.5
            shift = SQRT3 / 2
            for r in range(rows):
                for c in range(cols):
                    x = c * dx
                    if r % 2 == 1:
                        x += shift
                    y = r * dy
                    pos.append([x, y])
                    ang.append(0.0) # Flat top
        else:
            dx = 1.5
            dy = SQRT3
            shift = 0.75
            for r in range(rows):
                for c in range(cols):
                    y = c * dy
                    if r % 2 == 1:
                        y += shift
                    x = r * dx
                    pos.append([x, y])
                    ang.append(np.pi/6) # Pointy top

        pos = np.array(pos)
        center = pos.mean(axis=0)
        pos = pos - center
        return pos, ang

    inits = []
    # 3x4 Flat-Top
    pos_34, ang_34 = make_honeycomb_init(3, 4, True)
    inits.append((pos_34, ang_34, 4.5))
    
    # 4x3 Pointy-Top
    pos_43, ang_43 = make_honeycomb_init(4, 3, False)
    inits.append((pos_43, ang_43, 4.5))
    
    # Circular
    pos_circ = np.array([
        [3.0 * np.cos(np.radians(a)), 3.0 * np.sin(np.radians(a))] for a in range(0, 360, 30)
    ])
    ang_circ = np.zeros(N)
    inits.append((pos_circ, ang_circ, 4.5))

    best_s = np.inf
    best_params = None

    for pos_init, ang_init, s_init in inits:
        if time.time() > deadline - 10:
            break
            
        x0 = np.concatenate([pos_init.flatten(), ang_init, [s_init]])
        
        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            s_val = res.x[3*N]
            # Verify validity roughly
            # If penalty is 0, it's valid
            if res.fun < s_val + 1e-4: # Penalty is small
                if s_val < best_s:
                    best_s = s_val
                    best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback
        best_s = 5.0
        best_params = np.concatenate([inits[0][0].flatten(), inits[0][1], [best_s]])

    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:3*N]
    final_s = best_params[3*N]

    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(final_ang[i])] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
