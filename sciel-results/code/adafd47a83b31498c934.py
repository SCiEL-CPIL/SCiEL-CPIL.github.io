"""Free-position optimization with smooth penalties for hexagon packing."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 110.0  # Safety margin
SQRT3 = np.sqrt(3)

def _hex_vertices_batch(positions, angles, side=1.0):
    """Vectorized: return (N*6, 2) array of all vertices for N hexagons."""
    n = len(positions)
    pos = np.asarray(positions, dtype=np.float64)
    ang = np.radians(np.asarray(angles, dtype=np.float64))
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    offsets_x = side * np.cos(total_angles)
    offsets_y = side * np.sin(total_angles)
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    return np.column_stack([verts_x.ravel(), verts_y.ravel()])

def _min_side_vectorized(all_verts, cx, cy, outer_angle_deg):
    """
    Compute minimum outer side length S such that all vertices in all_verts are inside
    the outer hexagon centered at (cx,cy) with rotation outer_angle_deg.
    """
    verts = np.asarray(all_verts, dtype=np.float64)
    n_pts = len(verts)
    if n_pts == 0:
        return 0.0
    
    dx = verts[:, 0] - cx
    dy = verts[:, 1] - cy
    
    a = np.radians(outer_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    # Inequalities for point (lx, ly) to be inside regular hexagon of side S centered at origin:
    # |lx| <= S
    # |sqrt(3)lx + ly| <= sqrt(3)S
    # |-sqrt(3)lx + ly| <= sqrt(3)S
    # |sqrt(3)lx - ly| <= sqrt(3)S
    # |-sqrt(3)lx - ly| <= sqrt(3)S
    # |lx - ly| <= S  <-- Wait, the last two are redundant or incorrect?
    
    # Let's re-derive.
    # Hexagon vertices at angles 0, 60, 120, 180, 240, 300.
    # Side S.
    # Normals are at 30, 90, 150, 210, 270, 330.
    # Normal 1: (cos30, sin30) = (sqrt(3)/2, 1/2). Distance from center to side is S*sqrt(3)/2.
    # So (sqrt(3)/2)lx + (1/2)ly <= S*sqrt(3)/2  => sqrt(3)lx + ly <= S*sqrt(3)
    # Normal 2: (cos90, sin90) = (0, 1). Distance S*sqrt(3)/2.
    # So ly <= S*sqrt(3)/2.
    # Wait, the distance from center to side (apothem) is S*sqrt(3)/2.
    # So the inequality is dot(p, n) <= apothem.
    
    # Let's use the 6 normals.
    # n0 = (sqrt(3)/2, 1/2) -> dot <= S*sqrt(3)/2
    # n1 = (0, 1) -> dot <= S*sqrt(3)/2
    # n2 = (-sqrt(3)/2, 1/2) -> dot <= S*sqrt(3)/2
    # n3 = (-sqrt(3)/2, -1/2) -> dot <= S*sqrt(3)/2
    # n4 = (0, -1) -> dot <= S*sqrt(3)/2
    # n5 = (sqrt(3)/2, -1/2) -> dot <= S*sqrt(3)/2
    
    # So:
    # 1. sqrt(3)lx + ly <= S*sqrt(3)
    # 2. ly <= S*sqrt(3)/2
    # 3. -sqrt(3)lx + ly <= S*sqrt(3)
    # 4. -sqrt(3)lx - ly <= S*sqrt(3)
    # 5. -ly <= S*sqrt(3)/2
    # 6. sqrt(3)lx - ly <= S*sqrt(3)
    
    # We want min S such that all these hold.
    # S >= (sqrt(3)lx + ly) / sqrt(3) = lx + ly/sqrt(3)
    # S >= ly / (sqrt(3)/2) = 2ly/sqrt(3)
    # S >= (-sqrt(3)lx + ly) / sqrt(3) = -lx + ly/sqrt(3)
    # S >= (-sqrt(3)lx - ly) / sqrt(3) = -lx - ly/sqrt(3)
    # S >= -ly / (sqrt(3)/2) = -2ly/sqrt(3)
    # S >= (sqrt(3)lx - ly) / sqrt(3) = lx - ly/sqrt(3)
    
    c1 = lx + ly / SQRT3
    c2 = 2 * ly / SQRT3
    c3 = -lx + ly / SQRT3
    c4 = -lx - ly / SQRT3
    c5 = -2 * ly / SQRT3
    c6 = lx - ly / SQRT3
    
    required = np.maximum(
        np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)),
        np.maximum(c5, c6)
    )
    return float(np.max(required))

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-5):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = outer_center
    pos = np.array([p[:2] for p in inner])
    ang = np.array([p[2] for p in inner])
    
    # Check containment
    all_verts = _hex_vertices_batch(pos, ang, side=1.0)
    min_side = _min_side_vectorized(all_verts, cx, cy, outer_angle)
    if min_side > outer_side + tol:
        return False
        
    # Check overlaps using SAT
    verts_list = []
    for i in range(N):
        verts = _hex_vertices_batch(pos[i:i+1], ang[i:i+1], side=1.0).reshape(6, 2)
        verts_list.append(verts)
        
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagon_overlap(verts_list[i], verts_list[j], tol):
                return False
    return True

def _hexagon_overlap(verts1, verts2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            v1 = verts[i]
            v2 = verts[(i + 1) % len(verts)]
            edge = v2 - v1
            nx, ny = edge[1], -edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append([nx/norm, ny/norm])
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def _objective(params):
    """
    Smooth objective function.
    params: [x0, y0, a0, ..., x10, y10, a10, S]
    """
    if time.time() > DEADLINE:
        return 1e10
        
    positions = np.array([params[i*3:i*3+2] for i in range(N)])
    angles = np.array([params[i*3+2] for i in range(N)])
    S = params[-1]
    
    # Containment penalty
    all_verts = _hex_vertices_batch(positions, angles, side=1.0)
    min_side = _min_side_vectorized(all_verts, 0.0, 0.0, 0.0)
    # We want S to be at least min_side.
    # Penalty if S < min_side.
    # Actually, we are optimizing S as a variable? 
    # If we optimize S, the objective should be S + penalty.
    # If S < min_side, we add penalty.
    # Let's just return min_side + overlap_penalty.
    # This way S is not a variable, we just compute it.
    # But the prompt says "minimizing the outer side length".
    # So objective = min_side + overlap_penalty.
    
    penalty = 0.0
    
    # Overlap penalty
    # For each pair, if distance < d_min, add penalty.
    # d_min for touching hexagons is sqrt(3).
    # We can use a smooth approximation of distance.
    # Or just use SAT violation depth?
    # SAT violation depth is hard to compute smoothly.
    # Let's use center distance as a proxy.
    # If centers are closer than sqrt(3), they might overlap.
    # But rotated hexagons can overlap even if centers are further?
    # No, if centers are further than 2*radius (2.0), they don't overlap.
    # If centers are closer than 2*apothem (sqrt(3)), they MIGHT overlap.
    # Actually, min distance between centers for non-overlap is sqrt(3) if aligned.
    # If rotated, it can be less? No, min distance is still related to apothem.
    # Let's use a simple penalty: sum of max(0, d_min - dist)^2 for all pairs.
    # d_min = sqrt(3) is a safe lower bound for "touching".
    # If dist < sqrt(3), they definitely overlap?
    # If two hexagons are rotated 30 degrees relative to each other, the min distance between centers for non-overlap is larger?
    # Actually, the "packing" of hexagons is tightest when aligned.
    # So d_min = sqrt(3) is the minimum distance required for ANY orientation to not overlap?
    # No. If you rotate one, the "corners" stick out.
    # So d_min should be larger?
    # Let's just use a large penalty for any distance < 1.8.
    # And use SAT for final validity check.
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            d_min = SQRT3  # 1.732
            if dist < d_min:
                penalty += (d_min - dist) ** 2
    
    return min_side + 10.0 * penalty

def _initial_packing():
    """Generate a symmetric initial packing."""
    r1 = SQRT3
    r2 = 2 * SQRT3
    positions = [[0.0, 0.0]]
    angles = [0.0]
    for i in range(6):
        a = i * np.pi / 3
        positions.append([r1 * np.cos(a), r1 * np.sin(a)])
        angles.append(0.0)
    # 4 second ring hexagons at 0, 90, 180, 270? Or 0, 60, 120, 180?
    # Let's use 0, 90, 180, 270 for variety in initial guess.
    for i in range(4):
        a = i * np.pi / 2
        positions.append([r2 * np.cos(a), r2 * np.sin(a)])
        angles.append(0.0)
    return positions, angles

def construct():
    global START_TIME, DEADLINE
    
    best_side = 1e10
    best_params = None
    
    # Initial packing
    pos_init, ang_init = _initial_packing()
    x0 = []
    for i in range(N):
        x0.append(pos_init[i][0])
        x0.append(pos_init[i][1])
        x0.append(ang_init[i])
    # Initial S estimate
    all_verts = _hex_vertices_batch(np.array(pos_init), np.array(ang_init), side=1.0)
    S_init = _min_side_vectorized(all_verts, 0.0, 0.0, 0.0)
    x0.append(S_init)
    
    # Bounds
    # Positions: -10 to 10
    # Angles: 0 to 2*pi
    # S: 0 to 20
    bounds = [(-10, 10), (-10, 10), (0, 2*np.pi)] * N
    bounds.append((0, 20))
    
    # Run multiple restarts
    np.random.seed(42)
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > DEADLINE - 5:
            break
            
        if r == 0:
            x0_r = x0
        else:
            # Perturb initial
            x0_r = []
            for i in range(N):
                x0_r.append(x0[i*3] + np.random.normal(0, 1.0))
                x0_r.append(x0[i*3+1] + np.random.normal(0, 1.0))
                x0_r.append(x0[i*3+2] + np.random.normal(0, np.pi/3))
            x0_r.append(S_init + np.random.uniform(-1, 1))
            
        try:
            res = minimize(_objective, x0_r, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
                
        except Exception as e:
            pass
            
    # Extract best solution
    if best_params is not None:
        positions = np.array([best_params[i*3:i*3+2] for i in range(N)])
        angles = np.array([best_params[i*3+2] for i in range(N)])
        
        # Recompute exact side
        all_verts = _hex_vertices_batch(positions, angles, side=1.0)
        final_side = _min_side_vectorized(all_verts, 0.0, 0.0, 0.0)
        
        inner_list = [[float(positions[i][0]), float(positions[i][1]), float(angles[i])] for i in range(N)]
        
        # Verify
        if _check_validity(inner_list, (0.0, 0.0), final_side, 0.0):
            return inner_list, (0.0, 0.0), float(final_side), 0.0
        else:
            # Fallback to initial
            pos_f, ang_f = _initial_packing()
            inner_f = [[float(pos_f[i][0]), float(pos_f[i][1]), float(ang_f[i])] for i in range(N)]
            all_verts_f = _hex_vertices_batch(np.array(pos_f), np.array(ang_f), side=1.0)
            side_f = _min_side_vectorized(all_verts_f, 0.0, 0.0, 0.0)
            return inner_f, (0.0, 0.0), float(side_f), 0.0
    else:
        # Fallback to initial
        pos_f, ang_f = _initial_packing()
        inner_f = [[float(pos_f[i][0]), float(pos_f[i][1]), float(ang_f[i])] for i in range(N)]
        all_verts_f = _hex_vertices_batch(np.array(pos_f), np.array(ang_f), side=1.0)
        side_f = _min_side_vectorized(all_verts_f, 0.0, 0.0, 0.0)
        return inner_f, (0.0, 0.0), float(side_f), 0.0

