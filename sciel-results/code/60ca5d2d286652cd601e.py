import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _get_face_normals(angle_deg):
    """Return unit normals for the 6 faces of a hexagon at given angle."""
    angle_rad = np.radians(angle_deg)
    normals = []
    for k in range(6):
        a = angle_rad + np.pi/6 + k * np.pi / 3
        normals.append([np.cos(a), np.sin(a)])
    return np.array(normals)

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """
    Calculate the minimum side length required to contain all inner hexagons.
    """
    cx, cy = outer_center
    face_normals = _get_face_normals(outer_angle)
    
    max_proj = 0.0
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        iverts_rel = iverts - np.array(outer_center)
        
        for n in face_normals:
            projs = iverts_rel @ n
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
                
    return 2.0 * max_proj / np.sqrt(3)

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """
    Strictly check if the configuration is valid.
    """
    cx, cy = outer_center
    outer_normals = _get_face_normals(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2.0
    
    all_verts = []
    for i in range(N):
        ix, iy, iangle = inner_positions[i]
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        all_verts.append(verts)
        
    # 1. Check Containment
    for i in range(N):
        verts = all_verts[i]
        verts_rel = verts - np.array(outer_center)
        for n in outer_normals:
            projs = verts_rel @ n
            if np.max(projs) > inradius + tol:
                return False
                
    # 2. Check Disjointness (SAT)
    for i in range(N):
        for j in range(i + 1, N):
            v1 = all_verts[i]
            v2 = all_verts[j]
            
            n1 = _get_face_normals(inner_positions[i][2])
            n2 = _get_face_normals(inner_positions[j][2])
            
            disjoint = False
            for n in np.vstack([n1, n2]):
                proj1 = v1 @ n
                proj2 = v2 @ n
                
                if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                    disjoint = True
                    break
            
            if not disjoint:
                return False
                
    return True

def _force_directed_relaxation(initial_positions, outer_side, outer_center, outer_angle, max_iters=500, dt=0.01):
    """
    Simulate forces to push hexagons apart and into the container.
    """
    pos = np.array(initial_positions, dtype=float)
    vel = np.zeros_like(pos)
    
    k_repel = 50.0
    k_contain = 100.0
    damping = 0.9
    
    cx, cy = outer_center
    outer_normals = _get_face_normals(outer_angle)
    inradius = outer_side * np.sqrt(3) / 2.0
    
    for step in range(max_iters):
        if _remaining_time() < 5.0:
            return None
            
        forces = np.zeros_like(pos)
        
        # 1. Containment Forces
        for i in range(N):
            x, y, a = pos[i]
            verts = _hex_vertices(x, y, 1.0, a)
            verts_rel = verts - np.array(outer_center)
            
            push_x = 0.0
            push_y = 0.0
            
            for k in range(6):
                n = outer_normals[k]
                projs = verts_rel @ n
                max_proj = np.max(projs)
                
                if max_proj > inradius:
                    penetration = max_proj - inradius
                    push_x -= k_contain * penetration * n[0]
                    push_y -= k_contain * penetration * n[1]
            
            forces[i, 0] += push_x
            forces[i, 1] += push_y
            
        # 2. Repulsion Forces
        for i in range(N):
            for j in range(i + 1, N):
                diff = pos[i, :2] - pos[j, :2]
                d = np.linalg.norm(diff)
                
                # Minimum distance for side-1 hexagons is roughly 2*sqrt(3) ~ 3.464
                # But they can interlock. Use a soft potential.
                # If d < 3.0, strong repulsion.
                if d < 1e-6:
                    d = 1e-6
                    diff = np.array([0.01, 0.01])
                
                if d < 3.5:
                    # Force magnitude increases as they get closer
                    # Simple linear repulsion
                    f_mag = k_repel * (3.5 - d)
                    f_dir = diff / d
                    forces[i, 0] += f_mag * f_dir[0]
                    forces[i, 1] += f_mag * f_dir[1]
                    forces[j, 0] -= f_mag * f_dir[0]
                    forces[j, 1] -= f_mag * f_dir[1]
        
        vel = vel * damping + forces * dt
        pos = pos + vel
        
        # Keep angles fixed for simplicity in this relaxation step
        # pos[:, 2] = initial_positions[:, 2]
        
        # Check if valid
        if step % 50 == 0:
            if _check_validity(pos, (cx, cy), outer_side, outer_angle):
                return pos
        
    if _check_validity(pos, (cx, cy), outer_side, outer_angle):
        return pos
    return None

def construct():
    start_time = time.time()
    
    sqrt3 = np.sqrt(3)
    
    # Initial configuration: Hex grid
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2),
    ]
    
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append((x, y))
    
    centers = np.array(centers)
    initial_positions = np.column_stack([centers, np.zeros(N)])
    
    # Compute initial bounding box
    all_verts = []
    for cx, cy in centers:
        verts = _hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    cx_mean = all_verts[:, 0].mean()
    cy_mean = all_verts[:, 1].mean()
    
    best_side = float('inf')
    best_center = (cx_mean, cy_mean)
    best_angle = 0.0
    best_positions = initial_positions
    
    # Coarse search over angles
    for outer_angle in np.linspace(0, 60, 121):
        a = np.radians(outer_angle)
        normals = []
        for i in range(6):
            theta = a + np.pi/6 + i * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        dx = all_verts[:, 0] - cx_mean
        dy = all_verts[:, 1] - cy_mean
        
        projections = np.zeros((N * 6, 6))
        for i in range(6):
            projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
        
        max_proj = projections.max(axis=0)
        min_proj = projections.min(axis=0)
        required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
        side = required_s.max()
        
        if side < best_side:
            best_side = side
            best_center = (cx_mean, cy_mean)
            best_angle = outer_angle
    
    # Fine search around best angle
    if best_side < float('inf'):
        for outer_angle in np.linspace(best_angle - 1.0, best_angle + 1.0, 201):
            a = np.radians(outer_angle)
            normals = []
            for i in range(6):
                theta = a + np.pi/6 + i * np.pi / 3
                normals.append([np.cos(theta), np.sin(theta)])
            normals = np.array(normals)
            
            dx = all_verts[:, 0] - cx_mean
            dy = all_verts[:, 1] - cy_mean
            
            projections = np.zeros((N * 6, 6))
            for i in range(6):
                projections[:, i] = dx * normals[i, 0] + dy * normals[i, 1]
            
            max_proj = projections.max(axis=0)
            min_proj = projections.min(axis=0)
            required_s = np.maximum(max_proj, -min_proj) / (sqrt3 / 2)
            side = required_s.max()
            
            if side < best_side:
                best_side = side
                best_center = (cx_mean, cy_mean)
                best_angle = outer_angle
    
    # Binary search on outer side length with force-directed relaxation
    # Start with current best_side and try to shrink
    low = best_side * 0.95
    high = best_side
    
    # Try to find a smaller valid configuration
    # We will iterate a few times
    for _ in range(5):
        if _remaining_time() < 10.0:
            break
            
        mid = (low + high) / 2
        
        # Try to relax into a container of size 'mid'
        relaxed = _force_directed_relaxation(initial_positions, mid, best_center, best_angle, max_iters=300)
        
        if relaxed is not None:
            # Valid configuration found at 'mid'
            high = mid
            # Update best if this is smaller
            if mid < best_side:
                best_side = mid
                best_positions = relaxed
            
            # Recompute optimal center and angle for this configuration?
            # For now, keep center/angle fixed to save time, 
            # but we could refine them.
            
        else:
            low = mid
    
    # Final optimization: Use L-BFGS-B to minimize the outer side length
    # given the current best_positions
    if _remaining_time() > 20.0:
        def objective(params):
            # params: [x1, y1, a1, x2, y2, a2, ..., cx, cy, outer_angle]
            # But let's fix outer_angle and center for now to simplify
            # Or optimize center and angle too?
            # Let's optimize positions and angles, and compute the min enclosing side.
            
            if len(params) != N * 3:
                return 1e10
                
            positions = params.reshape(N, 3)
            
            # Check validity? No, objective should be smooth.
            # Use penalty for overlap.
            
            # 1. Compute min enclosing side
            # We need to optimize center too.
            # Let's just compute the side length assuming center is mean of vertices.
            
            all_verts = []
            for i in range(N):
                ix, iy, ia = positions[i]
                verts = _hex_vertices(ix, iy, 1.0, ia)
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            
            cx = all_verts[:, 0].mean()
            cy = all_verts[:, 1].mean()
            
            # Try a few angles for the outer hexagon to find the min side
            # This makes the objective non-smooth.
            # Better: fix outer angle to best_angle.
            
            side = _min_enclosing_side(positions, (cx, cy), best_angle)
            
            # 2. Penalty for overlap
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    v1 = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, positions[i, 2])
                    v2 = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, positions[j, 2])
                    
                    # Simple distance penalty
                    d = np.linalg.norm(positions[i, :2] - positions[j, :2])
                    if d < 3.0:
                        penalty += (3.0 - d) ** 2
            
            return side + 10.0 * penalty
        
        # Initial guess: best_positions
        x0 = best_positions.flatten()
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 200})
            
            if res.success or res.fun < best_side:
                new_positions = res.x.reshape(N, 3)
                
                # Verify validity
                # Compute optimal center and angle
                all_verts = []
                for i in range(N):
                    ix, iy, ia = new_positions[i]
                    verts = _hex_vertices(ix, iy, 1.0, ia)
                    all_verts.append(verts)
                all_verts = np.vstack(all_verts)
                
                cx = all_verts[:, 0].mean()
                cy = all_verts[:, 1].mean()
                
                # Find best angle
                best_s = float('inf')
                best_a = best_angle
                for outer_angle in np.linspace(best_angle - 2, best_angle + 2, 41):
                    s = _min_enclosing_side(new_positions, (cx, cy), outer_angle)
                    if s < best_s:
                        best_s = s
                        best_a = outer_angle
                
                if best_s < best_side:
                    # Check if valid (disjoint)
                    if _check_validity(new_positions, (cx, cy), best_s, best_a, tol=1e-4):
                        best_side = best_s
                        best_center = (cx, cy)
                        best_angle = best_a
                        best_positions = new_positions
                        
        except Exception:
            pass
    
    # Final verification and safety margin
    if best_side < float('inf'):
        # Add small safety margin
        best_side *= 1.001
        
        # Verify containment
        inner = [[float(cx), float(cy), float(a)] for cx, cy, a in best_positions]
        
        # Check if all vertices are inside
        all_check_verts = []
        for cx, cy, a in best_positions:
            all_check_verts.append(_hex_vertices(cx, cy, 1.0, a))
        all_check_verts = np.vstack(all_check_verts)
        
        a = np.radians(best_angle)
        normals = []
        for i in range(6):
            theta = a + np.pi/6 + i * np.pi / 3
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)
        
        dx = all_check_verts[:, 0] - best_center[0]
        dy = all_check_verts[:, 1] - best_center[1]
        
        max_violation = 0
        for i in range(6):
            proj = dx * normals[i, 0] + dy * normals[i, 1]
            max_violation = max(max_violation, proj.max() - best_side * sqrt3 / 2)
            max_violation = max(max_violation, -proj.min() - best_side * sqrt3 / 2)
            
        if max_violation > 1e-9:
            best_side += max_violation / (sqrt3 / 2) * 1.001
        
        return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)
    else:
        # Fallback
        inner = [[float(cx), float(cy), float(0.0)] for cx, cy in centers]
        return inner, (float(cx_mean), float(cy_mean)), float(best_side), float(best_angle)

BET: {mechanism: "Combining the robust hex-grid initialization with force-directed relaxation and L-BFGS-B optimization allows the program to escape the local minimum of the rigid grid and find a tighter, non-uniform packing, potentially reducing the outer side length significantly.", q: 0.8, delta: [0.5, 2.0]}
