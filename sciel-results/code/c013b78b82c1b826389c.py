"""Improved packing of 12 unit hexagons inside a regular hexagon using physics-based repulsion."""
import time
import numpy as np

N = 12
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])


def _point_in_hexagon(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    apothem = s * np.sqrt(3) / 2.0
    normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    dx = px - cx
    dy = py - cy
    for na in normal_angles:
        proj = dx * np.cos(na) + dy * np.sin(na)
        if proj > apothem:
            return False
    return True


def _hex_overlap_depth(h1, h2):
    """Compute minimum translation distance to separate two hexagons h1, h2.
    Each hex is (cx, cy, s, angle_deg). Returns 0 if no overlap, positive if overlapping."""
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    
    # Quick rejection: if centers are far apart, no overlap
    dist = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
    max_radius = s1 + s2
    if dist > 2 * max_radius:
        return 0.0
    
    v1 = _hex_verts(cx1, cy1, s1, a1)
    v2 = _hex_verts(cx2, cy2, s2, a2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            center = np.mean(verts, axis=0)
            if np.dot(normal, center - p1) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    if min_overlap == float('inf'):
        return 0.0
    
    return min_overlap  # Positive means overlap depth


def _containment_violation(inner, outer_center, outer_side, outer_angle):
    """Compute containment violation for inner hexagons. Returns list of (hex_idx, violation_depth)."""
    cx, cy = outer_center
    violations = []
    for idx, (ix, iy, ia) in enumerate(inner):
        verts = _hex_verts(ix, iy, 1.0, ia)
        max_viol = 0.0
        for v in verts:
            a = np.radians(outer_angle)
            apothem = outer_side * np.sqrt(3) / 2.0
            normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
            dx = v[0] - cx
            dy = v[1] - cy
            for na in normal_angles:
                proj = dx * np.cos(na) + dy * np.sin(na)
                if proj > apothem:
                    viol = proj - apothem
                    if viol > max_viol:
                        max_viol = viol
                    break
        if max_viol > 0:
            violations.append((idx, max_viol))
    return violations


def _disjointness_violations(inner):
    """Compute disjointness violations. Returns list of (i, j, overlap_depth)."""
    violations = []
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            h1 = (inner[i][0], inner[i][1], 1.0, inner[i][2])
            h2 = (inner[j][0], inner[j][1], 1.0, inner[j][2])
            depth = _hex_overlap_depth(h1, h2)
            if depth > 0:  # Overlapping
                violations.append((i, j, depth))
    return violations


def _physics_step(inner, outer_side, dt=0.01, damping=0.95):
    """One physics step: compute forces from overlaps and containment, update positions."""
    n = len(inner)
    forces = np.zeros((n, 3))  # fx, fy, fa for each hexagon
    
    # Repulsion forces from overlapping pairs
    for i in range(n):
        for j in range(i+1, n):
            h1 = (inner[i][0], inner[i][1], 1.0, inner[i][2])
            h2 = (inner[j][0], inner[j][1], 1.0, inner[j][2])
            depth = _hex_overlap_depth(h1, h2)
            if depth > 0:
                # Force proportional to overlap depth, pushing apart
                dx = inner[j][0] - inner[i][0]
                dy = inner[j][1] - inner[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx, dy = 1.0, 0.0
                    dist = 1.0
                else:
                    dx /= dist
                    dy /= dist
                
                # Strong repulsion force
                f = depth * 50.0
                forces[i, 0] -= f * dx
                forces[i, 1] -= f * dy
                forces[j, 0] += f * dx
                forces[j, 1] += f * dy
    
    # Containment forces: push hexagons back inside
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    cx, cy = outer_center
    a = np.radians(outer_angle)
    apothem = outer_side * np.sqrt(3) / 2.0
    normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    
    for idx, (ix, iy, ia) in enumerate(inner):
        verts = _hex_verts(ix, iy, 1.0, ia)
        for v in verts:
            dx = v[0] - cx
            dy = v[1] - cy
            for na in normal_angles:
                proj = dx * np.cos(na) + dy * np.sin(na)
                if proj > apothem:
                    viol = proj - apothem
                    # Force pushing back inside along the normal
                    fx = viol * np.cos(na) * 30.0
                    fy = viol * np.sin(na) * 30.0
                    forces[idx, 0] -= fx
                    forces[idx, 1] -= fy
                    break
    
    # Update positions with damping
    for i in range(n):
        inner[i][0] += forces[i, 0] * dt
        inner[i][1] += forces[i, 1] * dt
        # Keep angles fixed at 0 for simplicity (hexagons have 60-deg symmetry)
    
    return forces


def _init_hexagonal_grid(rows, cols, spacing_x, spacing_y, start_x=0, start_y=0):
    """Initialize inner hexagons in a hexagonal grid pattern."""
    inner = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = start_x + c * spacing_x + (r % 2) * spacing_x / 2
            y = start_y + r * spacing_y
            inner.append([x, y, 0.0])
            count += 1
    return inner


def _physics_simulate(init_inner, outer_side, max_steps=200, dt=0.01, damping=0.95, time_budget=5.0):
    """Run physics simulation to find non-overlapping packing."""
    inner = [list(p) for p in init_inner]
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > time_budget:
            break
        
        forces = _physics_step(inner, outer_side, dt=dt, damping=damping)
        
        # Check if converged (forces are small)
        max_force = np.max(np.abs(forces))
        if max_force < 1e-4:
            break
    
    return inner


def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = float('inf')
    best_angle = 0.0
    
    # Initial configurations to try
    configs = []
    
    # 3x4 grid
    configs.append(_init_hexagonal_grid(3, 4, 2.2, 2.0, -2.2, -2.0))
    # 4x3 grid
    configs.append(_init_hexagonal_grid(4, 3, 2.2, 2.0, -2.2, -3.0))
    # 2x6 grid
    configs.append(_init_hexagonal_grid(2, 6, 2.2, 2.0, -5.5, -1.0))
    # 6x2 grid
    configs.append(_init_hexagonal_grid(6, 2, 2.2, 2.0, -2.2, -5.5))
    
    # Random initializations
    np.random.seed(42)
    for seed in range(10):
        np.random.seed(seed)
        inner = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = 0.0  # Keep angles at 0 for simplicity
            inner.append([x, y, a])
        configs.append(inner)
    
    for init_inner in configs:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Estimate initial outer side
        all_x = [p[0] for p in init_inner]
        all_y = [p[1] for p in init_inner]
        span_x = max(all_x) - min(all_x) + 2.0
        span_y = max(all_y) - min(all_y) + 2.0
        init_side = max(span_x, span_y) / np.sqrt(3) + 1.5
        init_side = max(init_side, 3.5)
        
        # Run physics simulation
        simulated_inner = _physics_simulate(init_inner, init_side, max_steps=300, dt=0.005, time_budget=3.0)
        
        # Check if valid
        dp_violations = _disjointness_violations(simulated_inner)
        cp_violations = _containment_violation(simulated_inner, (0.0, 0.0), init_side, 0.0)
        
        if len(dp_violations) == 0 and len(cp_violations) == 0:
            if init_side < best_score:
                best_score = init_side
                best_inner = [list(p) for p in simulated_inner]
                best_center = (0.0, 0.0)
                best_side = init_side
                best_angle = 0.0
    
    # Try to improve by reducing outer side
    if best_inner is not None:
        for _ in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            new_side = best_side * 0.98  # Try 2% smaller
            if new_side < 3.0:
                break
            
            # Re-simulate with smaller outer side
            simulated_inner = _physics_simulate(best_inner, new_side, max_steps=200, dt=0.005, time_budget=3.0)
            
            dp_violations = _disjointness_violations(simulated_inner)
            cp_violations = _containment_violation(simulated_inner, (0.0, 0.0), new_side, 0.0)
            
            if len(dp_violations) == 0 and len(cp_violations) == 0:
                best_inner = [list(p) for p in simulated_inner]
                best_side = new_side
            else:
                break
    
    # If no valid solution found, return a conservative valid packing
    if best_inner is None:
        inner = []
        for i in range(N):
            x = (i - (N-1)/2.0) * 2.1
            y = 0.0
            inner.append([x, 0.0, 0.0])
        span = (N-1) * 2.1 + 2.0
        side = span / np.sqrt(3) + 0.5
        return inner, (0.0, 0.0), side, 0.0
    
    return best_inner, best_center, best_side, best_angle
