"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon."""
import numpy as np
import time
import random
import math

N = 11
DEADLINE = 150.0
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

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        a = np.radians(30 + k * 60)
        nx = np.cos(a)
        ny = np.sin(a)
        if lx * nx + ly * ny > inradius + 1e-9:
            return False
    return True

def _hex_contains_all_points(points, cx, cy, side, angle_deg):
    """Check if all points are inside the hexagon."""
    for p in points:
        if not _point_in_hex(p[0], p[1], cx, cy, side, angle_deg):
            return False
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons (as vertex arrays) are disjoint using SAT."""
    n1 = len(verts1)
    n2 = len(verts2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon (with given center and angle) to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / np.sqrt(3)
                best_side = max(best_side, required_side)
    
    return best_side

def _check_validity(inner_positions, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: all inner hexagons are inside outer, and no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        if not _hex_contains_all_points(iverts, cx, cy, outer_side, outer_angle):
            return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-6):
                return False
    
    return True

def _initial_lattice():
    """Create initial hexagonal lattice arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210, 270, 330]
    for a_deg in angles2[:4]:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    return positions

def _force_directed_relax(positions, max_iters=200, dt=0.01):
    """Simple force-directed relaxation to resolve overlaps."""
    positions = [list(p) for p in positions]
    
    for it in range(max_iters):
        if _remaining_time() < 10.0:
            break
        forces = np.zeros((N, 2))
        
        for i in range(N):
            vi = _hex_vertices(positions[i][0], positions[i][1], 1.0, positions[i][2])
            for j in range(i+1, N):
                vj = _hex_vertices(positions[j][0], positions[j][1], 1.0, positions[j][2])
                if not _hexagons_disjoint(vi, vj, tol=1e-4):
                    di = np.array([positions[i][0], positions[i][1]])
                    dj = np.array([positions[j][0], positions[j][1]])
                    diff = di - dj
                    norm = np.linalg.norm(diff)
                    if norm > 1e-9:
                        f = diff / norm * 0.5
                    else:
                        f = np.array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])
                    forces[i] += f
                    forces[j] -= f
        
        for i in range(N):
            positions[i][0] += forces[i][0] * dt
            positions[i][1] += forces[i][1] * dt
    
    return positions

def _simulated_annealing(positions, outer_center, outer_angle, t0=5.0, t1=0.01, cooling=0.995):
    """Simulated annealing to optimize the configuration."""
    best_positions = [list(p) for p in positions]
    best_side = _min_enclosing_side(best_positions, outer_center, outer_angle)
    
    current_positions = [list(p) for p in positions]
    current_side = _min_enclosing_side(current_positions, outer_center, outer_angle)
    
    T = t0
    step = 0.5
    
    while T > t1 and _remaining_time() > 15.0:
        new_positions = [list(p) for p in current_positions]
        for i in range(N):
            new_positions[i][0] += random.gauss(0, step)
            new_positions[i][1] += random.gauss(0, step)
            new_positions[i][2] += random.gauss(0, step * 10)
            new_positions[i][2] = new_positions[i][2] % 360
        
        new_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in new_positions]
        overlap_count = 0
        for i in range(N):
            for j in range(i+1, N):
                if not _hexagons_disjoint(new_verts[i], new_verts[j], tol=1e-5):
                    overlap_count += 1
        
        cur_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in current_positions]
        cur_overlap = 0
        for i in range(N):
            for j in range(i+1, N):
                if not _hexagons_disjoint(cur_verts[i], cur_verts[j], tol=1e-5):
                    cur_overlap += 1
        
        if overlap_count < cur_overlap:
            current_positions = new_positions
            current_side = _min_enclosing_side(current_positions, outer_center, outer_angle)
            if current_side < best_side:
                best_positions = [list(p) for p in current_positions]
                best_side = current_side
        elif overlap_count == cur_overlap:
            if random.random() < 0.1:
                current_positions = new_positions
                current_side = _min_enclosing_side(current_positions, outer_center, outer_angle)
                if current_side < best_side:
                    best_positions = [list(p) for p in current_positions]
                    best_side = current_side
        else:
            prob = np.exp(-(overlap_count - cur_overlap) / max(T, 1e-10))
            if random.random() < prob:
                current_positions = new_positions
                current_side = _min_enclosing_side(current_positions, outer_center, outer_angle)
        
        T *= cooling
        step = max(0.01, 0.5 * T / t0)
    
    return best_positions, best_side

def _local_search(positions, outer_center, outer_angle, max_iters=500):
    """Local search with greedy improvement to fine-tune positions."""
    positions = [list(p) for p in positions]
    current_side = _min_enclosing_side(positions, outer_center, outer_angle)
    best_positions = [list(p) for p in positions]
    best_side = current_side
    
    step = 0.1
    
    for it in range(max_iters):
        if _remaining_time() < 10.0:
            break
        
        improved = False
        for i in range(N):
            # Try small perturbations of each hexagon
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_pos = [list(p) for p in positions]
                    new_pos[i][0] += dx
                    new_pos[i][1] += dy
                    
                    # Check validity
                    if not _check_validity(new_pos, outer_center, best_side, outer_angle):
                        continue
                    
                    new_side = _min_enclosing_side(new_pos, outer_center, outer_angle)
                    if new_side < best_side - 1e-6:
                        positions = new_pos
                        best_positions = [list(p) for p in positions]
                        best_side = new_side
                        current_side = new_side
                        improved = True
        
        if not improved:
            step *= 0.8
            if step < 0.001:
                break
    
    return best_positions, best_side

def construct():
    """Main entry point. Returns inner positions, outer center, outer side, outer angle."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    # Try different initial configurations and outer angles
    outer_angles = [0.0, 30.0, 15.0, 45.0]
    
    for oa in outer_angles:
        if _remaining_time() < 30.0:
            break
        
        # Start with lattice
        positions = _initial_lattice()
        
        # Force-directed relaxation
        positions = _force_directed_relax(positions, max_iters=100, dt=0.02)
        
        # Simulated annealing
        positions, side = _simulated_annealing(positions, (0.0, 0.0), oa, t0=3.0, t1=0.05, cooling=0.997)
        
        # Local search for fine-tuning
        positions, side = _local_search(positions, (0.0, 0.0), oa, max_iters=300)
        
        # Verify validity
        if _check_validity(positions, (0.0, 0.0), side, oa):
            if side < best_score:
                best_score = side
                best_result = (positions, (0.0, 0.0), side, oa)
        
        # Also try with slightly adjusted outer center
        for dcx in [-0.1, 0.0, 0.1]:
            for dcy in [-0.1, 0.0, 0.1]:
                if _remaining_time() < 20.0:
                    break
                oc = (dcx, dcy)
                side2 = _min_enclosing_side(positions, oc, oa)
                if _check_validity(positions, oc, side2, oa):
                    if side2 < best_score:
                        best_score = side2
                        best_result = (positions, oc, side2, oa)
    
    if best_result is not None:
        return best_result
    
    # Fallback: use a known valid configuration
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
