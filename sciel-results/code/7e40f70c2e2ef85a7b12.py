"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Uses Simulated Annealing to minimize the outer hexagon side length.
"""
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
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
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
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT."""
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
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
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

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle, tol=tol):
                return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False
    
    return True

def _initial_lattice():
    """Create initial hexagonal lattice arrangement with correct spacing."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)  # Minimum distance for non-overlapping unit hexagons
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    # Add 4 more hexagons in a second ring
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    return positions

def _initial_random():
    """Create a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _simulated_annealing(positions, outer_angle, max_iters=5000, start_temp=1.0, end_temp=0.01):
    """Simulated Annealing to minimize outer side length."""
    positions = [list(p) for p in positions]
    current_side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    best_positions = [list(p) for p in positions]
    best_side = current_side
    
    # Pre-compute normals for SAT check speedup? No, angles change.
    
    for it in range(max_iters):
        if _remaining_time() < 5.0:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
        
        # Pick a random hexagon to move
        i = random.randint(0, N - 1)
        
        # Generate a perturbation
        dx = random.gauss(0, temp)
        dy = random.gauss(0, temp)
        da = random.gauss(0, temp * 10) # Angle perturbation in degrees
        
        new_positions = [list(p) for p in positions]
        new_positions[i][0] += dx
        new_positions[i][1] += dy
        new_positions[i][2] = (new_positions[i][2] + da) % 360
        
        # Check validity and compute new side
        new_side = _min_enclosing_side(new_positions, (0.0, 0.0), outer_angle)
        
        # Check overlaps for the moved hexagon against all others
        overlap_penalty = 0.0
        moved_verts = _hex_vertices(new_positions[i][0], new_positions[i][1], 1.0, new_positions[i][2])
        
        is_valid = True
        for j in range(N):
            if i == j: continue
            other_verts = _hex_vertices(new_positions[j][0], new_positions[j][1], 1.0, new_positions[j][2])
            if not _hexagons_disjoint(moved_verts, other_verts, tol=1e-5):
                is_valid = False
                # Estimate penalty based on center distance
                ci = np.mean(moved_verts, axis=0)
                cj = np.mean(other_verts, axis=0)
                d = np.linalg.norm(ci - cj)
                overlap_penalty += max(0, 2*np.sqrt(3) - d) ** 2
                break # Early exit if overlap found
        
        if not is_valid:
            # Accept with probability exp(-delta_E / T)
            delta_E = 100.0 * overlap_penalty
            if random.random() < math.exp(-delta_E / max(temp, 1e-9)):
                positions = new_positions
                current_side = new_side
            continue
        
        # If valid, accept if better or with probability exp(-delta_E / T)
        delta_E = new_side - current_side
        if delta_E < 0 or random.random() < math.exp(-delta_E / max(temp, 1e-9)):
            positions = new_positions
            current_side = new_side
            
            if current_side < best_side:
                best_side = current_side
                best_positions = [list(p) for p in positions]
                
    return best_positions, best_side

def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0, 15.0, 45.0]
    
    for oa in outer_angles:
        if _remaining_time() < 20.0:
            break
        
        initial_configs = [
            _initial_lattice(),
            _initial_random(),
            _initial_random(),
        ]
        
        for init_pos in initial_configs:
            if _remaining_time() < 10.0:
                break
            
            positions = [list(p) for p in init_pos]
            
            # Run Simulated Annealing
            positions, side = _simulated_annealing(positions, oa, max_iters=2000, start_temp=0.5, end_temp=0.001)
            
            # Verify validity
            if _check_validity(positions, (0.0, 0.0), side, oa, tol=1e-5):
                if side < best_score:
                    best_score = side
                    best_result = (positions, (0.0, 0.0), side, oa)
    
    if best_result is not None:
        return best_result
    
    # Fallback: known valid configuration
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
