"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    cos_a, sin_a = np.cos(-a), np.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    apothem = side * np.sqrt(3) / 2
    for k in range(6):
        theta = (2 * k + 1) * np.pi / 6
        if lx * np.cos(theta) + ly * np.sin(theta) > apothem:
            return False
    return True

def _hexagon_overlap(h1, h2):
    """Check if two hexagons overlap. h1=(cx,cy,angle), h2=(cx,cy,angle). Both side=1."""
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    if dist > 2.0:
        return False
    
    if _point_in_hex(cx1, cy1, cx2, cy2, 1.0, a2):
        return True
    if _point_in_hex(cx2, cy2, cx1, cy1, 1.0, a1):
        return True
    
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
    for v in verts1:
        if _point_in_hex(v[0], v[1], cx2, cy2, 1.0, a2):
            return True
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
    for v in verts2:
        if _point_in_hex(v[0], v[1], cx1, cy1, 1.0, a1):
            return True
    
    return False

def _min_required_side(positions, angles, outer_center=(0.0, 0.0)):
    """Compute minimum outer hexagon side length to contain all unit hexagons.
    positions: list of (x,y), angles: list of angle_degrees.
    Outer hexagon centered at outer_center, 0 degrees rotation.
    """
    max_val = 0.0
    thetas = [(2*k+1)*np.pi/6 for k in range(6)]
    ox, oy = outer_center
    
    for i, (px, py) in enumerate(positions):
        verts = _hex_vertices(px, py, 1.0, angles[i])
        for vx, vy in verts:
            lx = vx - ox
            ly = vy - oy
            for theta in thetas:
                val = lx * np.cos(theta) + ly * np.sin(theta)
                if val > max_val:
                    max_val = val
    
    return max_val / (np.sqrt(3) / 2)

def _check_validity(positions, angles):
    """Check that all hexagons are pairwise disjoint."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if _hexagon_overlap((positions[i][0], positions[i][1], angles[i]),
                                (positions[j][0], positions[j][1], angles[j])):
                return False
    return True

def _hex_grid_positions(n):
    """Generate n positions in a hex grid, roughly centered."""
    positions = []
    sqrt3 = np.sqrt(3)
    # Add center
    positions.append((0.0, 0.0))
    ring = 1
    while len(positions) < n:
        for q in range(-ring, ring + 1):
            for r in range(-ring, ring + 1):
                if max(abs(q), abs(r), abs(q + r)) == ring:
                    x = 1.5 * q
                    y = sqrt3 * (r + q / 2.0)
                    if len(positions) < n:
                        positions.append((x, y))
        ring += 1
    return positions[:n]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Start with hexagonal grid arrangement
    positions = _hex_grid_positions(N)
    angles = [0.0] * N
    
    current_side = _min_required_side(positions, angles)
    best_side = current_side
    best_positions = [p[:] for p in positions]
    best_angles = angles[:]
    
    # Verify validity
    if not _check_validity(positions, angles):
        positions = [(4.0 * k, 0.0) for k in range(-(N//2), N - N//2)]
        angles = [0.0] * N
        current_side = _min_required_side(positions, angles)
        best_side = current_side
        best_positions = [p[:] for p in positions]
        best_angles = angles[:]
    
    # Local search with random perturbations
    np.random.seed(42)
    
    def _objective(x):
        """Objective: minimize max distance from origin (proxy for side length)."""
        pos = [(x[2*i], x[2*i+1]) for i in range(N)]
        ang = [x[2*i+2] for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if _hexagon_overlap((pos[i][0], pos[i][1], ang[i]),
                                    (pos[j][0], pos[j][1], ang[j])):
                    return 1e6
        side = _min_required_side(pos, ang)
        return side
    
    x0 = []
    for i in range(N):
        x0.extend([positions[i][0], positions[i][1], angles[i]])
    
    x_best = x0[:]
    f_best = _objective(x_best)
    
    # Simple random restart local search
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            x = x_best[:]
        else:
            x = x_best + np.random.randn(3*N) * 0.5
        
        for step in range(200):
            if time.time() > deadline - 5.0:
                break
            
            dx = np.random.randn(3*N) * 0.1
            x_new = x + dx
            
            f_new = _objective(x_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new[:]
                x = x_new[:]
            elif f_new < 1e6:
                if np.random.random() < 0.1:
                    x = x_new[:]
    
    # Extract best solution
    best_positions = [(x_best[2*i], x_best[2*i+1]) for i in range(N)]
    best_angles = [x_best[2*i+2] for i in range(N)]
    best_side = _min_required_side(best_positions, best_angles)
    
    # Verify final validity
    if not _check_validity(best_positions, best_angles):
        best_positions = [p[:] for p in positions]
        best_angles = angles[:]
        best_side = _min_required_side(best_positions, best_angles)
    
    # Add tiny epsilon to ensure strict containment
    best_side *= (1.0 + 1e-6)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
