import numpy as np
import time
import scipy.optimize as opt

N = 11
APOTHEM_INNER = np.sqrt(3) / 2.0
MIN_DIST = 2 * APOTHEM_INNER

def _hexagon_vertices(center, angle_deg):
    """Returns vertices of a unit hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    # Flat-topped hexagon vertices
    angles = np.arange(6) * np.pi / 3.0 + angle_rad
    x = center[0] + np.cos(angles)
    y = center[1] + np.sin(angles)
    return np.column_stack((x, y))

def _outer_hexagon_normals(angle_deg):
    """Returns normal vectors for the sides of the outer hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    # Normals to sides of a flat-topped hexagon are at 0, 60, ... if vertices are at 30, 90...
    # Let's assume standard orientation: vertices at 0, 60, ...
    # Then side normals are at 30, 90, ...
    angles = np.arange(6) * np.pi / 3.0 + np.pi / 6.0 + angle_rad
    nx = np.cos(angles)
    ny = np.sin(angles)
    return np.column_stack((nx, ny))

def _calculate_required_side(positions, angles_deg, outer_angle_deg=0.0):
    """Calculates the minimum side length of the outer hexagon to contain all inner hexagons."""
    normals = _outer_hexagon_normals(outer_angle_deg)
    max_proj = 0.0
    for i in range(len(positions)):
        verts = _hexagon_vertices(positions[i], angles_deg[i])
        # Project vertices onto normals
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    # Apothem of outer hexagon must be >= max_proj
    # Apothem = L * sqrt(3) / 2
    return max_proj * 2.0 / np.sqrt(3)

def _penalty(positions):
    """Penalty for overlapping inner hexagons."""
    penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < MIN_DIST:
                penalty += (MIN_DIST - dist) ** 2
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    
    # Initial configurations
    # 1. Hexagonal lattice cluster
    # 2. Random
    
    def run_optimization(init_pos, init_angles):
        nonlocal best_side, best_positions, best_angles
        if time.time() > deadline:
            return
        
        def objective(vars):
            pos = vars[:2 * N].reshape(N, 2)
            ang = vars[2 * N:]
            
            # Check time inside objective? No, too slow.
            
            side = _calculate_required_side(pos, ang)
            pen = _penalty(pos)
            
            # If penalty is high, we want to reduce it.
            # We want to minimize side, but side is only valid if pen is 0.
            # Use a large penalty coefficient.
            if pen > 0.01:
                return side + 100.0 * pen
            return side
        
        x0 = np.concatenate([init_pos.flatten(), init_angles])
        
        # Use L-BFGS-B
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
            if res.success or time.time() < deadline:
                pos = res.x[:2 * N].reshape(N, 2)
                ang = res.x[2 * N:]
                side = _calculate_required_side(pos, ang)
                pen = _penalty(pos)
                if pen < 1e-4:
                    if side < best_side:
                        best_side = side
                        best_positions = pos
                        best_angles = ang
        except Exception:
            pass

    # Restart 1: Lattice
    # Center at 0,0
    # 11 hexagons. 
    # Hex grid coords:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1) -> 7
    # Next ring: (2,0), (-2,0), (0,2), (0,-2), (2,1), (-2,-1), (1,2), (-1,-2), (2,-1), (-2,1), (1,-2), (-1,2)
    # We need 4 more.
    # Let's just place them in a 3x3 block minus corners?
    # Spacing = MIN_DIST
    
    spacing = MIN_DIST
    # Generate a dense set of points and pick 11
    candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Hex to cartesian
            x = spacing * (i + j / 2.0)
            y = spacing * (j * np.sqrt(3) / 2.0)
            candidates.append((x, y))
    
    # Sort by distance to center
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos1 = np.array(candidates[:N])
    init_angles1 = np.zeros(N)
    
    run_optimization(init_pos1, init_angles1)
    
    # Restart 2: Random
    for k in range(5):
        if time.time() > deadline:
            break
        init_pos2 = np.random.uniform(-2, 2, (N, 2))
        init_angles2 = np.zeros(N)
        run_optimization(init_pos2, init_angles2)
        
    if best_positions is None:
        # Fallback to a valid but large solution
        best_positions = np.array([[4.0 * k, 0.0] for k in range(-(N // 2), N - N // 2)])
        best_angles = np.zeros(N)
        best_side = 40.0

    return [list(p) + [a] for p, a in zip(best_positions, best_angles)], (0.0, 0.0), best_side, 0.0

