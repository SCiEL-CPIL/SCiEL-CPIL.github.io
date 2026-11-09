import numpy as np
import time
import scipy.optimize as opt

N = 11
APOTHEM_INNER = np.sqrt(3) / 2.0
# For same orientation, min center distance is sqrt(3). Add small epsilon.
MIN_DIST = np.sqrt(3) + 0.01

def _hexagon_vertices(center, angle_deg=0.0):
    """Returns vertices of a unit hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + angle_rad
    x = center[0] + np.cos(angles)
    y = center[1] + np.sin(angles)
    return np.column_stack((x, y))

def _outer_hexagon_normals(angle_deg=0.0):
    """Returns normal vectors for the sides of the outer hexagon."""
    angle_rad = np.deg2rad(angle_deg)
    angles = np.arange(6) * np.pi / 3.0 + np.pi / 6.0 + angle_rad
    nx = np.cos(angles)
    ny = np.sin(angles)
    return np.column_stack((nx, ny))

def _calculate_required_side(positions, outer_angle_deg=0.0):
    """Calculates the minimum side length of the outer hexagon to contain all inner hexagons (all at angle 0)."""
    normals = _outer_hexagon_normals(outer_angle_deg)
    max_proj = 0.0
    for i in range(len(positions)):
        verts = _hexagon_vertices(positions[i], 0.0)
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    return max_proj * 2.0 / np.sqrt(3)

def _penalty(positions):
    """Penalty for overlapping inner hexagons (same orientation)."""
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
    
    # Fallback valid configuration: Linear chain with safe spacing
    # For same orientation, linear chain along x-axis requires dist >= 2.0?
    # No, if aligned, they can touch side-by-side?
    # In a linear chain x-axis, if angle is 0 (vertex at 1,0), the width is 2.
    # So dist 2.0 is needed for a simple linear chain along x.
    # But we can pack better.
    # Fallback: Honeycomb-like or just a safe grid.
    # Let's use a safe grid with spacing 2.0 to be absolutely safe for fallback.
    spacing = 2.0
    cols = int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    fallback_positions = []
    for r in range(rows):
        for c in range(cols):
            if len(fallback_positions) < N:
                x = c * spacing
                y = r * spacing * np.sqrt(3) / 2.0 # Hexagonal packing offset
                if c % 2 == 1:
                    y += spacing / 2.0
                fallback_positions.append([x, y])
    fallback_positions = np.array(fallback_positions[:N])
    
    best_positions = fallback_positions.copy()
    best_side = _calculate_required_side(best_positions)

    def run_optimization(init_pos):
        nonlocal best_side, best_positions
        if time.time() > deadline:
            return
        
        def objective(vars):
            pos = vars.reshape(N, 2)
            pen = _penalty(pos)
            if pen > 1e-6:
                return 1e6 + pen
            side = _calculate_required_side(pos)
            return side
        
        x0 = init_pos.flatten()
        
        try:
            res = opt.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            if time.time() > deadline:
                return
            pos = res.x.reshape(N, 2)
            
            pen = _penalty(pos)
            if pen < 1e-4:
                side = _calculate_required_side(pos)
                if side < best_side:
                    best_side = side
                    best_positions = pos
        except Exception:
            pass

    # Restart 1: Fallback
    run_optimization(fallback_positions)
    
    # Restart 2: Perturbations
    for k in range(5):
        if time.time() > deadline:
            break
        
        if best_positions is not None:
            perturbed_pos = best_positions + np.random.uniform(-0.2, 0.2, (N, 2))
            run_optimization(perturbed_pos)
            
            perturbed_pos2 = best_positions + np.random.uniform(-0.5, 0.5, (N, 2))
            run_optimization(perturbed_pos2)

    # Final validation
    valid = True
    n = len(best_positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_positions[i] - best_positions[j])
            if dist < MIN_DIST:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        best_positions = fallback_positions
        best_side = _calculate_required_side(best_positions)

    # Return angles as 0.0 for all inner hexagons
    return [list(p) + [0.0] for p in best_positions], (0.0, 0.0), best_side, 0.0
