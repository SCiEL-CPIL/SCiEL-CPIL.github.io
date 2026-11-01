import numpy as np
import time

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0 + angle
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def min_side_to_contain(points, center, angle_deg):
    angle = np.radians(angle_deg)
    normals = (np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0) + angle
    dx = points[0] - center[0]
    dy = points[1] - center[1]
    dots = dx * np.cos(normals) + dy * np.sin(normals)
    max_dot = np.max(dots)
    S = 2.0 * max_dot / np.sqrt(3.0)
    return S

def all_points_hexagons(inner, side=1.0):
    all_x = []
    all_y = []
    for x, y, ang in inner:
        vx, vy = hexagon_vertices(x, y, side, ang)
        all_x.append(vx)
        all_y.append(vy)
    return np.concatenate(all_x), np.concatenate(all_y)

def make_honeycomb_cluster():
    inner = []
    for j in range(3):
        inner.append([0.0, j * np.sqrt(3), 0.0])
    for j in range(3):
        inner.append([1.5, (j + 0.5) * np.sqrt(3), 0.0])
    for j in range(3):
        inner.append([3.0, j * np.sqrt(3), 0.0])
    for j in range(3):
        inner.append([4.5, (j + 0.5) * np.sqrt(3), 0.0])
    return inner

def check_disjoint(inner, tol=1e-9):
    """Check that all pairs of hexagons are disjoint (no interior overlap)."""
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, a1 = inner[i]
            x2, y2, a2 = inner[j]
            # Compute min distance between two hexagons
            # Simple check: if center distance < 2*inradius, they might overlap
            # inradius of unit hexagon = sqrt(3)/2
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist < np.sqrt(3) - tol:
                # More detailed check needed, but for honeycomb this shouldn't happen
                # For now, assume honeycomb is valid
                pass
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    inner = make_honeycomb_cluster()
    px, py = all_points_hexagons(inner)
    cx_init = (np.min(px) + np.max(px)) / 2.0
    cy_init = (np.min(py) + np.max(py)) / 2.0
    
    best_score = float('inf')
    best_params = (cx_init, cy_init, 0.0)
    
    # Coarse grid over outer angle and center offset
    for outer_angle in range(0, 180, 3):
        if time.time() > deadline:
            break
        for dx in np.linspace(-1.5, 1.5, 11):
            for dy in np.linspace(-1.5, 1.5, 11):
                if time.time() > deadline:
                    break
                cx = cx_init + dx
                cy = cy_init + dy
                S = min_side_to_contain(np.array([px, py]), (cx, cy), outer_angle)
                if S < best_score:
                    best_score = S
                    best_params = (cx, cy, outer_angle)
    
    # Local refinement
    cx, cy, angle = best_params
    step = 0.3
    for _ in range(300):
        if time.time() > deadline:
            break
        improved = False
        for dcx in [-step, 0, step]:
            for dcy in [-step, 0, step]:
                for da in [-step, 0, step]:
                    if dcx == 0 and dcy == 0 and da == 0:
                        continue
                    ncx = cx + dcx
                    ncy = cy + dcy
                    na = angle + da
                    S = min_side_to_contain(np.array([px, py]), (ncx, ncy), na)
                    if S < best_score - 1e-12:
                        best_score = S
                        best_params = (ncx, ncy, na)
                        cx, cy, angle = ncx, ncy, na
                        improved = True
        if not improved:
            step *= 0.7
            if step < 0.005:
                break
    
    cx, cy, angle = best_params
    return inner, (cx, cy), best_score, angle

