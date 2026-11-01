import numpy as np
from scipy.optimize import minimize
import time

N = 11
_DEADLINE = 150.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _remaining():
    return max(0.0, _DEADLINE - 3.0 - _elapsed())

def _hex_vertices(cx, cy, s, angle_rad):
    """Vertices of a regular hexagon with side s, center (cx,cy), rotation angle_rad (radians)."""
    a = angle_rad
    return np.array([
        [cx + s * np.cos(a + np.pi * k / 3.0),
         cy + s * np.sin(a + np.pi * k / 3.0)]
        for k in range(6)
    ])

def _min_outer_side(centers, outer_rot_rad, cx, cy):
    """Minimum side length of outer hexagon (center (cx,cy), rotation outer_rot_rad) containing all centers."""
    # The 3 normal directions of the outer hexagon (to the midpoints of sides)
    # For a hexagon with rotation a, the outward normal to the side at index 0 is at angle a + pi/6
    normals = [outer_rot_rad + np.pi / 6.0 + np.pi * k / 3.0 for k in range(3)]
    max_proj = 0.0
    for n in normals:
        nx, ny = np.cos(n), np.sin(n)
        # project all centers onto this direction, subtract center
        d = np.max(np.dot(centers - np.array([cx, cy]), np.array([nx, ny])))
        if d > max_proj:
            max_proj = d
    # For a regular hexagon, the distance from center to each side is s * cos(pi/6) = s * sqrt(3)/2
    # So s = d_max / cos(pi/6), where d_max is the max projection onto any normal
    side = max_proj / (np.sqrt(3.0) / 2.0) + 1.0  # +1 for the inner hexagon's own extent
    return side

def _objective(params):
    phi, outer_rot, cx, cy = params
    # Build hexagonal lattice (3-4-3) rotated by phi
    s3 = np.sqrt(3.0)
    # Base centers (unrotated): rows of 3, 4, 3
    row_y = [-1.5, 0.0, 1.5]
    base = []
    for r, y in enumerate(row_y):
        if r == 1:
            xs = [-1.5 * s3, -0.5 * s3, 0.5 * s3, 1.5 * s3]
        else:
            xs = [-s3, 0.0, s3]
        for x in xs:
            base.append((x, y))
    base = np.array(base)
    # Rotate by phi
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    centers = base @ R.T
    side = _min_outer_side(centers, outer_rot, cx, cy)
    return side

def construct():
    if _remaining() < 5.0:
        # Not enough time; return the 3-4-3 layout with default orientation
        s3 = np.sqrt(3.0)
        inner = []
        for y, xs in [(-1.5, [-s3, 0.0, s3]), (0.0, [-1.5 * s3, -0.5 * s3, 0.5 * s3, 1.5 * s3]), (1.5, [-s3, 0.0, s3])]:
            for x in xs:
                inner.append([x, y, 0.0])
        side = _min_outer_side(np.array([[x, y] for x, y, _ in inner]), 0.0, 0.0, 0.0)
        return inner, [0.0, 0.0], side, 0.0

    best_params = None
    best_val = float('inf')

    # Try multiple initial points for Nelder-Mead
    initial_guesses = []
    for phi0 in [0.0, np.pi / 6.0, np.pi / 12.0]:
        for rot0 in [0.0, np.pi / 6.0]:
            initial_guesses.append([phi0, rot0, 0.0, 0.0])

    for guess in initial_guesses:
        if _remaining() < 5.0:
            break
        try:
            res = minimize(_objective, guess, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        # Fallback: use the first guess
        best_params = initial_guesses[0]
        best_val = _objective(best_params)

    phi, outer_rot, cx, cy = best_params
    s3 = np.sqrt(3.0)
    row_y = [-1.5, 0.0, 1.5]
    base = []
    for r, y in enumerate(row_y):
        if r == 1:
            xs = [-1.5 * s3, -0.5 * s3, 0.5 * s3, 1.5 * s3]
        else:
            xs = [-s3, 0.0, s3]
        for x in xs:
            base.append((x, y))
    base = np.array(base)
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s], [s, c]])
    centers = base @ R.T

    inner = []
    for (x, y) in centers:
        inner.append([float(x), float(y), float(np.degrees(phi))])

    side = float(best_val)
    return inner, [float(cx), float(cy)], side, float(np.degrees(outer_rot))
