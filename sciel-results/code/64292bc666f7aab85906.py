# EVOLVE-BLOCK-START
import time
import numpy as np
# EVOLVE-BLOCK-END

# Precompute sofa points once at module level for speed
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFA = len(_SOFA_PTS)


def _generate_placements(waypoints):
    """
    Generate the full list of placements (including start (0,0,0) and end (-2,0,90)).
    Each placement is (x, y, a_degrees).
    """
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]
    placements = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append((1 - t) * p1 + t * p2)
    return np.array(placements, dtype=np.float64)


def _compute_area_fast(waypoints):
    """
    Compute the area of the sofa (intersection of all corridor placements).
    """
    placements = _generate_placements(waypoints)
    total = len(placements)
    if total == 0:
        return 0.0

    mask = np.ones(_N_SOFA, dtype=bool)
    for p in placements:
        xx, yy, aa = p[0], p[1], np.radians(p[2])
        ca, sa = np.cos(aa), np.sin(aa)
        dx = _SOFA_PTS[:, 0] - xx
        dy = _SOFA_PTS[:, 1] - yy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        inside = ((u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)) | \
                 ((u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0))
        mask &= inside
        if not mask.any():
            return 0.0
    area = mask.sum() * 0.0001
    return area


def _arc_path(cx, cy, r, theta0, theta1, n_pts=20):
    """
    Generate waypoints along a circular arc from (cx + r*cos(theta0), cy + r*sin(theta0))
    to (cx + r*cos(theta1), cy + r*sin(theta1)), with angles going from 0 to 90 degrees.
    theta0 and theta1 are in radians.
    """
    if theta1 < theta0:
        # Ensure we go the right way (counterclockwise from theta0 to theta1)
        theta1 += 2 * np.pi
    thetas = np.linspace(theta0, theta1, n_pts)
    angles = np.linspace(0, 90, n_pts)
    xs = cx + r * np.cos(thetas)
    ys = cy + r * np.sin(thetas)
    return np.column_stack([xs, ys, angles])


def _straight_arc_straight_path(cx, cy, r, n_arc=15, n_straight=3):
    """
    Generate a path that goes:
    1. Straight from (0,0) toward the arc
    2. Along the arc
    3. Straight from the arc to (-2, 0)
    
    The arc is centered at (cx, cy) with radius r.
    """
    # Arc from angle 0 to angle 90 (relative to center)
    theta0 = 0.0
    theta1 = np.pi / 2.0
    
    arc_pts = _arc_path(cx, cy, r, theta0, theta1, n_arc)
    
    # Start of arc
    x_arc_start = cx + r * np.cos(theta0)
    y_arc_start = cy + r * np.sin(theta0)
    
    # End of arc
    x_arc_end = cx + r * np.cos(theta1)
    y_arc_end = cy + r * np.sin(theta1)
    
    # Straight segment from (0,0) to arc start
    straight1 = []
    for i in range(1, n_straight):
        t = i / n_straight
        x = t * x_arc_start
        y = t * y_arc_start
        a = t * arc_pts[0, 2]  # angle interpolates from 0
        straight1.append([x, y, a])
    
    # Straight segment from arc end to (-2, 0)
    straight2 = []
    for i in range(1, n_straight):
        t = i / n_straight
        x = x_arc_end + t * (-2.0 - x_arc_end)
        y = y_arc_end + t * (0.0 - y_arc_end)
        a = arc_pts[-1, 2] + t * (90.0 - arc_pts[-1, 2])
        straight2.append([x, y, a])
    
    all_pts = straight1 + [list(p) for p in arc_pts] + straight2
    return np.array(all_pts, dtype=np.float64)


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses a parametric circular arc model optimized via coordinate descent.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin

    # Initial guess: arc centered at (-0.5, 0.5) with radius 1.0
    best_cx, best_cy, best_r = -0.5, 0.5, 1.0
    best_wp = _straight_arc_straight_path(best_cx, best_cy, best_r)
    best_area = _compute_area_fast(best_wp)

    # Coordinate descent on (cx, cy, r)
    n_arc = 20
    n_straight = 4

    cx, cy, r = best_cx, best_cy, best_r
    curr_area = best_area

    # Step sizes for coordinate descent
    steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    for step in steps:
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False

            # Try moving cx
            for dx in [-step, step]:
                new_cx = cx + dx
                wp = _straight_arc_straight_path(new_cx, cy, r, n_arc, n_straight)
                area = _compute_area_fast(wp)
                if area > curr_area:
                    cx = new_cx
                    curr_area = area
                    improved = True
                    break
            else:
                # Try moving cy
                for dy in [-step, step]:
                    new_cy = cy + dy
                    wp = _straight_arc_straight_path(cx, new_cy, r, n_arc, n_straight)
                    area = _compute_area_fast(wp)
                    if area > curr_area:
                        cy = new_cy
                        curr_area = area
                        improved = True
                        break
                else:
                    # Try moving r
                    for dr in [-step, step]:
                        new_r = r + dr
                        if new_r <= 0:
                            continue
                        wp = _straight_arc_straight_path(cx, cy, new_r, n_arc, n_straight)
                        area = _compute_area_fast(wp)
                        if area > curr_area:
                            r = new_r
                            curr_area = area
                            improved = True
                            break

        if curr_area > best_area:
            best_area = curr_area
            best_wp = _straight_arc_straight_path(cx, cy, r, n_arc, n_straight)

    # Final refinement with more points
    if (time.time() - start_time) < deadline:
        for n_arc_fine in [30, 50]:
            wp = _straight_arc_straight_path(cx, cy, r, n_arc_fine, n_straight)
            area = _compute_area_fast(wp)
            if area > best_area:
                best_area = area
                best_wp = wp

    return best_wp
# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
