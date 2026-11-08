"""Hexagon packing: place N unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3.0)
# Half-width (across flats) of unit hexagon = SQRT3/2
HF = SQRT3 / 2.0
# Full width across flats
FW = SQRT3

# EVOLVE-BLOCK-START
def _hexagon_vertices(center, angle_deg):
    """Return 6 vertices of a unit hexagon (circumradius 1) centered at center, rotated by angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return center[0] + np.cos(angles), center[1] + np.sin(angles)


def _outer_hexagon_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given side length (circumradius = side)."""
    a = np.deg2rad(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return center[0] + side * np.cos(angles), center[1] + side * np.sin(angles)


def _edge_distance(p1, p2, q1, q2):
    """Compute the distance between two line segments (p1,p2) and (q1,q2), and a gradient-like push."""
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    denom = a * c - b * b
    if denom < 1e-12:
        denom = 1e-12
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)
    closest_p = p1 + s * d1
    closest_q = q1 + t * d2
    diff = closest_p - closest_q
    dist = np.linalg.norm(diff)
    return dist, diff


def _min_distance_between_hexagons(cx1, cy1, a1, cx2, cy2, a2):
    """Find minimum distance between two unit hexagons. Returns (min_dist, push_vector).
    If overlapping (min_dist <= 0), push_vector points from hex2 toward hex1 (to separate)."""
    v1x, v1y = _hexagon_vertices([cx1, cy1], a1)
    v2x, v2y = _hexagon_vertices([cx2, cy2], a2)
    min_dist = 1e18
    push = np.array([0.0, 0.0])
    for i in range(6):
        p1 = np.array([v1x[i], v1y[i]])
        p2 = np.array([v1x[(i+1)%6], v1y[(i+1)%6]])
        for j in range(6):
            q1 = np.array([v2x[j], v2y[j]])
            q2 = np.array([v2x[(j+1)%6], v2y[(j+1)%6]])
            dist, diff = _edge_distance(p1, p2, q1, q2)
            if dist < min_dist:
                min_dist = dist
                # diff = closest_p - closest_q, so push hex1 away from hex2 along diff
                push = diff
    # Also check vertex-vertex distances
    for i in range(6):
        for j in range(6):
            p = np.array([v1x[i], v1y[i]])
            q = np.array([v2x[j], v2y[j]])
            diff = p - q
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                min_dist = dist
                push = diff
    return min_dist, push


def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside a regular hexagon with center (cx,cy), side length, rotation."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = -np.deg2rad(angle_deg)
    # Rotate point into hexagon's frame
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # Regular hexagon with circumradius = side
    # |lx| <= side, |ly| <= SQRT3/2 * side, and |ly| <= SQRT3 * (side - |lx|) for the angled sides
    if abs(lx) > side or abs(ly) > SQRT3 / 2 * side:
        return False
    # The hexagon is the intersection of 6 half-planes
    # For a hexagon with flat top/bottom (angle 0):
    # |ly| <= SQRT3/2 * side and |ly| <= SQRT3*(side - |lx|)
    if abs(ly) > SQRT3 * (side - abs(lx)):
        return False
    return True


def _min_distance_to_outer_boundary(cx, cy, a_deg, outer_cx, outer_cy, outer_side, outer_angle):
    """Find minimum distance from a unit hexagon to the outer hexagon boundary.
    Negative means the hexagon vertex is outside the outer hexagon."""
    vx, vy = _hexagon_vertices([cx, cy], a_deg)
    min_margin = 1e18
    push = np.array([0.0, 0.0])
    for i in range(6):
        px, py = vx[i], vy[i]
        if not _point_in_hexagon(px, py, outer_cx, outer_cy, outer_side, outer_angle):
            # This vertex is outside; compute distance to boundary
            # Find the closest point on the outer hexagon boundary
            ovx, ovy = _outer_hexagon_vertices([outer_cx, outer_cy], outer_side, outer_angle)
            best_dist = 1e18
            best_diff = np.array([0.0, 0.0])
            p = np.array([px, py])
            for j in range(6):
                q1 = np.array([ovx[j], ovy[j]])
                q2 = np.array([ovx[(j+1)%6], ovy[(j+1)%6]])
                dist, diff = _edge_distance(p, p, q1, q2)
                if dist < best_dist:
                    best_dist = dist
                    best_diff = diff  # points from boundary toward p (outward)
            # The hexagon needs to move inward (opposite to best_diff)
            if best_dist < min_margin:
                min_margin = -best_dist  # negative = violation
                push = -best_diff  # push hexagon inward
    return min_margin, push


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: hexagons in a row, evenly spaced
    # Try a few initial configurations
    best_result = None
    best_score = 1e18
    
    configs = [
        # Row of hexagons
        lambda i: (i * (2.0 + 0.01), 0.0, 0.0),
        # Row with slight alternating y offset
        lambda i: (i * 2.0, 0.0, 0.0),
    ]
    
    for cfg_idx, cfg in enumerate(configs):
        if time.time() > deadline:
            break
        
        # N hexagons, centered
        offset = -(N - 1) / 2.0
        positions = np.array([cfg(offset + i) for i in range(N)])
        # positions: (N, 3) = (x, y, angle_deg)
        
        # Optimize outer side length
        # Start with a reasonable estimate
        # For a row: total width = (N-1)*spacing + 2 (for the end hexagons' half-widths)
        # spacing = 2.0 (touching), so width = (N-1)*2 + 2 = 2N
        # Outer side (circumradius) needs to contain this
        # For a hexagon with flat top, width across flats = SQRT3 * side
        # So side >= total_width / SQRT3
        total_width = (N - 1) * 2.0 + 2.0  # = 2N
        outer_side = total_width / SQRT3 + 0.5  # initial guess
        outer_cx, outer_cy = 0.0, 0.0
        outer_angle = 0.0  # flat-top orientation
        
        # Gradient descent on positions and outer_side
        lr_pos = 0.05
        lr_side = 0.02
        penalty_weight = 10.0
        
        for step in range(3000):
            if time.time() > deadline:
                break
            
            # Compute total penalty
            total_loss = 0.0
            grad_pos = np.zeros_like(positions)
            grad_side = 0.0
            
            # Pairwise distances between inner hexagons
            for i in range(N):
                for j in range(i + 1, N):
                    dist, push = _min_distance_between_hexagons(
                        positions[i, 0], positions[i, 1], positions[i, 2],
                        positions[j, 0], positions[j, 1], positions[j, 2]
                    )
                    if dist < 0.01:  # overlapping or nearly touching
                        penalty = (0.01 - dist) ** 2
                        total_loss += penalty_weight * penalty
                        # push is from hex j toward hex i
                        grad_pos[i, 0] += penalty_weight * 2 * (0.01 - dist) * push[0]
                        grad_pos[i, 1] += penalty_weight * 2 * (0.01 - dist) * push[1]
                        grad_pos[j, 0] -= penalty_weight * 2 * (0.01 - dist) * push[0]
                        grad_pos[j, 1] -= penalty_weight * 2 * (0.01 - dist) * push[1]
            
            # Distance to outer boundary
            for i in range(N):
                margin, push = _min_distance_to_outer_boundary(
                    positions[i, 0], positions[i, 1], positions[i, 2],
                    outer_cx, outer_cy, outer_side, outer_angle
                )
                if margin < 0.01:
                    penalty = (0.01 - margin) ** 2
                    total_loss += penalty_weight * penalty
                    grad_pos[i, 0] += penalty_weight * 2 * (0.01 - margin) * push[0]
                    grad_pos[i, 1] += penalty_weight * 2 * (0.01 - margin) * push[1]
                    # Reducing outer_side makes it harder, so gradient on side is negative
                    grad_side -= penalty_weight * 2 * (0.01 - margin)
            
            # Objective: minimize outer_side + penalty
            total_loss += outer_side
            
            # Update
            positions -= lr_pos * grad_pos
            outer_side -= lr_side * grad_side
            outer_side = max(outer_side, 5.0)  # keep it reasonable
            
            # Keep hexagons roughly centered
            cx_mean = np.mean(positions[:, 0])
            cy_mean = np.mean(positions[:, 1])
            positions[:, 0] -= 0.01 * cx_mean
            positions[:, 1] -= 0.01 * cy_mean
            
            # Check convergence
            if step % 100 == 0:
                # Verify validity
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        dist, _ = _min_distance_between_hexagons(
                            positions[i, 0], positions[i, 1], positions[i, 2],
                            positions[j, 0], positions[j, 1], positions[j, 2]
                        )
                        if dist < -0.01:
                            valid = False
                    margin, _ = _min_distance_to_outer_boundary(
                        positions[i, 0], positions[i, 1], positions[i, 2],
                        outer_cx, outer_cy, outer_side, outer_angle
                    )
                    if margin < -0.01:
                        valid = False
                if valid:
                    score = outer_side
                    if score < best_score:
                        best_score = score
                        best_result = (positions.copy(), outer_cx, outer_cy, outer_side, outer_angle)
        
        # Final validation
        if best_result is not None:
            positions, outer_cx, outer_cy, outer_side, outer_angle = best_result
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    dist, _ = _min_distance_between_hexagons(
                        positions[i, 0], positions[i, 1], positions[i, 2],
                        positions[j, 0], positions[j, 1], positions[j, 2]
                    )
                    if dist < -0.001:
                        valid = False
                margin, _ = _min_distance_to_outer_boundary(
                    positions[i, 0], positions[i, 1], positions[i, 2],
                    outer_cx, outer_cy, outer_side, outer_angle
                )
                if margin < -0.001:
                    valid = False
            if valid:
                return [[float(positions[i, 0]), float(positions[i, 1]), float(positions[i, 2])] for i in range(N)], \
                       (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle)
    
    # Fallback: return the seed solution
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
