"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 110.0  # safety margin before 120s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3.0
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def segment_distance(p1, p2, q1, q2):
    """Minimum distance between two line segments."""
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12:
        p1 = p1
        p2 = p1
        a = 0
    if e < 1e-12:
        q1 = q1
        q2 = q1
        e = 0
    
    if a == 0 and e == 0:
        return np.linalg.norm(p1 - q1)
    
    c = np.dot(d1, r)
    if e == 0:
        s = max(0, min(1, -c / a))
        t = 0
    else:
        b = np.dot(d1, d2)
        denom = a * e - b * b
        if denom < 1e-12:
            s = max(0, min(1, -c / a))
            t = (b * s + f) / e
            t = max(0, min(1, t))
            if s < 0:
                s = 0
                t = max(0, min(1, f / e))
            elif s > 1:
                s = 1
                t = max(0, min(1, (b + f) / e))
        else:
            s = (b * f - e * c) / denom
            if s < 0:
                s = 0
                t = max(0, min(1, f / e))
            elif s > 1:
                s = 1
                t = max(0, min(1, (b + f) / e))
            else:
                t = (b * s + f) / e
                if t < 0:
                    t = 0
                    s = max(0, min(1, -c / a))
                elif t > 1:
                    t = 1
                    s = max(0, min(1, (b - c) / a))
    
    return np.linalg.norm(p1 + s * d1 - (q1 + t * d2))

def hex_min_distance(v1, v2):
    """Minimum distance between two hexagons (arrays of 6 vertices each)."""
    min_d = np.inf
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            q1 = v2[j]
            q2 = v2[(j + 1) % 6]
            d = segment_distance(p1, p2, q1, q2)
            if d < min_d:
                min_d = d
    return min_d

def hex_in_hexagon(v_hex, center, side, angle_deg):
    """Check if all vertices of hex_hex are inside the outer hexagon.
    Returns minimum signed distance (negative if outside)."""
    outer = hex_vertices(center[0], center[1], side, angle_deg)
    min_dist = np.inf
    for i in range(6):
        v = v_hex[i]
        for j in range(6):
            # Distance from point v to edge (outer[j], outer[(j+1)%6])
            e1 = outer[j]
            e2 = outer[(j + 1) % 6]
            edge = e2 - e1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            edge_unit = edge / edge_len
            # Normal pointing inward
            normal = np.array([-edge_unit[1], edge_unit[0]])
            # Check which side the center is on
            to_center = center - e1
            if np.dot(to_center, normal) < 0:
                normal = -normal
            dist = np.dot(v - e1, normal)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def compute_cost(params):
    """Compute the cost: outer side length + penalties for violations."""
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        return 1e10
    
    # Unpack parameters
    # For each inner hex: cx, cy, angle (3 params each) = 36
    # Outer: cx, cy, side, angle (4 params)
    inner_params = params[:N * 3].reshape(N, 3)
    outer_cx, outer_cy, outer_side, outer_angle = params[N * 3:N * 3 + 4]
    
    if outer_side < 1.0:
        return 1e10
    
    # Compute vertices
    inner_verts = []
    for i in range(N):
        cx, cy, ang = inner_params[i]
        v = hex_vertices(cx, cy, 1.0, ang)
        inner_verts.append(v)
    
    # Penalty for overlaps
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = hex_min_distance(inner_verts[i], inner_verts[j])
            # Smooth penalty: if d < 0, penalize
            if d < 0:
                overlap_penalty += d * d
    
    # Penalty for being outside outer hexagon
    outside_penalty = 0.0
    for i in range(N):
        min_d = hex_in_hexagon(inner_verts[i], (outer_cx, outer_cy), outer_side, outer_angle)
        if min_d < 0:
            outside_penalty += min_d * min_d
    
    total_penalty = 100.0 * (overlap_penalty + outside_penalty)
    
    return outer_side + total_penalty

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial configuration: place hexagons in a grid-like pattern
    # Try multiple initial configurations and pick the best
    
    best_result = None
    best_score = np.inf
    
    # Initial position patterns
    patterns = []
    
    # Pattern 1: Linear arrangement
    for scale in [3.5, 4.0, 4.5, 5.0]:
        inner = [[scale * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        patterns.append(inner)
    
    # Pattern 2: 2x6 grid
    for scale in [3.5, 4.0]:
        inner = []
        for row in range(2):
            for col in range(6):
                x = (col - 2.5) * scale * 1.8
                y = (row - 0.5) * scale * 1.5
                inner.append([x, y, 0.0])
        patterns.append(inner)
    
    # Pattern 3: Hexagonal packing
    for scale in [3.5, 4.0]:
        inner = []
        # 4-3-4 arrangement or similar
        positions = [
            (-3, 0), (-1.5, 0.866), (0, 0), (1.5, 0.866), (3, 0),
            (-2.25, -0.866), (-0.75, -0.866), (0.75, -0.866), (2.25, -0.866),
            (-1.5, -1.732), (0, -1.732), (1.5, -1.732)
        ]
        for x, y in positions:
            inner.append([x * scale, y * scale, 0.0])
        patterns.append(inner)
    
    # Pattern 4: Circular arrangement
    for radius in [4.0, 5.0]:
        inner = []
        for i in range(N):
            angle = 2 * np.pi * i / N
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            inner.append([x, y, 0.0])
        patterns.append(inner)
    
    # Pattern 5: Two rows
    for dx in [3.0, 3.5]:
        inner = []
        for i in range(6):
            inner.append([(i - 2.5) * dx, 1.2, 0.0])
            inner.append([(i - 2.5) * dx, -1.2, 0.0])
        patterns.append(inner)
    
    # Optimize each pattern
    for init_inner in patterns:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        # Compute initial outer hexagon
        all_x = [p[0] for p in init_inner]
        all_y = [p[1] for p in init_inner]
        outer_cx = (min(all_x) + max(all_x)) / 2
        outer_cy = (min(all_y) + max(all_y)) / 2
        
        # Estimate initial side length
        max_dist = 0
        for p in init_inner:
            d = np.sqrt((p[0] - outer_cx)**2 + (p[1] - outer_cy)**2)
            max_dist = max(max_dist, d)
        outer_side = max_dist + 1.5
        
        # Build initial params
        params = []
        for p in init_inner:
            params.extend([p[0], p[1], p[2]])
        params.extend([outer_cx, outer_cy, outer_side, 0.0])
        params = np.array(params)
        
        # Optimize
        try:
            result = minimize(
                compute_cost,
                params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            if result.success or result.fun < best_score:
                best_params = result.x
                best_score = result.fun
                
                # Extract result
                inner_params = best_params[:N * 3].reshape(N, 3)
                outer_cx, outer_cy, outer_side, outer_angle = best_params[N * 3:N * 3 + 4]
                
                # Verify validity
                inner_verts = []
                valid = True
                for i in range(N):
                    v = hex_vertices(inner_params[i, 0], inner_params[i, 1], 1.0, inner_params[i, 2])
                    inner_verts.append(v)
                
                for i in range(N):
                    for j in range(i + 1, N):
                        d = hex_min_distance(inner_verts[i], inner_verts[j])
                        if d < -0.01:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    for i in range(N):
                        min_d = hex_in_hexagon(inner_verts[i], (outer_cx, outer_cy), outer_side, outer_angle)
                        if min_d < -0.01:
                            valid = False
                            break
                    
                    if valid:
                        inner_result = [[float(inner_params[i, 0]), float(inner_params[i, 1]), float(inner_params[i, 2])] for i in range(N)]
                        if best_result is None or outer_side < best_result[2]:
                            best_result = (inner_result, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_angle))
        except Exception:
            continue
    
    # Fallback if no valid result found
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
