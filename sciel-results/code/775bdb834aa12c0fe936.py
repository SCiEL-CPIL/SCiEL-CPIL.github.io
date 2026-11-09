import numpy as np
import time

N = 11
START_TIME = time.time()
TIME_LIMIT = 140.0

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def min_enclosing_hexagon_side(points, cx, cy, angle_deg):
    if len(points) == 0:
        return 0.0
    points = np.array(points)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    sqrt3 = np.sqrt(3)
    val1 = lx + ly / sqrt3
    val2 = ly * 2 / sqrt3
    val3 = -lx + ly / sqrt3
    val4 = -lx - ly / sqrt3
    val5 = -ly * 2 / sqrt3
    val6 = lx - ly / sqrt3
    return float(np.max(np.maximum(np.maximum(val1, val2), np.maximum(val3, val4)), np.maximum(val5, val6)))

def get_all_vertices(positions, angles):
    all_verts = []
    for (x, y), ang in zip(positions, angles):
        verts = hexagon_vertices(x, y, 1.0, ang)
        all_verts.extend(verts.tolist())
    return all_verts

def evaluate_arrangement(positions, angles):
    all_verts = get_all_vertices(positions, angles)
    cx0 = np.mean([p[0] for p in positions])
    cy0 = np.mean([p[1] for p in positions])
    best_side = float('inf')
    best_angle = 0
    for outer_angle in range(0, 180, 15):
        side = min_enclosing_hexagon_side(all_verts, cx0, cy0, outer_angle)
        if side < best_side:
            best_side = side
            best_angle = outer_angle
    return best_side, best_angle, (cx0, cy0)

def construct():
    best_overall_side = float('inf')
    best_overall_result = None
    
    # Generate hexagonal grid positions (flat-top honeycomb)
    hex_positions = []
    for r in range(-3, 4):
        for q in range(-3, 4):
            x = 3.0 * (q + r / 2.0)
            y = np.sqrt(3) * r
            hex_positions.append((x, y))
    
    # Try various subsets of 11 positions
    # First, try the 11 closest to origin
    hex_positions_sorted = sorted(hex_positions, key=lambda p: p[0]**2 + p[1]**2)
    
    # Try different cluster shapes
    clusters = []
    
    # Cluster 1: 11 closest
    clusters.append(hex_positions_sorted[:11])
    
    # Cluster 2: Try a 4-3-4 arrangement manually
    # Row 0: 4 hexagons, Row 1: 3 hexagons, Row 2: 4 hexagons
    # In hex grid coords...
    
    # Let's just try the closest 11 and a few variations
    for cluster in clusters:
        if len(cluster) < 11:
            continue
        positions = [(x, y) for x, y in cluster[:11]]
        angles = [0.0] * 11
        
        side, outer_angle, center = evaluate_arrangement(positions, angles)
        if side < best_overall_side:
            best_overall_side = side
            best_overall_result = (positions, angles, center, side, outer_angle)
    
    if best_overall_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions, angles, center, side, outer_angle = best_overall_result
    inner = [[x, y, a] for (x, y), a in zip(positions, angles)]
    return inner, center, side, outer_angle
