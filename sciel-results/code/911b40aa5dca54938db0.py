import numpy as np
import math
import time
import random

N = 11
DEADLINE = 120.0
SAFETY_MARGIN = 15.0

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angles = np.radians(angle_deg) + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def project_polygon(vertices, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    dots = vertices @ axis
    return dots.min(), dots.max()

def polygons_overlap_sat(v1, v2):
    """Check if two convex polygons overlap using Separating Axis Theorem.
    Returns True if they overlap (including touching)."""
    all_vertices = [v1, v2]
    for poly in all_vertices:
        n = len(poly)
        for i in range(n):
            # Edge from i to (i+1)%n
            edge = poly[(i+1) % n] - poly[i]
            # Normal to edge (perpendicular)
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            
            # Project both polygons onto this normal
            min1, max1 = project_polygon(v1, normal)
            min2, max2 = project_polygon(v2, normal)
            
            # Check for separation
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return False  # Separating axis found, no overlap
    return True  # No separating axis, polygons overlap

def check_all_disjoint(inner_hexes, eps=1e-6):
    """Check that all pairs of inner hexagons are disjoint."""
    n = len(inner_hexes)
    vertices_list = []
    for i in range(n):
        xi, yi, ai = inner_hexes[i]
        vi = hex_vertices(xi, yi, 1.0, ai)
        vertices_list.append(vi)
    
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap_sat(vertices_list[i], vertices_list[j]):
                return False
    return True

def compute_min_side(inner_hexes, center=(0.0, 0.0), outer_angle=0.0):
    """Compute the minimum outer hexagon side length to contain all inner hexagons."""
    cx, cy = center
    max_radius = 0.0
    for i in range(len(inner_hexes)):
        xi, yi, ai = inner_hexes[i]
        vi = hex_vertices(xi, yi, 1.0, ai)
        for k in range(6):
            a = np.radians(outer_angle) + k * np.pi / 3
            nx, ny = np.cos(a), np.sin(a)
            center_proj = cx * nx + cy * ny
            projections = vi[:, 0] * nx + vi[:, 1] * ny
            max_proj = projections.max()
            min_proj = projections.min()
            required = max(abs(max_proj - center_proj), abs(min_proj - center_proj))
            max_radius = max(max_radius, required)
    return 2.0 * max_radius / np.sqrt(3)

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Valid Honeycomb Lattice Generator
    # Nearest neighbor distance is sqrt(3)
    d = math.sqrt(3)
    
    # Generate a large grid of honeycomb positions
    # Basis vectors for honeycomb lattice (center to center)
    # v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 1.5) -- Wait, standard hex lattice vectors
    # Let's use axial coordinates (q, r)
    # x = d * (q + r/2)
    # y = d * (sqrt(3)/2 * r)
    
    positions = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            x = d * (q + r/2.0)
            y = d * (math.sqrt(3)/2.0 * r)
            positions.append((x, y))
            
    # Sort by distance from origin to pick the most compact cluster
    positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the 11 closest to origin
    # This forms a valid non-overlapping configuration
    initial_positions = positions[:N]
    
    # Center the configuration
    xs = [p[0] for p in initial_positions]
    ys = [p[1] for p in initial_positions]
    cx0 = np.mean(xs)
    cy0 = np.mean(ys)
    centered = [(x - cx0, y - cy0) for x, y in initial_positions]
    
    # Start SA with all angles 0
    current = [(x, y, 0.0) for x, y in centered]
    
    # Verify initial validity (should be true)
    if not check_all_disjoint(current):
        # Fallback to a very sparse valid config if somehow invalid
        current = [(k * 2.0, 0.0, 0.0) for k in range(N)]
        
    current_side = compute_min_side(current, center=(0.0, 0.0), outer_angle=0.0)
    
    best_in_sa = current[:]
    best_side_in_sa = current_side
    
    T = 0.3
    cooling = 0.9995
    n_steps = 20000 # Reduced steps to ensure speed, relying on good start
    
    for step in range(n_steps):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
            
        # Generate perturbation
        idx = random.randint(0, N - 1)
        dx = random.gauss(0, 0.2)
        dy = random.gauss(0, 0.2)
        da = random.gauss(0, 5.0)
        
        new_state = current[:]
        x, y, a = new_state[idx]
        new_state[idx] = (x + dx, y + dy, a + da)
        
        # Check overlap
        if not check_all_disjoint(new_state):
            T *= cooling
            continue
        
        # Compute new side length
        new_side = compute_min_side(new_state, center=(0.0, 0.0), outer_angle=0.0)
        
        # Acceptance
        delta = new_side - current_side
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            current = new_state
            current_side = new_side
            if current_side < best_side_in_sa:
                best_side_in_sa = current_side
                best_in_sa = current[:]
        
        T *= cooling
    
    # Final optimization of outer angle
    if best_in_sa is not None:
        for outer_ang in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
            side = compute_min_side(best_in_sa, center=(0.0, 0.0), outer_angle=outer_ang)
            if side < best_side:
                best_side = side
                best_inner = best_in_sa[:]
                best_center = (0.0, 0.0)
                best_outer_angle = outer_ang
                
    # Fallback if best_inner is still None (should not happen with valid start)
    if best_inner is None:
        # Sparse linear arrangement, spacing 2.0 > sqrt(3) ~ 1.732
        inner = [[k * 2.0, 0.0, 0.0] for k in range(N)]
        # Calculate side for this fallback
        side = compute_min_side(inner, center=(0.0, 0.0), outer_angle=0.0)
        return inner, (0.0, 0.0), side, 0.0
        
    return best_inner, best_center, best_side, best_outer_angle
