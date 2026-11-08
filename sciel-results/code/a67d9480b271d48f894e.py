"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def hexagon_vertices(cx, cy, side, angle_degrees):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle_rad = np.deg2rad(angle_degrees)
    angles = angle_rad + np.pi * np.arange(6) / 3.0
    verts = np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])
    return verts

def polygon_area(verts):
    """Shoelace formula for polygon area."""
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def point_in_polygon(x, y, verts):
    """Ray-casting point in polygon test."""
    n = len(verts)
    inside = False
    px, py = verts[-1]
    for i in range(n):
        qx, qy = verts[i]
        if ((py > y) != (qy > y)) and (x < (qx - px) * (y - py) / (qy - py) + px):
            inside = not inside
        px, py = qx, qy
    return inside

def polygons_intersect(verts1, verts2):
    """Check if two convex polygons intersect using separating axis theorem."""
    edges = []
    for verts in [verts1, verts2]:
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-12:
                normal /= np.linalg.norm(normal)
            edges.append(normal)
    
    for normal in edges:
        proj1 = np.dot(verts1, normal)
        proj2 = np.dot(verts2, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 or max2 < min1:
            return False
    return True

def check_validity(inner, center, side, angle):
    """Check if all inner hexagons are disjoint and contained in the outer hexagon."""
    outer_verts = hexagon_vertices(center[0], center[1], side, angle)
    inner_verts = []
    for x, y, ang in inner:
        inner_verts.append(hexagon_vertices(x, y, 1.0, ang))
    
    # Check containment
    for i, iv in enumerate(inner_verts):
        # All vertices of inner hexagon must be inside outer
        for v in iv:
            if not point_in_polygon(v[0], v[1], outer_verts):
                return False
    
    # Check disjointness
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if polygons_intersect(inner_verts[i], inner_verts[j]):
                return False
    
    return True

def energy(params, center, side, angle):
    """Compute energy: overlap penalties + containment penalty."""
    n_params = len(params)
    x = params[:n_params // 3]
    y = params[n_params // 3: 2 * n_params // 3]
    ang = params[2 * n_params // 3:]
    
    inner_verts = []
    for i in range(N):
        inner_verts.append(hexagon_vertices(x[i], y[i], 1.0, ang[i]))
    
    outer_verts = hexagon_vertices(center[0], center[1], side, angle)
    
    energy_val = 0.0
    
    # Containment penalty
    for i in range(N):
        for v in inner_verts[i]:
            if not point_in_polygon(v[0], v[1], outer_verts):
                # Compute distance to outer polygon (approximate via closest vertex/edge)
                # Simplified: penalize based on how far outside
                min_dist = float('inf')
                for j in range(len(outer_verts)):
                    p1 = outer_verts[j]
                    p2 = outer_verts[(j + 1) % len(outer_verts)]
                    # Project point onto edge
                    edge = p2 - p1
                    t = np.dot(v - p1, edge) / (np.dot(edge, edge) + 1e-12)
                    t = max(0, min(1, t))
                    closest = p1 + t * edge
                    dist = np.linalg.norm(v - closest)
                    min_dist = min(min_dist, dist)
                energy_val += min_dist ** 2
    
    # Overlap penalty between inner hexagons
    for i in range(N):
        for j in range(i + 1, N):
            if polygons_intersect(inner_verts[i], inner_verts[j]):
                # Approximate overlap area via centroid distance
                c1 = np.mean(inner_verts[i], axis=0)
                c2 = np.mean(inner_verts[j], axis=0)
                dist = np.linalg.norm(c1 - c2)
                # If centroids are close, heavy penalty
                energy_val += max(0, 2.0 - dist) ** 2
    
    return energy_val

def construct():
    deadline = time.time() + 145.0  # 145s limit for safety
    
    # Initial compact honeycomb arrangement
    # Hexagonal lattice spacing for unit hexagons: distance between centers = sqrt(3)
    spacing = np.sqrt(3)
    positions = []
    angles = []
    
    # Place 12 hexagons in a compact cluster
    # Row 0: 3 hexagons
    for k in range(-1, 2):
        positions.append([k * spacing, 0.0])
        angles.append(0.0)
    # Row 1: 3 hexagons
    for k in range(-1, 2):
        positions.append([k * spacing, spacing * 0.866])
        angles.append(0.0)
    # Row 2: 3 hexagons
    for k in range(-1, 2):
        positions.append([k * spacing, -spacing * 0.866])
        angles.append(0.0)
    # Row 3: 3 hexagons
    for k in range(-1, 2):
        positions.append([k * spacing, spacing * 1.732])
        angles.append(0.0)
    
    # Compute initial bounding hexagon
    all_pts = np.array(positions)
    cx = np.mean(all_pts[:, 0])
    cy = np.mean(all_pts[:, 1])
    
    # Estimate side length
    max_dist = np.max(np.linalg.norm(all_pts - [cx, cy], axis=1))
    side = max_dist / np.cos(np.pi / 6) + 0.1  # Add small margin
    
    # Flatten parameters
    x0 = np.array([p[0] for p in positions])
    y0 = np.array([p[1] for p in positions])
    a0 = np.array(angles)
    params0 = np.concatenate([x0, y0, a0])
    
    best_params = params0.copy()
    best_energy = energy(params0, (0.0, 0.0), side, 0.0)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            energy,
            params0,
            args=((0.0, 0.0), side, 0.0),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False}
        )
        
        if time.time() < deadline - 5.0:
            if result.fun < best_energy:
                best_params = result.x
                best_energy = result.fun
    except Exception:
        pass
    
    # Check if valid, if not, try with slightly larger side
    x = best_params[:N]
    y = best_params[N:2*N]
    ang = best_params[2*N:]
    
    inner = [[x[i], y[i], ang[i]] for i in range(N)]
    
    # Verify validity and adjust side if needed
    for trial_side in [side, side * 1.05, side * 1.1, side * 1.2, side * 1.5, 10.0, 20.0, 40.0]:
        if check_validity(inner, (0.0, 0.0), trial_side, 0.0):
            return inner, (0.0, 0.0), trial_side, 0.0
    
    # Fallback: return valid configuration with large side
    return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
