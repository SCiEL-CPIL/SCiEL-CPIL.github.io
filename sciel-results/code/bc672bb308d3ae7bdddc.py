"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def get_hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    angles = a + math.pi / 6 + np.arange(6) * math.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def sat_overlap_check(h1, h2, tol=1e-9):
    """Check if two convex polygons overlap using SAT.
    h1, h2: arrays of shape (6, 2) - vertices of the hexagons."""
    # Get all edges from both polygons
    edges = []
    for poly in [h1, h2]:
        for i in range(6):
            j = (i + 1) % 6
            edges.append(poly[j] - poly[i])
    
    for edge in edges:
        # Normal to the edge
        normal = np.array([-edge[1], edge[0]])
        if np.linalg.norm(normal) < 1e-12:
            continue
        normal = normal / np.linalg.norm(normal)
        
        # Project both polygons onto the normal
        proj1 = h1 @ normal
        proj2 = h2 @ normal
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for gap
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    
    return True

def check_all_disjoint(positions, angles, tol=1e-9):
    """Check that all hexagons are pairwise disjoint."""
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    
    for i in range(n):
        for j in range(i + 1, n):
            # Quick rejection
            dist = math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
            if dist > 2.0 + tol:
                continue
            if sat_overlap_check(verts_list[i], verts_list[j], tol):
                return False
    return True

def compute_required_side(ocx, ocy, oor_deg, positions, angles):
    """Compute the minimum outer hexagon side length to contain all inner hexagons.
    Outer hexagon: center (ocx, ocy), side S, rotation oor_deg."""
    max_val = 0.0
    oor = math.radians(oor_deg)
    # The 6 constraint directions for the outer hexagon
    thetas = oor + math.pi / 6 + np.arange(6) * math.pi / 3
    
    for i in range(len(positions)):
        verts = get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for vx, vy in verts:
            # Transform to outer hexagon's local frame
            dx, dy = vx - ocx, vy - ocy
            cos_a, sin_a = math.cos(-oor), math.sin(-oor)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            for k in range(6):
                theta = math.pi / 6 + k * math.pi / 3
                val = lx * math.cos(theta) + ly * math.sin(theta)
                if val > max_val:
                    max_val = val
    
    return max_val / (math.sqrt(3) / 2)

def compute_required_side_smooth(ocx, ocy, oor_deg, positions, angles, beta=50.0):
    """Smooth approximation of required side length using log-sum-exp."""
    oor = math.radians(oor_deg)
    all_vals = []
    
    for i in range(len(positions)):
        verts = get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for vx, vy in verts:
            dx, dy = vx - ocx, vy - ocy
            cos_a, sin_a = math.cos(-oor), math.sin(-oor)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            for k in range(6):
                theta = math.pi / 6 + k * math.pi / 3
                val = lx * math.cos(theta) + ly * math.sin(theta)
                all_vals.append(val)
    
    all_vals = np.array(all_vals)
    max_val = all_vals.max()
    # Log-sum-exp for smooth max
    if beta > 0:
        lse = max_val + math.log(np.sum(np.exp(beta * (all_vals - max_val)))) / beta
    else:
        lse = max_val
    
    return lse / (math.sqrt(3) / 2)

def check_inside(positions, angles, oc, side, oor_deg, tol=1e-9):
    """Check if all inner hexagons are inside the outer hexagon."""
    ocx, ocy = oc
    oor = math.radians(oor_deg)
    
    for i in range(len(positions)):
        verts = get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for vx, vy in verts:
            dx, dy = vx - ocx, vy - ocy
            cos_a, sin_a = math.cos(-oor), math.sin(-oor)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            for k in range(6):
                theta = math.pi / 6 + k * math.pi / 3
                if lx * math.cos(theta) + ly * math.sin(theta) > side * math.sqrt(3) / 2 + tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    sqrt3 = math.sqrt(3)
    
    # Generate hex grid positions for N items
    def hex_grid_positions(n):
        positions = [(0.0, 0.0)]
        ring = 1
        while len(positions) < n:
            for q in range(-ring, ring + 1):
                for r in range(-ring, ring + 1):
                    if max(abs(q), abs(r), abs(q + r)) == ring:
                        x = 1.5 * q
                        y = sqrt3 * (r + q / 2.0)
                        if len(positions) < n:
                            positions.append((x, y))
            ring += 1
        return positions[:n]
    
    # Fixed inner hexagons in hexagonal lattice
    positions = hex_grid_positions(N)
    angles = [0.0] * N
    
    # Verify initial validity
    if not check_all_disjoint(positions, angles):
        # Fallback: linear arrangement
        positions = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N
    
    # Optimize outer hexagon: center (ocx, ocy) and rotation (oor)
    # Only 3 variables instead of 3*N
    
    def objective(x):
        if time.time() > deadline - 5.0:
            return 1e6
        ocx, ocy, oor = x
        return compute_required_side_smooth(ocx, ocy, oor, positions, angles, beta=50.0)
    
    best_side = 1e10
    best_oc = (0.0, 0.0)
    best_oor = 0.0
    
    # Try multiple initial rotations to exploit symmetry
    for oor_init in [0.0, 30.0, 60.0, 15.0, 45.0]:
        if time.time() > deadline - 20.0:
            break
        
        x0 = [0.0, 0.0, oor_init]
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_side:
                ocx, ocy, oor = res.x
                exact_side = compute_required_side(ocx, ocy, oor, positions, angles)
                if check_inside(positions, angles, (ocx, ocy), exact_side, oor) and check_all_disjoint(positions, angles):
                    best_side = exact_side
                    best_oc = (ocx, ocy)
                    best_oor = oor
        except Exception:
            pass
    
    # Polish with sharper objective
    if time.time() < deadline - 15.0:
        def objective_sharp(x):
            if time.time() > deadline - 2.0:
                return 1e6
            ocx, ocy, oor = x
            return compute_required_side_smooth(ocx, ocy, oor, positions, angles, beta=200.0)
        
        x0 = [best_oc[0], best_oc[1], best_oor]
        try:
            res = minimize(objective_sharp, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_side:
                ocx, ocy, oor = res.x
                exact_side = compute_required_side(ocx, ocy, oor, positions, angles)
                if check_inside(positions, angles, (ocx, ocy), exact_side, oor) and check_all_disjoint(positions, angles):
                    best_side = exact_side
                    best_oc = (ocx, ocy)
                    best_oor = oor
        except Exception:
            pass
    
    # Add tiny epsilon to ensure strict containment
    best_side *= (1.0 + 1e-8)
    
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    return inner, best_oc, best_side, best_oor
# EVOLVE-BLOCK-END
