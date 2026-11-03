import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0

def get_hex_vertices(cx, cy, side, angle_deg):
    angle = math.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def project(vertices, nx, ny):
    projs = np.dot(vertices, [nx, ny])
    return np.min(projs), np.max(projs)

def are_disjoint_sat(v1, v2, eps=1e-9):
    # SAT check for two convex polygons
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = p2 - p1
        nx = -edge[1]
        ny = edge[0]
        min1, max1 = project(v1, nx, ny)
        min2, max2 = project(v2, nx, ny)
        if max1 < min2 + eps or max2 < min1 + eps:
            return True
    
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = p2 - p1
        nx = -edge[1]
        ny = edge[0]
        min1, max1 = project(v1, nx, ny)
        min2, max2 = project(v2, nx, ny)
        if max1 < min2 + eps or max2 < min1 + eps:
            return True
        
    return False

def compute_outer_side(centers, angles, ocx, ocy, otheta):
    """
    Computes the side length of the outer hexagon (center ocx, ocy, angle otheta)
    that contains all inner hexagons (centers, angles, side=1).
    """
    max_proj = 0.0
    otheta_rad = math.radians(otheta)
    # Normals for the 6 sides of the outer hexagon
    # A regular hexagon with "pointy top" or "flat top"?
    # Standard: vertices at theta + k*pi/3.
    # Normals are perpendicular to edges.
    # Edge k is between vertex k and k+1.
    # Midpoint angle of edge k is theta + (k + 0.5)*pi/3.
    # Normal points outward.
    for k in range(6):
        normal_angle = otheta_rad + math.pi / 6 + k * math.pi / 3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        
        # Project all inner vertices onto this normal
        # Optimization: We only need the max projection for each inner hex
        for i in range(len(centers)):
            cx, cy = centers[i]
            angle_rad = math.radians(angles[i])
            # Vertices of inner hex i
            # We can precompute relative vertices if angles are fixed, but they change in SA.
            for vi in range(6):
                vert_angle = angle_rad + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = (vx - ocx) * nx + (vy - ocy) * ny
                if proj > max_proj:
                    max_proj = proj
                    
    # The distance from center to side is R * cos(pi/6) = R * sqrt(3)/2
    # Here max_proj is the distance from center to the furthest point in that direction.
    # For the hexagon to contain the point, the side must be at least max_proj.
    # So, apothem a = max_proj.
    # Side length s = 2 * a / sqrt(3)
    return 2.0 * max_proj / math.sqrt(3)

def objective_full(params):
    """
    Objective for the full optimization.
    params: [cx1, cy1, a1, ..., cx12, cy12, a12, ocx, ocy, otheta]
    """
    if time.time() > DEADLINE - 5:
        return 1e9

    centers = []
    angles = []
    for i in range(12):
        centers.append([params[i * 3], params[i * 3 + 1]])
        angles.append(params[i * 3 + 2])
    
    ocx, ocy = params[36], params[37]
    otheta = params[38]
    
    # Check disjointness
    verts = []
    for i in range(12):
        v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        verts.append(v)
        
    for i in range(12):
        for j in range(i + 1, 12):
            if not are_disjoint_sat(verts[i], verts[j]):
                # Return penalty if overlapping
                # Estimate overlap penalty based on center distance
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Min distance for disjoint unit hexes is sqrt(3)
                penalty = max(0, (math.sqrt(3) - dist)) * 100
                return penalty + compute_outer_side(centers, angles, ocx, ocy, otheta)
                
    return compute_outer_side(centers, angles, ocx, ocy, otheta)

def generate_hexagonal_cluster():
    d = math.sqrt(3) + 0.01
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])
    
    centers = [np.array([0.0, 0.0])]
    ring1_vectors = [u, v, v - u, -u, -v, u - v]
    for vec in ring1_vectors:
        centers.append(vec.copy())
        
    ring2_candidates = []
    basis = [u, v, v - u, -u, -v, u - v]
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers.append(ring2_candidates[i].copy())
        
    angles = [30.0] * 12
    return centers, angles

def generate_rectangular_cluster():
    # 3 rows of 4
    d = math.sqrt(3) + 0.01
    dy = 1.5 + 0.01 # Height of hex is sqrt(3), distance between rows in rect packing is 1.5 + ...?
    # Actually, for flat-topped hexes, horizontal dist is 2, vertical is sqrt(3).
    # For pointy-topped, horizontal is sqrt(3), vertical is 1.5.
    # Let's use a simple grid and let optimization fix it.
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * 1.8
            y = r * 1.6
            if r % 2 == 1:
                x += 0.9
            centers.append([x, y])
    angles = [0.0] * 12
    return centers, angles

def construct():
    deadline = time.time() + DEADLINE
    
    # 1. Generate initial configurations
    configs = []
    
    # Hex lattice
    c1, a1 = generate_hexagonal_cluster()
    configs.append((c1, a1))
    
    # Rectangular
    c2, a2 = generate_rectangular_cluster()
    configs.append((c2, a2))
    
    # Random
    random.seed(42)
    c3 = [[random.uniform(-3, 3), random.uniform(-3, 3)] for _ in range(12)]
    a3 = [random.uniform(0, 60) for _ in range(12)]
    configs.append((c3, a3))
    
    best_result = None
    best_val = float('inf')
    
    for centers, angles in configs:
        if time.time() > deadline - 30:
            break
            
        # Convert to param vector
        params = []
        for c, a in zip(centers, angles):
            params.extend([c[0], c[1], a])
        params.extend([0.0, 0.0, 0.0]) # ocx, ocy, otheta
        
        # Simulated Annealing
        current_params = params
        current_val = objective_full(current_params)
        
        if current_val < best_val:
            best_val = current_val
            best_result = current_params[:]
            
        T = 1.0
        T_min = 0.01
        alpha = 0.9995
        steps = 5000
        
        for step in range(steps):
            if time.time() > deadline - 15:
                break
                
            if T < T_min:
                break
                
            # Perturb
            new_params = current_params[:]
            # Move one hexagon
            idx = random.randint(0, 11)
            base = idx * 3
            new_params[base] += random.gauss(0, 0.2)
            new_params[base+1] += random.gauss(0, 0.2)
            new_params[base+2] += random.gauss(0, 5.0) # Angle in degrees
            
            # Occasionally perturb outer hex
            if random.random() < 0.1:
                new_params[38] += random.gauss(0, 2.0)
                
            new_val = objective_full(new_params)
            
            # Acceptance
            if new_val < current_val or random.random() < math.exp((current_val - new_val) / T):
                current_params = new_params
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_result = current_params[:]
                    
            T *= alpha
            
        # Local Polish with Nelder-Mead
        if time.time() < deadline - 10:
            try:
                res = minimize(objective_full, current_params, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5})
                if res.fun < current_val:
                    current_val = res.fun
                    current_params = res.x
                    if current_val < best_val:
                        best_val = current_val
                        best_result = current_params[:]
            except Exception:
                pass
    
    if best_result is None:
        # Fallback
        c1, a1 = generate_hexagonal_cluster()
        best_result = []
        for c, a in zip(c1, a1):
            best_result.extend([c[0], c[1], a])
        best_result.extend([0.0, 0.0, 0.0])
        
    # Final Optimization of Outer Hexagon
    # Extract inner params
    centers = []
    angles = []
    for i in range(12):
        centers.append([best_result[i * 3], best_result[i * 3 + 1]])
        angles.append(best_result[i * 3 + 2])
        
    ocx, ocy, otheta = best_result[36], best_result[37], best_result[38]
    
    # Grid search for outer hex orientation and center
    best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
    
    # Coarse search
    for da in np.linspace(-15, 15, 11):
        for dx in np.linspace(-1, 1, 5):
            for dy in np.linspace(-1, 1, 5):
                s = compute_outer_side(centers, angles, ocx + dx, ocy + dy, otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx, best_ocy, best_otheta = ocx + dx, ocy + dy, otheta + da
                    
    # Fine search
    for da in np.linspace(-2, 2, 9):
        for dx in np.linspace(-0.2, 0.2, 5):
            for dy in np.linspace(-0.2, 0.2, 5):
                s = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, best_otheta + da)
                if s < best_side:
                    best_side = s
                    best_ocx, best_ocy, best_otheta = best_ocx + dx, best_ocy + dy, best_otheta + da

    # Verify validity one last time
    verts = []
    for i in range(12):
        v = get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        verts.append(v)
    valid = True
    for i in range(12):
        for j in range(i + 1, 12):
            if not are_disjoint_sat(verts[i], verts[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid configuration (Hex lattice)
        c1, a1 = generate_hexagonal_cluster()
        centers = [[c[0], c[1]] for c in c1]
        angles = a1
        best_ocx, best_ocy, best_otheta = 0.0, 0.0, 0.0
        best_side = compute_outer_side(centers, angles, best_ocx, best_ocy, best_otheta)

    inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_side
    outer_angle = best_otheta
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
