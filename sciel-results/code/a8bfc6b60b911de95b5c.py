"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons.
    inner_hexagons: list of (cx, cy, angle_deg)
    outer_center: (x, y)
    outer_angle: rotation of outer hexagon in degrees
    Returns the minimum side length."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    # The 6 side normals of the outer hexagon
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                d = np.dot(v - np.array([ox, oy]), n)
                if d > 0:
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    # Check no overlaps between inner hexagons using SAT
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            # SAT check
            axes = []
            for v in (v1, v2):
                n = len(v)
                for k in range(n):
                    edge = v[(k+1) % n] - v[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-12:
                        axes.append(normal / norm)
            
            overlap = True
            for axis in axes:
                proj1 = np.dot(v1, axis)
                proj2 = np.dot(v2, axis)
                if proj1.max() < proj2.min() - 1e-8 or proj2.max() < proj1.min() - 1e-8:
                    overlap = False
                    break
            if overlap:
                return False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            # Check if point is inside outer hexagon
            inside = True
            n = len(outer_verts)
            for i in range(n):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i+2) % n] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def objective(params):
    """Objective function: outer side length + penalty for overlaps."""
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    # Compute required outer side length
    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                axes = []
                for v in (v1, v2):
                    nv = len(v)
                    for k in range(nv):
                        edge = v[(k+1) % nv] - v[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            axes.append(normal / norm)
                
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    gap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                    if gap < 0:
                        penalty += gap * gap * 100
                        break
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # For N=11, use a hexagonal arrangement: center + ring of 6 + 4 from next ring
    # The optimal arrangement for 11 hexagons is likely:
    # 1 center, 6 in first ring (distance sqrt(3) from center), 4 in second ring
    
    # Hexagonal lattice positions
    # For unit hexagons (side 1), apothem = sqrt(3)/2, so adjacent centers are sqrt(3) apart
    
    # Create initial hexagonal arrangement
    positions = [(0, 0)]
    # Ring 1: 6 positions at distance sqrt(3)
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        positions.append((x, y))
    # Ring 2: add 4 more (we only need 4 more for total 11)
    for k in range(4):
        angle = np.radians(60 * k + 30)  # Offset by 30 degrees for ring 2
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        positions.append((x, y))
    
    positions = positions[:N]
    
    # Initial parameters: [cx, cy, angle_rad] * N, then [ox, oy, outer_angle_rad]
    x0 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions):
        x0[3*i] = px
        x0[3*i+1] = py
        x0[3*i+2] = 0.0
    x0[3*N] = 0.0
    x0[3*N+1] = 0.0
    x0[3*N+2] = 0.0
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple initial configurations
    initial_conditions = []
    
    # Condition 1: Hexagonal lattice as above
    initial_conditions.append(x0.copy())
    
    # Condition 2: Slightly perturbed
    np.random.seed(42)
    ic2 = x0 + np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic2)
    
    # Condition 3: Different arrangement - linear
    ic3 = np.zeros(3*N+3)
    for i in range(N):
        ic3[3*i] = i * np.sqrt(3)
        ic3[3*i+1] = 0
        ic3[3*i+2] = 0
    ic3[3*N] = (N-1) * np.sqrt(3) / 2
    ic3[3*N+1] = 0
    ic3[3*N+2] = 0
    initial_conditions.append(ic3)
    
    # Condition 4: Two rows
    ic4 = np.zeros(3*N+3)
    for i in range(N):
        row = i // 6
        col = i % 6
        ic4[3*i] = col * np.sqrt(3) + (row % 2) * np.sqrt(3)/2
        ic4[3*i+1] = row * 1.5
        ic4[3*i+2] = 0
    ic4[3*N] = 2.5 * np.sqrt(3)
    ic4[3*N+1] = 0.75
    ic4[3*N+2] = 0
    initial_conditions.append(ic4)
    
    # Condition 5: Tighter hexagonal arrangement
    ic5 = np.zeros(3*N+3)
    # Try a more compact arrangement
    compact_positions = [
        (0, 0),
        (np.sqrt(3), 0),
        (-np.sqrt(3), 0),
        (0, np.sqrt(3)),
        (0, -np.sqrt(3)),
        (np.sqrt(3)/2, np.sqrt(3)*1.5),
        (-np.sqrt(3)/2, np.sqrt(3)*1.5),
        (np.sqrt(3)/2, -np.sqrt(3)*1.5),
        (-np.sqrt(3)/2, -np.sqrt(3)*1.5),
        (np.sqrt(3), np.sqrt(3)*1.5),
        (-np.sqrt(3), np.sqrt(3)*1.5),
    ]
    for i, (px, py) in enumerate(compact_positions[:N]):
        ic5[3*i] = px
        ic5[3*i+1] = py
        ic5[3*i+2] = 0
    ic5[3*N] = 0
    ic5[3*N+1] = 0
    ic5[3*N+2] = 0
    initial_conditions.append(ic5)
    
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(objective, ic, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                
                # Compute the actual outer side length
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                
                # Verify validity
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
        except Exception as e:
            pass
    
    # If no valid result found, use a fallback
    if best_result is None:
        inner = []
        for i, (px, py) in enumerate(positions):
            inner.append([px, py, 0.0])
        ox, oy = 0.0, 0.0
        outer_angle = 0.0
        S = compute_outer_side_length(inner, (ox, oy), outer_angle)
        S += 0.1
        best_result = (inner, (ox, oy), S, outer_angle)
    
    # Time check
    if time.time() > deadline:
        return best_result
    
    # Try to refine the best result if time permits
    if time.time() < deadline - 40:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(objective, params_best, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Try a few more random restarts if time permits
    if time.time() < deadline - 20:
        np.random.seed(123)
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            
            # Random initial condition near the best solution
            ic_rand = np.zeros(3*N+3)
            for i, (px, py) in enumerate(positions):
                ic_rand[3*i] = px + np.random.randn() * 0.3
                ic_rand[3*i+1] = py + np.random.randn() * 0.3
                ic_rand[3*i+2] = np.random.randn() * 0.1
            ic_rand[3*N] = np.random.randn() * 0.2
            ic_rand[3*N+1] = np.random.randn() * 0.2
            ic_rand[3*N+2] = np.random.randn() * 0.1
            
            try:
                res = minimize(objective, ic_rand, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < best_score:
                    params = res.x
                    inner = []
                    for i in range(N):
                        cx = params[3*i]
                        cy = params[3*i+1]
                        ang = np.degrees(params[3*i+2]) % 360
                        inner.append([cx, cy, ang])
                    ox = params[3*N]
                    oy = params[3*N+1]
                    outer_angle = np.degrees(params[3*N+2]) % 360
                    
                    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                    
                    if check_validity(inner, (ox, oy), S, outer_angle):
                        if S < best_score:
                            best_score = S
                            best_result = (inner, (ox, oy), S, outer_angle)
            except Exception:
                pass
    
    return best_result

# EVOLVE-BLOCK-END
