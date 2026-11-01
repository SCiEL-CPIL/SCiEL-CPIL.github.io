"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def compute_required_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    
    # Six outward normals of the outer hexagon
    normals = []
    for k in range(6):
        theta = oa + math.radians(30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    min_S = 0.0
    for cx, cy, ang_deg in inner:
        ang = math.radians(ang_deg)
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                d = (vx - ox) * nx + (vy - oy) * ny
                if d > 0:
                    S_needed = 2.0 * d / SQRT3
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def sat_overlap_check(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if overlapping."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            ex = v[(i+1) % n][0] - v[i][0]
            ey = v[(i+1) % n][1] - v[i][1]
            nx, ny = -ey, ex
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
    
    for (nx, ny) in axes:
        proj1 = [vx * nx + vy * ny for (vx, vy) in v1]
        proj2 = [vx * nx + vy * ny for (vx, vy) in v2]
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def check_all_disjoint(inner, tol=1e-8):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner)
    verts = []
    for cx, cy, ang_deg in inner:
        ang = math.radians(ang_deg)
        verts.append(get_hex_vertices(cx, cy, 1.0, ang))
    
    for i in range(n):
        for j in range(i+1, n):
            if sat_overlap_check(verts[i], verts[j], tol):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for cx, cy, ang_deg in inner:
        ang = math.radians(ang_deg)
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        for (px, py) in verts:
            inside = True
            for i in range(6):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % 6]
                ex = v2[0] - v1[0]
                ey = v2[1] - v1[1]
                nx, ny = -ey, ex
                d1 = nx * (px - v1[0]) + ny * (py - v1[1])
                v3 = outer_verts[(i+2) % 6]
                d2 = nx * (v3[0] - v1[0]) + ny * (v3[1] - v1[1])
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth approximation of required side length using log-sum-exp."""
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    
    normals = []
    for k in range(6):
        theta = oa + math.radians(30 + 60 * k)
        normals.append((math.cos(theta), math.sin(theta)))
    
    all_d = []
    for cx, cy, ang_deg in inner:
        ang = math.radians(ang_deg)
        verts = get_hex_vertices(cx, cy, 1.0, ang)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                d = (vx - ox) * nx + (vy - oy) * ny
                all_d.append(d)
    
    # Use log-sum-exp to get smooth max
    all_d = np.array(all_d)
    max_d = np.max(all_d)
    if max_d < 0:
        return 0.0
    # Smooth max approximation
    log_sum_exp = max_d + math.log(np.sum(np.exp((all_d - max_d) / beta))) * beta
    S = 2.0 * max(0, log_sum_exp) / SQRT3
    return S

def objective(params):
    """Objective function: smooth outer side + smooth overlap penalty."""
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = math.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = math.degrees(params[3*n+2]) % 360
    
    # Smooth containment cost
    S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta=20.0)
    
    # Smooth overlap penalty
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            dx = cx1 - cx2
            dy = cy1 - cy2
            d = math.sqrt(dx*dx + dy*dy)
            if d < 3.0:  # Only check close pairs
                v1 = get_hex_vertices(cx1, cy1, 1.0, math.radians(a1))
                v2 = get_hex_vertices(cx2, cy2, 1.0, math.radians(a2))
                if sat_overlap_check(v1, v2, tol=1e-8):
                    penalty += (2.0 - d) ** 2 * 50.0
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    def hex_lattice_positions(n_hex, scale=1.0, rotation_deg=0):
        """Generate positions for n_hex hexagons on a hexagonal lattice."""
        positions = [(0.0, 0.0)]
        rot = math.radians(rotation_deg)
        
        # Ring 1: 6 neighbors at distance sqrt(3)
        for k in range(6):
            angle = math.radians(60 * k) + rot
            x = SQRT3 * scale * math.cos(angle)
            y = SQRT3 * scale * math.sin(angle)
            positions.append((x, y))
            if len(positions) >= n_hex:
                break
        
        # Ring 2: 12 positions
        if len(positions) < n_hex:
            for k in range(6):
                angle = math.radians(60 * k) + rot
                x = 2 * SQRT3 * scale * math.cos(angle)
                y = 2 * SQRT3 * scale * math.sin(angle)
                positions.append((x, y))
                if len(positions) >= n_hex:
                    break
            for k in range(6):
                angle = math.radians(30 + 60 * k) + rot
                x = 2 * SQRT3 * scale * math.cos(angle)
                y = 2 * SQRT3 * scale * math.sin(angle)
                positions.append((x, y))
                if len(positions) >= n_hex:
                    break
        
        return positions[:n_hex]
    
    # Generate diverse initial conditions
    initial_conditions = []
    
    # Arrangement 1: Standard hexagonal lattice (1 + 6 + 4 from ring 2)
    positions1 = hex_lattice_positions(N, scale=1.0, rotation_deg=0)
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions1):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    ic1[3*N] = 0.0
    ic1[3*N+1] = 0.0
    ic1[3*N+2] = 0.0
    initial_conditions.append(ic1)
    
    # Arrangement 2: Rotated by 30 degrees
    positions2 = hex_lattice_positions(N, scale=1.0, rotation_deg=30)
    ic2 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions2):
        ic2[3*i] = px
        ic2[3*i+1] = py
        ic2[3*i+2] = 0.0
    ic2[3*N] = 0.0
    ic2[3*N+1] = 0.0
    ic2[3*N+2] = math.radians(30)
    initial_conditions.append(ic2)
    
    # Arrangement 3: Center + 6 at 1.732 + 4 at 3.464 (ring 2 corners)
    positions3 = [(0.0, 0.0)]
    for k in range(6):
        angle = math.radians(60 * k)
        positions3.append((SQRT3 * math.cos(angle), SQRT3 * math.sin(angle)))
    for k in range(4):
        angle = math.radians(60 * k)
        positions3.append((2 * SQRT3 * math.cos(angle), 2 * SQRT3 * math.sin(angle)))
    ic3 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions3[:N]):
        ic3[3*i] = px
        ic3[3*i+1] = py
        ic3[3*i+2] = 0.0
    ic3[3*N] = 0.0
    ic3[3*N+1] = 0.0
    ic3[3*N+2] = 0.0
    initial_conditions.append(ic3)
    
    # Arrangement 4: Perturbed lattice
    np.random.seed(123)
    ic4 = ic1.copy()
    ic4[:3*N] += np.random.randn(3*N) * 0.1
    initial_conditions.append(ic4)
    
    # Arrangement 5: Different outer rotation
    ic5 = ic1.copy()
    ic5[3*N+2] = math.radians(15)
    initial_conditions.append(ic5)
    
    # Arrangement 6: Tighter packing attempt
    positions6 = [(0.0, 0.0)]
    for k in range(6):
        angle = math.radians(60 * k)
        positions6.append((1.7 * math.cos(angle), 1.7 * math.sin(angle)))
    for k in range(4):
        angle = math.radians(30 + 60 * k)
        positions6.append((3.2 * math.cos(angle), 3.2 * math.sin(angle)))
    ic6 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(positions6[:N]):
        ic6[3*i] = px
        ic6[3*i+1] = py
        ic6[3*i+2] = 0.0
    ic6[3*N] = 0.0
    ic6[3*N+1] = 0.0
    ic6[3*N+2] = 0.0
    initial_conditions.append(ic6)
    
    best_result = None
    best_score = float('inf')
    
    # Optimize each initial condition with multiple methods
    for ic in initial_conditions:
        if time.time() > deadline - 25:
            break
        
        for method in ['L-BFGS-B', 'SLSQP']:
            if time.time() > deadline - 25:
                break
            
            try:
                maxiter = 300 if method == 'L-BFGS-B' else 150
                res = minimize(objective, ic, method=method, 
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
                
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = math.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = math.degrees(params[3*N+2]) % 360
                
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
            except Exception:
                pass
    
    # Fallback: use the best lattice arrangement
    if best_result is None:
        positions_fb = hex_lattice_positions(N, scale=1.0, rotation_deg=0)
        inner = [[px, py, 0.0] for px, py in positions_fb]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.01
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Final refinement with tight tolerances
    if time.time() < deadline - 30:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = math.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = math.radians(best_result[3])
            
            res = minimize(objective, params_best, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            params = res.x
            inner = []
            for i in range(N):
                cx = params[3*i]
                cy = params[3*i+1]
                ang = math.degrees(params[3*i+2]) % 360
                inner.append([cx, cy, ang])
            ox = params[3*N]
            oy = params[3*N+1]
            outer_angle = math.degrees(params[3*N+2]) % 360
            S = compute_required_side(inner, (ox, oy), outer_angle)
            
            if check_validity(inner, (ox, oy), S, outer_angle) and S < best_score:
                best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
