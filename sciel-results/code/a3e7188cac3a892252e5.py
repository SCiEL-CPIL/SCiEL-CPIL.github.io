"""Optimized hexagon packing: two-phase compact-then-contain approach for N unit hexagons."""
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

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i+1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False, axis
    return True, None

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
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
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
            if overlap:
                return False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            n = len(outer_verts)
            inside = True
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

def max_pairwise_distance(centers):
    """Compute maximum pairwise distance between centers."""
    n = len(centers)
    max_d = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
            if d > max_d:
                max_d = d
    return max_d

def compactness_objective(params, deadline):
    """Phase 1: Minimize max pairwise distance + collision penalty."""
    if time.time() > deadline:
        return float('inf')
    
    n = N
    centers = []
    angles = []
    for i in range(n):
        cx = params[2*i]
        cy = params[2*i+1]
        centers.append((cx, cy))
        angles.append(np.degrees(params[2*n + i]) % 360)
    
    # Max pairwise distance
    max_d = max_pairwise_distance(centers)
    
    # Collision penalty
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1 = centers[i]
            cx2, cy2 = centers[j]
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.5:
                a1 = angles[i]
                a2 = angles[j]
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
                if overlap:
                    penalty += (2.0 - d) ** 2 * 200
    
    return max_d + penalty

def outer_objective(params, inner_fixed, deadline):
    """Phase 2: Minimize outer side length given fixed inner configuration."""
    if time.time() > deadline:
        return float('inf')
    
    ox, oy, oa = params
    S = compute_outer_side_length(inner_fixed, (ox, oy), oa)
    return S

def joint_objective(params, deadline):
    """Phase 3: Joint optimization of all parameters."""
    if time.time() > deadline:
        return float('inf')
    
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
    
    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.5:
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
                if overlap:
                    penalty += (2.0 - d) ** 2 * 100
    
    return S + penalty

def construct():
    """Main entry point: two-phase compact-then-contain optimization."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Hexagonal lattice positions for N=11
    hex_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
        if len(hex_positions) >= N:
            break
    hex_positions = hex_positions[:N]
    
    best_result = None
    best_score = float('inf')
    
    # ========== PHASE 1: Compact the cluster ==========
    # Minimize max pairwise distance with collision constraints
    # Parameters: 2*N centers + N angles = 3*N params
    compact_inits = []
    
    # Init 1: Hexagonal lattice
    ic1 = np.zeros(3 * N)
    for i, (px, py) in enumerate(hex_positions):
        ic1[2*i] = px
        ic1[2*i+1] = py
        ic1[2*N + i] = 0.0
    compact_inits.append(ic1)
    
    # Init 2: Perturbed
    np.random.seed(42)
    ic2 = ic1 + np.random.randn(3*N) * 0.15
    compact_inits.append(ic2)
    
    # Init 3: Tighter scale
    ic3 = np.zeros(3 * N)
    for i, (px, py) in enumerate(hex_positions):
        ic3[2*i] = px * 0.98
        ic3[2*i+1] = py * 0.98
        ic3[2*N + i] = 0.0
    compact_inits.append(ic3)
    
    # Init 4: Rotated angles
    ic4 = ic1.copy()
    for i in range(N):
        ic4[2*N + i] = np.radians(30)
    compact_inits.append(ic4)
    
    # Init 5: Different perturbation
    np.random.seed(123)
    ic5 = ic1 + np.random.randn(3*N) * 0.08
    compact_inits.append(ic5)
    
    compacted_configs = []
    
    for ic in compact_inits:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(compactness_objective, ic, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < float('inf'):
                centers = []
                angles = []
                for i in range(N):
                    centers.append((res.x[2*i], res.x[2*i+1]))
                    angles.append(np.degrees(res.x[2*N + i]) % 360)
                compacted_configs.append((centers, angles, res.fun))
        except Exception:
            pass
    
    # ========== PHASE 2: Optimize outer container for each compacted config ==========
    # For each compacted configuration, find best outer hexagon
    candidates = []
    
    # Also include original lattice as a candidate
    orig_inner = [(px, py, 0.0) for px, py in hex_positions]
    candidates.append((orig_inner, "original"))
    
    for centers, angles, compact_score in compacted_configs:
        if time.time() > deadline - 40:
            break
        inner_config = [(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        candidates.append((inner_config, "compacted"))
    
    # For each candidate, optimize outer parameters
    for inner_config, label in candidates:
        if time.time() > deadline - 30:
            break
        
        # Try multiple outer initial conditions
        outer_inits = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.radians(30)],
            [0.1, 0.1, 0.0],
            [-0.1, 0.05, np.radians(15)],
            [0.0, 0.0, np.radians(60)],
        ]
        
        for oi in outer_inits:
            if time.time() > deadline - 25:
                break
            try:
                res = minimize(outer_objective, oi, args=(inner_config, deadline),
                             method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
                
                if res.fun < best_score and res.fun > 0:
                    ox, oy, oa = res.x
                    outer_angle = np.degrees(oa) % 360
                    S = res.fun
                    
                    if check_validity(inner_config, (ox, oy), S, outer_angle):
                        best_score = S
                        best_result = (inner_config, (ox, oy), S, outer_angle)
            except Exception:
                pass
    
    # ========== PHASE 3: Joint refinement ==========
    if best_result is not None and time.time() < deadline - 20:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            # Multiple joint refinement attempts
            for attempt in range(3):
                if time.time() > deadline - 15:
                    break
                try:
                    if attempt == 0:
                        ic = params_best
                    elif attempt == 1:
                        np.random.seed(99)
                        ic = params_best + np.random.randn(3*N+3) * 0.03
                    else:
                        np.random.seed(77)
                        ic = params_best + np.random.randn(3*N+3) * 0.01
                    
                    res = minimize(joint_objective, ic, args=(deadline,), method='L-BFGS-B',
                                  options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
                    
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
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        inner = [[px, py, 0.0] for px, py in hex_positions]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result

# EVOLVE-BLOCK-END
