"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import time
import math
import numpy as np

N = 11
DEADLINE = 115.0  # safety margin before 120s limit

def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation in degrees."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + math.radians(60 * k)
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return np.array(verts)

def _hexagon_contains_hexagon(outer_center, outer_side, outer_angle, inner_center, inner_side, inner_angle):
    """Check if inner hexagon is fully inside outer hexagon using separating axis theorem."""
    ov = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    iv = _hex_vertices(inner_center[0], inner_center[1], inner_side, inner_angle)
    
    # Edges of both hexagons give 12 axes
    axes = []
    for i in range(6):
        for vset in (ov, iv):
            e = vset[(i + 1) % 6] - vset[i]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    for ax in axes:
        o_proj = np.dot(ov, ax)
        i_proj = np.dot(iv, ax)
        if o_proj.min() > i_proj.max() or o_proj.max() < i_proj.min():
            return False
    return True

def _hexagons_disjoint(h1_center, h1_angle, h2_center, h2_angle, side=1.0, tol=1e-9):
    """Check if two unit hexagons are disjoint (no interior overlap)."""
    v1 = _hex_vertices(h1_center[0], h1_center[1], side, h1_angle)
    v2 = _hex_vertices(h2_center[0], h2_center[1], side, h2_angle)
    
    axes = []
    for i in range(6):
        for vset in (v1, v2):
            e = vset[(i + 1) % 6] - vset[i]
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    for ax in axes:
        p1 = np.dot(v1, ax)
        p2 = np.dot(v2, ax)
        # Disjoint if separated by at least tol
        if p1.max() + tol < p2.min() or p2.max() + tol < p1.min():
            return True
    return False

def _min_enclosing_hex_side(vertices, center):
    """Find the minimum side length of a regular hexagon (centered at given center, any rotation) 
    that encloses all vertices. Returns (side, angle_deg)."""
    rel = vertices - np.array(center)
    best_r = float('inf')
    best_phi = 0.0
    # Search over rotation in fine steps
    for phi_deg in range(0, 30, 1):  # 30 degree periodicity, 1 degree steps
        phi = math.radians(phi_deg)
        r_max = 0.0
        for k in range(6):
            angle = math.radians(30 + 60 * k) + phi
            n = np.array([math.cos(angle), math.sin(angle)])
            proj = np.max(np.dot(rel, n))
            r_max = max(r_max, proj)
        if r_max < best_r:
            best_r = r_max
            best_phi = phi_deg
    # Refine around best
    for phi_deg in np.linspace(best_phi - 1, best_phi + 1, 101):
        phi = math.radians(phi_deg)
        r_max = 0.0
        for k in range(6):
            angle = math.radians(30 + 60 * k) + phi
            n = np.array([math.cos(angle), math.sin(angle)])
            proj = np.max(np.dot(rel, n))
            r_max = max(r_max, proj)
        if r_max < best_r:
            best_r = r_max
            best_phi = phi_deg % 30
    side = 2.0 * best_r / math.sqrt(3.0)
    return side, best_phi

def construct():
    start_time = time.time()
    
    # Use the known optimal "flower" configuration as the starting point
    # This is much better than random or grid starts
    eps = 1e-9
    sqrt3 = math.sqrt(3)
    
    # Ring 0: center
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 hexagons at distance sqrt(3) + eps
    for k in range(6):
        angle = math.radians(k * 60)
        r = sqrt3 + eps
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 4 hexagons at distance 3 + 2*eps, angles 30, 90, 150, 270
    for k in range(4):
        angle = math.radians(30 + k * 60)
        r = 3.0 + 2 * eps
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        centers.append((x, y))
    
    # All inner hexagons rotated 30 degrees (flat-top)
    inner_angles = [30.0] * N
    
    # Initial params
    params0 = []
    for i in range(N):
        params0.append(centers[i][0])
        params0.append(centers[i][1])
        params0.append(inner_angles[i])
    params0 = np.array(params0, dtype=float)
    
    best_params = params0.copy()
    
    # Compute initial score
    def objective(params):
        if time.time() - start_time > DEADLINE:
            return float('inf')
        
        # Extract positions and angles
        pos = []
        ang = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            pos.append((x, y))
            ang.append(a)
        
        # Check disjointness
        for i in range(N):
            for j in range(i + 1, N):
                if not _hexagons_disjoint(pos[i], ang[i], pos[j], ang[j]):
                    # Penalty for overlap
                    penalty = 0.0
                    v1 = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
                    v2 = _hex_vertices(pos[j][0], pos[j][1], 1.0, ang[j])
                    # Check all vertex-edge pairs
                    for vi in v1:
                        for k in range(6):
                            e1 = v2[(k+1)%6] - v2[k]
                            n = np.array([-e1[1], e1[0]])
                            norm = np.linalg.norm(n)
                            if norm > 1e-12:
                                n = n / norm
                                d = np.dot(vi - v2[k], n)
                                if d < 0:
                                    penalty += -d
                    for vi in v2:
                        for k in range(6):
                            e1 = v1[(k+1)%6] - v1[k]
                            n = np.array([-e1[1], e1[0]])
                            norm = np.linalg.norm(n)
                            if norm > 1e-12:
                                n = n / norm
                                d = np.dot(vi - v1[k], n)
                                if d < 0:
                                    penalty += -d
                    return 10.0 * penalty + 100.0  # Large penalty
        
        # Compute min enclosing hexagon
        all_vertices = []
        for i in range(N):
            v = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
            all_vertices.append(v)
        
        # Find center as centroid of all vertices
        all_v = np.vstack(all_vertices)
        center = np.mean(all_v, axis=0)
        
        side, angle = _min_enclosing_hex_side(all_v, center)
        
        # Verify containment
        for i in range(N):
            if not _hexagon_contains_hexagon((center[0], center[1]), side, angle, 
                                             pos[i], 1.0, ang[i]):
                return side + 10.0  # Penalty if not contained
        
        return side
    
    best_score = objective(best_params)
    
    # Nelder-Mead optimization with restarts
    from scipy.optimize import minimize
    
    restarts = 3
    for restart in range(restarts):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if restart == 0:
            p = params0.copy()
        else:
            # Perturb initial
            p = params0 + 0.3 * np.random.randn(3 * N)
        
        try:
            result = minimize(
                objective, p, method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Final refinement with tighter tolerance
    if time.time() - start_time < DEADLINE - 10:
        try:
            result = minimize(
                objective, best_params, method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final solution
    pos = []
    ang = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        a = best_params[3*i + 2]
        pos.append((x, y))
        ang.append(a)
    
    # Compute final enclosing hexagon
    all_vertices = []
    for i in range(N):
        v = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
        all_vertices.append(v)
    all_v = np.vstack(all_vertices)
    center = np.mean(all_v, axis=0)
    side, angle = _min_enclosing_hex_side(all_v, center)
    
    # Build inner list: (x, y, angle_degrees)
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
    
    return inner, (center[0], center[1]), side, angle
