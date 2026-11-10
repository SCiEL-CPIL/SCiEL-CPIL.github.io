import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Returns (inner, outer, side) where:
      inner: (11,3) array of (x, y, angle_deg)
      outer: (3,) array of (x, y, angle_deg)
      side: float side length of outer hexagon
    """
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    N = 11
    sqrt3 = np.sqrt(3.0)
    
    def hex_vertices(cx, cy, s, a_deg):
        a = np.radians(a_deg)
        angles = a + np.arange(6) * np.pi / 3.0
        return np.column_stack((cx + s * np.cos(angles), cy + s * np.sin(angles)))
    
    def edge_normals(vertices):
        """Return outward unit normals for each of 6 edges of a polygon (CCW vertices)."""
        normals = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            e = v2 - v1
            n = np.array([e[1], -e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-15:
                n = n / nn
            normals.append(n)
        return np.array(normals)
    
    def separation_check(hex1, hex2, tol=1e-7):
        """
        Check if two hexagons are separated using the 12-edge-normal test.
        Returns (is_separated, min_gap) where min_gap is the maximum over all 12 normals
        of the gap between projected intervals (positive = separated, negative = overlap).
        """
        v1 = hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
        v2 = hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
        n1 = edge_normals(v1)
        n2 = edge_normals(v2)
        
        max_gap = -np.inf
        for normal in np.vstack([n1, n2]):
            p1 = v1 @ normal
            p2 = v2 @ normal
            gap = max(np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
            if gap > max_gap:
                max_gap = gap
        
        return max_gap > tol, max_gap
    
    def all_separated(inner, tol=1e-7):
        """Check all pairs are separated. Returns (bool, worst_gap)."""
        worst_gap = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                sep, gap = separation_check(inner[i], inner[j], tol)
                if gap < worst_gap:
                    worst_gap = gap
                if not sep:
                    return False, gap
        return True, worst_gap
    
    def outer_contains(inner, outer, side, tol=1e-7):
        """Check all inner hexagons are inside the outer hexagon."""
        ox, oy, oa = outer
        ov = hex_vertices(ox, oy, side, oa)
        on = edge_normals(ov)
        
        for i in range(N):
            iv = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            # Shift by outer center
            ivs = iv - np.array([ox, oy])
            ovs = ov - np.array([ox, oy])
            
            for k in range(6):
                p1 = ovs[k]
                p2 = ovs[(k + 1) % 6]
                e = p2 - p1
                n = np.array([e[1], -e[0]])
                nn = np.linalg.norm(n)
                if nn > 1e-15:
                    n = n / nn
                
                # Inner vertices should be on the inward side
                proj_inner = ivs @ n
                proj_edge = p1 @ n
                if np.max(proj_inner) > proj_edge + tol:
                    return False
        return True
    
    def is_valid(inner, outer, side, tol=1e-7):
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
            return False
        if side < np.sqrt(11) - tol:
            return False
        sep, _ = all_separated(inner, tol)
        if not sep:
            return False
        if not outer_contains(inner, outer, side, tol):
            return False
        return True
    
    def enclosing_side(centers, alpha_deg):
        """Compute minimum outer hex side length to contain all unit hexagons at given centers."""
        angles = np.radians(alpha_deg) + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        base_v = np.array([[1.0, 0.0], [0.5, sqrt3/2], [-0.5, sqrt3/2],
                           [-1.0, 0.0], [-0.5, -sqrt3/2], [0.5, -sqrt3/2]])
        
        max_proj = 0.0
        for c in centers:
            verts = base_v + c
            projs = verts @ normals.T
            max_proj = max(max_proj, np.max(projs))
        return 2.0 * max_proj / sqrt3
    
    def smooth_penalty(inner, beta=30.0):
        """
        Smooth penalty for overlaps using soft-min over the 12 separation gaps.
        For each pair, compute a soft measure of how much they overlap.
        """
        total_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                v1 = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                v2 = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                n1 = edge_normals(v1)
                n2 = edge_normals(v2)
                
                gaps = []
                for normal in np.vstack([n1, n2]):
                    p1 = v1 @ normal
                    p2 = v2 @ normal
                    gap = max(np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
                    gaps.append(gap)
                
                gaps = np.array(gaps)
                # Soft min: if all gaps <= 0, there's overlap
                # We want penalty when max_gap < 0
                # Use: -softmax(beta * gaps) / beta as a smooth approximation of -max(gaps)
                # Actually we want penalty when the hexagons overlap, i.e., when max_gap < 0
                # Use a smooth hinge: max(0, -max_gap) approximated smoothly
                # soft_max of gaps:
                shifted = beta * gaps
                max_shifted = np.max(shifted)
                soft_max = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted))) / beta
                # soft_max approximates max(gaps)
                # penalty = max(0, -soft_max) -> use smooth hinge
                if soft_max < 0:
                    total_pen += (-soft_max) ** 2
        return total_pen
    
    def objective_smooth(x, beta=30.0):
        """Objective: enclosing side + smooth overlap penalty."""
        centers = x[:22].reshape(11, 2)
        alpha_deg = np.degrees(x[22])
        base = enclosing_side(centers, alpha_deg)
        inner = np.column_stack((centers, np.zeros(11)))
        pen = smooth_penalty(inner, beta)
        return base + 50.0 * pen
    
    def objective_exact(x):
        """Objective: enclosing side + exact overlap penalty."""
        centers = x[:22].reshape(11, 2)
        alpha_deg = np.degrees(x[22])
        base = enclosing_side(centers, alpha_deg)
        inner = np.column_stack((centers, np.zeros(11)))
        sep, worst_gap = all_separated(inner, tol=0.0)
        pen = 0.0
        if worst_gap < 0:
            pen = (-worst_gap) ** 2
        return base + 100.0 * pen
    
    def initial_packing():
        """7-flower + 4 extras arrangement."""
        pos = np.zeros((11, 2))
        pos[0] = [0, 0]
        for i in range(6):
            angle = i * np.pi / 3
            pos[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
        # 4 extras in gaps
        pos[7] = [2*sqrt3, 0]
        pos[8] = pos[3] + sqrt3 * np.array([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
        pos[9] = pos[5] + sqrt3 * np.array([np.cos(4*np.pi/3), np.sin(4*np.pi/3)])
        pos[10] = pos[2] + sqrt3 * np.array([np.cos(np.pi/3), np.sin(np.pi/3)])
        return pos.flatten()
    
    x0 = initial_packing()
    x0 = np.append(x0, 0.0)  # alpha = 0 rad
    
    bounds = [(-6, 6)] * 22 + [(0, np.pi/3)]
    
    best_x = x0.copy()
    best_val = np.inf
    
    def track_result(x, val):
        nonlocal best_x, best_val
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    track_result(x0, objective_smooth(x0))
    
    # Phase 1: Nelder-Mead with smooth objective
    if time.time() < deadline - 60:
        try:
            res = minimize(objective_smooth, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_val:
                track_result(res.x, res.fun)
        except Exception:
            pass
    
    # Phase 2: Multiple random restarts with smooth objective
    np.random.seed(123)
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        x_r = best_x.copy()
        x_r[:22] += np.random.randn(22) * 0.3
        x_r[22] = np.random.uniform(0, np.pi/3)
        
        try:
            res = minimize(objective_smooth, x_r, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_val:
                track_result(res.x, res.fun)
        except Exception:
            pass
    
    # Phase 3: Polish with SLSQP using exact objective
    if time.time() < deadline - 30:
        try:
            res = minimize(objective_exact, best_x, method='SLSQP', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < best_val:
                track_result(res.x, res.fun)
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 20:
        try:
            res = minimize(objective_exact, best_x, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_val:
                track_result(res.x, res.fun)
        except Exception:
            pass
    
    # Extract best result
    centers = best_x[:22].reshape(11, 2)
    alpha_deg = np.degrees(best_x[22])
    side = enclosing_side(centers, alpha_deg)
    
    inner = np.column_stack((centers, np.zeros(11)))
    outer = np.array([0.0, 0.0, alpha_deg])
    
    # Validate and fix
    if not is_valid(inner, outer, side):
        # Try increasing side slightly
        for _ in range(50):
            side *= 1.001
            if is_valid(inner, outer, side):
                break
        else:
            # Fallback: use a known valid packing
            inner = np.array([
                [0, 0, 0],
                [-sqrt3, 0, 0],
                [sqrt3, 0, 0],
                [-sqrt3/2, 1.5, 0],
                [sqrt3/2, 1.5, 0],
                [-sqrt3/2, -1.5, 0],
                [sqrt3/2, -1.5, 0],
                [-2*sqrt3, 0, 0],
                [2*sqrt3, 0, 0],
                [-sqrt3, 3.0, 0],
                [sqrt3, -3.0, 0],
            ])
            outer = np.array([0.0, 0.0, 0.0])
            c = inner[:, :2]
            side = enclosing_side(c, 0.0)
            for _ in range(50):
                if is_valid(inner, outer, side):
                    break
                side *= 1.01
    
    return inner, outer, float(side)
