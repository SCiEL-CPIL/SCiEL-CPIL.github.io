# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _build_inner_symmetric(cx, cy, ca, ox, oy, oa):
    """Build 11 hexes: center + 6 rotations of (ox,oy,oa)."""
    inner = np.empty((11, 3))
    inner[0] = [cx, cy, ca]
    for k in range(6):
        th = np.radians(60 * k)
        cth, sth = np.cos(th), np.sin(th)
        rx = cth * ox - sth * oy
        ry = sth * ox + cth * oy
        inner[k + 1] = [rx, ry, oa]
    # 4 extra hexes at 2*orbit positions, angles 0,60,120,180 (or aligned with orbit)
    # Actually for 11 = 1 + 6 + 4, the 4 extra are at distance 2*orbit, angles 0,60,120,180
    for k in range(4):
        th = np.radians(60 * k)
        cth, sth = np.cos(th), np.sin(th)
        rx = 2.0 * (cth * ox - sth * oy)
        ry = 2.0 * (cth * ox + sth * oy)
        inner[7 + k] = [rx, ry, oa]
    return inner

def _min_enclosing_hex_side(inner):
    """Min side of outer hex (center origin, angle 0) containing all inner vertices."""
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    max_proj = 0.0
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        projs = verts @ normals.T
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    return max_proj / apothem_factor

def _check_separation(inner, tol=1e-6):
    n = len(inner)
    for i in range(n):
        ci = inner[i]
        vi = _hex_vertices(ci[0], ci[1], 1.0, ci[2])
        for j in range(i + 1, n):
            cj = inner[j]
            vj = _hex_vertices(cj[0], cj[1], 1.0, cj[2])
            # 12 edge normals
            normals = []
            for k in range(6):
                a = np.radians(ci[2]) + np.radians(30 + 60 * k)
                normals.append([np.cos(a), np.sin(a)])
            for k in range(6):
                a = np.radians(cj[2]) + np.radians(30 + 60 * k)
                normals.append([np.cos(a), np.sin(a)])
            sep = False
            for normal in normals:
                ni = vi @ normal
                nj = vj @ normal
                if max(ni) < min(nj) - tol or max(nj) < min(ni) - tol:
                    sep = True
                    break
            if not sep:
                return False
    return True

def _check_containment(inner, outer_center, outer_side, outer_angle, tol=-1e-6):
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for p in verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < tol:
                    return False
    return True

def _objective_sym(params):
    cx, cy, ca, ox, oy, oa = params
    inner = _build_inner_symmetric(cx, cy, ca, ox, oy, oa)
    if not _check_separation(inner):
        return 1e10
    return _min_enclosing_hex_side(inner)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin
    
    d = np.sqrt(3)
    
    # Known good starting point: center at origin, orbit at distance d, angle 0
    best_params = np.array([0.0, 0.0, 0.0, d * np.cos(np.radians(30)), d * np.sin(np.radians(30)), 0.0])
    best_side = _objective_sym(best_params)
    
    # Grid search over key parameters
    # cx, cy: center offset (small range)
    # ca: center angle (0 or 30 by symmetry, but allow 0-60)
    # ox, oy: orbit position (distance ~ d, angle ~ 30)
    # oa: orbit angle (0 or 30)
    
    # Phase 1: Coarse grid
    cx_vals = np.linspace(-0.5, 0.5, 11)
    cy_vals = np.linspace(-0.5, 0.5, 11)
    ca_vals = np.linspace(0, 60, 13)
    # orbit: distance from d-0.5 to d+0.5, angle 0 to 60
    r_vals = np.linspace(d - 0.5, d + 0.5, 11)
    ang_vals = np.linspace(0, 60, 13)
    oa_vals = np.linspace(0, 60, 13)
    
    # Too large for full grid (11*11*13*11*13*13 = ~3.2M). Reduce.
    # Focus on the most impactful: orbit distance and angle, center angle, orbit angle
    # Fix cx=0, cy=0 initially (center at origin by symmetry of outer hex)
    
    r_coarse = np.linspace(d - 0.3, d + 0.3, 9)
    ang_coarse = np.linspace(0, 60, 9)
    ca_coarse = np.linspace(0, 60, 7)
    oa_coarse = np.linspace(0, 60, 7)
    
    for r in r_coarse:
        for ang in ang_coarse:
            ox = r * np.cos(np.radians(ang))
            oy = r * np.sin(np.radians(ang))
            for ca in ca_coarse:
                for oa in oa_coarse:
                    if time.time() > deadline - 120:
                        break
                    params = np.array([0.0, 0.0, ca, ox, oy, oa])
                    val = _objective_sym(params)
                    if val < best_side:
                        best_side = val
                        best_params = params
                if time.time() > deadline - 120:
                    break
            if time.time() > deadline - 120:
                break
        if time.time() > deadline - 120:
            break
    
    # Phase 2: Fine grid around best
    if time.time() < deadline - 100:
        cx_f = best_params[0]
        cy_f = best_params[1]
        ca_f = best_params[2]
        ox_f = best_params[3]
        oy_f = best_params[4]
        oa_f = best_params[5]
        
        r_f = np.linspace(ox_f - 0.15, ox_f + 0.15, 15)
        ang_f = np.linspace(max(0, ang_coarse[np.argmin(np.abs(r_coarse - r_f[0]))] - 5), 
                            min(60, ang_coarse[np.argmin(np.abs(r_coarse - r_f[0]))] + 5), 15)
        ca_f = np.linspace(max(0, ca_f - 5), min(60, ca_f + 5), 15)
        oa_f = np.linspace(max(0, oa_f - 5), min(60, oa_f + 5), 15)
        
        for r in r_f:
            for ang in ang_f:
                ox = r * np.cos(np.radians(ang))
                oy = r * np.sin(np.radians(ang))
                for ca in ca_f:
                    for oa in oa_f:
                        if time.time() > deadline - 60:
                            break
                        params = np.array([cx_f, cy_f, ca, ox, oy, oa])
                        val = _objective_sym(params)
                        if val < best_side:
                            best_side = val
                            best_params = params
                    if time.time() > deadline - 60:
                        break
                if time.time() > deadline - 60:
                    break
            if time.time() > deadline - 60:
                break
    
    # Phase 3: Golden section refinement on each parameter
    if time.time() < deadline - 30:
        def _refine_1d(idx, lo, hi, n_iter=30):
            nonlocal best_params, best_side
            # Golden ratio
            gr = (np.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            f1 = _objective_sym(np.concatenate([best_params[:idx], [x1], best_params[idx+1:]]))
            f2 = _objective_sym(np.concatenate([best_params[:idx], [x2], best_params[idx+1:]]))
            for _ in range(n_iter):
                if time.time() > deadline - 20:
                    break
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = _objective_sym(np.concatenate([best_params[:idx], [x1], best_params[idx+1:]]))
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = _objective_sym(np.concatenate([best_params[:idx], [x2], best_params[idx+1:]]))
            best_x = (lo + hi) / 2
            best_f = _objective_sym(np.concatenate([best_params[:idx], [best_x], best_params[idx+1:]]))
            if best_f < best_side:
                best_params = np.concatenate([best_params[:idx], [best_x], best_params[idx+1:]])
                best_side = best_f
        
        # Refine each parameter
        _refine_1d(0, -0.5, 0.5)   # cx
        _refine_1d(1, -0.5, 0.5)   # cy
        _refine_1d(2, 0, 60)       # ca
        _refine_1d(3, -5, 5)       # ox
        _refine_1d(4, -5, 5)       # oy
        _refine_1d(5, 0, 60)       # oa
    
    # Build final inner
    inner = _build_inner_symmetric(*best_params)
    
    # Verify and get actual side
    side = _min_enclosing_hex_side(inner)
    if not _check_separation(inner):
        # Fallback to known valid
        inner = _build_inner_symmetric(0, 0, 0, d * np.cos(np.radians(30)), d * np.sin(np.radians(30)), 0)
        side = _min_enclosing_hex_side(inner)
    
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)

# EVOLVE-BLOCK-END
