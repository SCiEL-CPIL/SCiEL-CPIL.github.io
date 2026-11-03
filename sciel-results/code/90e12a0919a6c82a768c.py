import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 143.0

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_sqrt3 = math.sqrt(3)
_inv_sqrt3 = 1.0 / _sqrt3


def construct():
    t0 = time.time()
    
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]

    def check_disjoint(positions, angles_rad):
        """Vectorized separating axis test for all pairs. Returns True if all disjoint."""
        # Fast rejection: pairwise distances
        dx = positions[:, 0:1] - positions[1:, 0:1].T  # N x (N-1)
        dy = positions[:, 1:2] - positions[1:, 1:2].T
        dist_sq = dx*dx + dy*dy
        # Pairs that are close enough to need full check
        close_mask = dist_sq < 4.5
        
        if not close_mask.any():
            return True
        
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        # Rotated vertices: (N, 6, 2)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]  # (N, 6)
        verts_y = ry + positions[:, 1:2]  # (N, 6)
        
        # Rotated edge normals: (N, 6, 2)
        enx_r = _enx[None, :] * c[:, None] - _eny[None, :] * s[:, None]
        eny_r = _enx[None, :] * s[:, None] + _eny[None, :] * c[:, None]
        
        # Get indices of close pairs
        close_i, close_j = np.where(close_mask)
        
        for ii, jj in zip(close_i, close_j):
            j2 = jj + 1  # actual index in positions
            # Project both hexagons onto each of the 12 normals
            # Hexagon i's normals
            for k in range(6):
                nx, ny = enx_r[ii, k], eny_r[ii, k]
                p1 = verts_x[ii] * nx + verts_y[ii] * ny
                p2 = verts_x[j2] * nx + verts_y[j2] * ny
                if p1.max() < p2.min() or p2.max() < p1.min():
                    break
            else:
                # Hexagon j's normals
                for k in range(6):
                    nx, ny = enx_r[j2, k], eny_r[j2, k]
                    p1 = verts_x[ii] * nx + verts_y[ii] * ny
                    p2 = verts_x[j2] * nx + verts_y[j2] * ny
                    if p1.max() < p2.min() or p2.max() < p1.min():
                        break
                else:
                    return False  # No separating axis found -> overlap
        
        return True

    def compute_outer_side(positions, angles_deg):
        """Compute minimum enclosing hexagon side length, rotation, and center."""
        angles_rad = np.radians(angles_deg)
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]  # (N, 6)
        verts_y = ry + positions[:, 1:2]  # (N, 6)
        verts_flat = np.column_stack((verts_x.ravel(), verts_y.ravel()))  # (72, 2)
        center = verts_flat.mean(axis=0)
        offsets = verts_flat - center
        
        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            normal_angles = phi + np.pi/6 + np.arange(6) * np.pi/3
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
            abs_proj = np.abs(proj)
            max_projs = np.max(abs_proj, axis=0)
            side_vals = (2.0 / _sqrt3) * max_projs
            return np.max(side_vals)
        
        best_side = float('inf')
        best_phi = 0.0
        for phi_deg in range(0, 60, 1):
            sv = eval_phi(phi_deg)
            if sv < best_side:
                best_side = sv
                best_phi = float(phi_deg)
        
        # Fine tune
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        try:
            res = minimize_scalar_local(eval_phi, lo, hi)
            if res < best_side:
                best_side = res
        except Exception:
            pass
        
        return best_side, best_phi, center[0], center[1]

    def minimize_scalar_local(func, lo, hi):
        """Simple golden-section minimization."""
        gr = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = func(c), func(d)
        for _ in range(50):
            if b - a < 1e-8:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = func(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = func(d)
        return min(fc, fd)

    def objective(params):
        """Objective for Nelder-Mead: returns outer side length or penalty."""
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = params[3*i]
            pos[i, 1] = params[3*i+1]
            ang[i] = params[3*i+2]
        
        angles_rad = np.radians(ang)
        if not check_disjoint(pos, angles_rad):
            return 1e6
        
        side, _, _, _ = compute_outer_side(pos, ang)
        return side

    # --- Initial configurations ---
    def make_honeycomb():
        a1 = np.array([_sqrt3, 0.0])
        a2 = np.array([_sqrt3/2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        return pos, ang

    def make_ring():
        # 12 hexagons arranged in a ring around center
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        r = 2.2
        for i in range(N):
            theta = 2 * math.pi * i / N
            pos[i, 0] = r * math.cos(theta)
            pos[i, 1] = r * math.sin(theta)
            ang[i] = theta * 180.0 / math.pi
        return pos, ang

    def make_two_rows():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        a1 = np.array([_sqrt3, 0.0])
        a2 = np.array([_sqrt3/2.0, 1.5])
        # Two rows of 6
        for i in range(6):
            pos[i, :] = i * a1
            pos[i+6, :] = i * a1 + a2
        return pos, ang

    def make_compact():
        # Compact hexagonal cluster: center + 6 ring + 5 more
        a1 = np.array([_sqrt3, 0.0])
        a2 = np.array([_sqrt3/2.0, 1.5])
        coords = [(0,0)]
        for i in range(6):
            theta = i * math.pi / 3
            # In hex lattice, neighbors are at 60 degree intervals
            coords.append((i % 2, (i+1) % 2 - 1))
        # Just use a simple grid
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        coords = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,-1),(-2,1),(1,1),(-1,-1),(2,0)]
        for idx, (i, j) in enumerate(coords):
            pos[idx] = i * a1 + j * a2
        return pos, ang

    def make_random(seed):
        rng = random.Random(seed)
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(N):
            pos[i, 0] = rng.gauss(0, 2.0)
            pos[i, 1] = rng.gauss(0, 2.0)
            ang[i] = rng.uniform(0, 60)
        return pos, ang

    # --- Multi-start Nelder-Mead ---
    initializers = [
        make_honeycomb,
        make_ring,
        make_two_rows,
        make_compact,
    ]
    
    best_side = float('inf')
    best_pos = None
    best_ang = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Run each initializer
    for init_func in initializers:
        if time.time() - t0 > DEADLINE:
            break
        
        pos, ang = init_func()
        
        # Check if valid
        angles_rad = np.radians(ang)
        if not check_disjoint(pos, angles_rad):
            # Try to fix by spreading out
            for attempt in range(20):
                for i in range(N):
                    pos[i, 0] += random.gauss(0, 0.5)
                    pos[i, 1] += random.gauss(0, 0.5)
                angles_rad = np.radians(ang)
                if check_disjoint(pos, angles_rad):
                    break
            else:
                continue
        
        # Evaluate
        side, phi, cx, cy = compute_outer_side(pos, ang)
        if side < best_side:
            best_side = side
            best_pos = pos.copy()
            best_ang = ang.copy()
            best_phi = phi
            best_cx = cx
            best_cy = cy
        
        # Run Nelder-Mead from this start
        if time.time() - t0 > DEADLINE:
            break
        
        init_params = np.zeros(3*N)
        for i in range(N):
            init_params[3*i] = pos[i, 0]
            init_params[3*i+1] = pos[i, 1]
            init_params[3*i+2] = ang[i]
        
        time_per_nm = min(20.0, (DEADLINE - (time.time() - t0)) / max(1, len(initializers) - initializers.index(init_func)))
        max_iter = int(3000 * time_per_nm / 10.0)
        max_iter = max(max_iter, 500)
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9,
                                    'adaptive': True})
            
            if res.fun < 1e5:  # Valid result
                new_pos = np.zeros((N, 2))
                new_ang = np.zeros(N)
                for i in range(N):
                    new_pos[i, 0] = res.x[3*i]
                    new_pos[i, 1] = res.x[3*i+1]
                    new_ang[i] = res.x[3*i+2]
                
                angles_rad = np.radians(new_ang)
                if check_disjoint(new_pos, angles_rad):
                    side, phi, cx, cy = compute_outer_side(new_pos, new_ang)
                    if side < best_side:
                        best_side = side
                        best_pos = new_pos
                        best_ang = new_ang
                        best_phi = phi
                        best_cx = cx
                        best_cy = cy
        except Exception:
            pass

    # Add random starts if time remains
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10.0 and best_pos is not None:
        n_random = min(5, int(remaining / 15.0))
        for k in range(n_random):
            if time.time() - t0 > DEADLINE:
                break
            pos, ang = make_random(k * 12345 + 678)
            angles_rad = np.radians(ang)
            if not check_disjoint(pos, angles_rad):
                continue
            
            side, phi, cx, cy = compute_outer_side(pos, ang)
            if side < best_side:
                best_side = side
                best_pos = pos.copy()
                best_ang = ang.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy
            
            if time.time() - t0 > DEADLINE:
                break
            
            init_params = np.zeros(3*N)
            for i in range(N):
                init_params[3*i] = pos[i, 0]
                init_params[3*i+1] = pos[i, 1]
                init_params[3*i+2] = ang[i]
            
            max_iter = max(300, int(2000 * (DEADLINE - (time.time() - t0)) / 20.0))
            try:
                res = minimize(objective, init_params, method='Nelder-Mead',
                               options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9,
                                        'adaptive': True})
                if res.fun < 1e5:
                    new_pos = np.zeros((N, 2))
                    new_ang = np.zeros(N)
                    for i in range(N):
                        new_pos[i, 0] = res.x[3*i]
                        new_pos[i, 1] = res.x[3*i+1]
                        new_ang[i] = res.x[3*i+2]
                    angles_rad = np.radians(new_ang)
                    if check_disjoint(new_pos, angles_rad):
                        side, phi, cx, cy = compute_outer_side(new_pos, new_ang)
                        if side < best_side:
                            best_side = side
                            best_pos = new_pos
                            best_ang = new_ang
                            best_phi = phi
                            best_cx = cx
                            best_cy = cy
            except Exception:
                pass

    # Final polish with coordinate descent on best solution
    if time.time() - t0 < DEADLINE - 2.0 and best_pos is not None:
        pos_cd = best_pos.copy()
        ang_cd = best_ang.copy()
        side_cd, phi_cd, cx_cd, cy_cd = compute_outer_side(pos_cd, ang_cd)
        
        step = 0.05
        min_step = 0.0005
        t_cd_start = time.time()
        cd_limit = min(DEADLINE - (time.time() - t0) - 1.0, 15.0)
        
        while time.time() - t_cd_start < cd_limit and step > min_step:
            improved = False
            for idx in range(N):
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = pos_cd.copy()
                        new_ang = ang_cd.copy()
                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        elif dim == 1:
                            new_pos[idx, 1] += direction * step
                        else:
                            new_ang[idx] += direction * step
                        
                        angles_rad = np.radians(new_ang)
                        if not check_disjoint(new_pos, angles_rad):
                            continue
                        
                        new_side, _, _, _ = compute_outer_side(new_pos, new_ang)
                        if new_side < side_cd - 1e-12:
                            pos_cd = new_pos
                            ang_cd = new_ang
                            side_cd = new_side
                            improved = True
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                step *= 0.7
                continue
            break
        
        if side_cd < best_side:
            best_side = side_cd
            best_pos = pos_cd
            best_ang = ang_cd
            best_side, best_phi, best_cx, best_cy = compute_outer_side(best_pos, best_ang)

    # Fallback
    if best_pos is None:
        best_pos, best_ang = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_outer_side(best_pos, best_ang)

    # Final validation
    angles_rad = np.radians(best_ang)
    if not check_disjoint(best_pos, angles_rad):
        best_pos, best_ang = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_outer_side(best_pos, best_ang)

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)
    
    return inner, center, side, angle
