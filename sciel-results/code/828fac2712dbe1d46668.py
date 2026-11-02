import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices_local(angle_deg, cx=0.0, cy=0.0):
    """Vertices of unit hexagon centered at (cx,cy) with rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _support_hex(cx, cy, angle_deg, direction):
    """Support function of unit hexagon in given direction (unit vector)."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    verts = np.column_stack([np.cos(angles), np.sin(angles)])
    return cx * direction[0] + cy * direction[1] + np.max(verts @ direction)

def _separating_axis_exists(v1, v2):
    """Check if two hexagons (as vertex arrays) have a separating axis."""
    # Collect all edge normals from both polygons
    axes = []
    for v in (v1, v2):
        for i in range(6):
            e = v[(i + 1) % 6] - v[i]
            n = np.array([-e[1], e[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            axes.append(n)
    for n in axes:
        proj1 = v1 @ n
        proj2 = v2 @ n
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return True
    return False

def _check_valid(inner, outer_center, outer_side, outer_angle_deg):
    """Check if all inner hexagons are inside outer and non-overlapping."""
    oc = np.array(outer_center)
    oa = np.radians(outer_angle_deg)
    # Outer hexagon vertices
    o_angles = oa + np.arange(6) * np.pi / 3.0
    o_verts = oc + outer_side * np.column_stack([np.cos(o_angles), np.sin(o_angles)])
    
    inner_verts = []
    for (cx, cy, ang) in inner:
        v = _hex_vertices_local(ang, cx, cy)
        inner_verts.append(v)
        # Check containment: all vertices of inner must be inside outer
        # Use separating axis: for each edge normal of outer, check inner is on correct side
        for i in range(6):
            e = o_verts[(i + 1) % 6] - o_verts[i]
            n = np.array([-e[1], e[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            # Ensure n points inward
            mid = (o_verts[i] + o_verts[(i+1)%6]) / 2
            if np.dot(n, oc - mid) < 0:
                n = -n
            # All inner vertices must satisfy dot(n, v - o_verts[i]) <= 0 (inside)
            if np.any((v - o_verts[i]) @ n > 1e-9):
                return False
    
    # Check pairwise non-overlap
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if not _separating_axis_exists(inner_verts[i], inner_verts[j]):
                return False
    return True

def _objective(params, outer_center, outer_side, outer_angle_deg):
    """Penalty-based objective: 0 if valid, large penalty if invalid."""
    inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
    oc = np.array(outer_center)
    oa = np.radians(outer_angle_deg)
    o_angles = oa + np.arange(6) * np.pi / 3.0
    o_verts = oc + outer_side * np.column_stack([np.cos(o_angles), np.sin(o_angles)])
    
    penalty = 0.0
    inner_verts = []
    for idx, (cx, cy, ang) in enumerate(inner):
        v = _hex_vertices_local(ang, cx, cy)
        inner_verts.append(v)
        for i in range(6):
            e = o_verts[(i + 1) % 6] - o_verts[i]
            n = np.array([-e[1], e[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            mid = (o_verts[i] + o_verts[(i+1)%6]) / 2
            if np.dot(n, oc - mid) < 0:
                n = -n
            viol = np.max((v - o_verts[i]) @ n)
            if viol > 0:
                penalty += viol * viol
    
    for i in range(N):
        for j in range(i + 1, N):
            v1, v2 = inner_verts[i], inner_verts[j]
            min_gap = 0.0
            axes = []
            for v in (v1, v2):
                for k in range(6):
                    e = v[(k + 1) % 6] - v[k]
                    n = np.array([-e[1], e[0]])
                    n = n / (np.linalg.norm(n) + 1e-15)
                    axes.append(n)
            for n in axes:
                p1 = v1 @ n
                p2 = v2 @ n
                gap = min(p2.min() - p1.max(), p1.min() - p2.max())
                if gap < 0:
                    min_gap = min(min_gap, gap)
            if min_gap < 0:
                penalty += min_gap * min_gap * 10
    
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_side = float('inf')
    best_result = None
    
    # Try multiple outer rotations and initial configurations
    outer_rotations = [0.0, 30.0, 60.0]
    
    # Initial configurations: hexagonal lattice patterns
    def _initial_config(mode):
        """Generate initial positions for N hexagons."""
        if mode == 0:
            # Linear arrangement
            positions = []
            for k in range(N):
                x = (k - (N-1)/2.0) * 2.1
                positions.append([x, 0.0, 0.0])
            return positions
        elif mode == 1:
            # Hexagonal cluster
            positions = []
            # Center
            positions.append([0.0, 0.0, 0.0])
            # Ring 1
            for i in range(6):
                a = i * np.pi / 3.0
                positions.append([2.0*np.cos(a), 2.0*np.sin(a), 0.0])
            # Remaining
            for i in range(N - 7):
                a = (i + 3) * np.pi / 3.0
                positions.append([4.0*np.cos(a), 4.0*np.sin(a), 0.0])
            return positions[:N]
        elif mode == 2:
            # Two-row arrangement
            positions = []
            row1 = N // 2
            row2 = N - row1
            for k in range(row1):
                positions.append([(k - row1/2.0) * 2.1, 1.8, 0.0])
            for k in range(row2):
                positions.append([(k - row2/2.0) * 2.1, -1.8, 0.0])
            return positions
        else:
            # Random
            np.random.seed(mode)
            positions = []
            for _ in range(N):
                positions.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5), 
                                  np.random.choice([0.0, 30.0, 60.0])])
            return positions
    
    modes = [0, 1, 2, 3, 4]
    
    for rot in outer_rotations:
        for mode in modes:
            if time.time() > deadline:
                break
            if best_result is not None and best_side < 9.0:
                break
            
            init = _initial_config(mode)
            x0 = np.array([p for pos in init for p in pos])
            
            # Estimate initial outer side
            # For a rough bound, use the extent of initial positions
            xs = [p[0] for p in init]
            ys = [p[1] for p in init]
            rough_side = max(max(xs) - min(xs), max(ys) - min(ys)) / 1.0 + 4.0
            rough_side = max(rough_side, 8.0)
            
            # Optimize: minimize outer side by adjusting positions
            # We'll do a two-phase approach:
            # Phase 1: Fix outer side, optimize positions to fit
            # Phase 2: Binary search on outer side
            
            def _fit_check(side_len):
                """Try to fit all hexagons in outer hexagon of given side."""
                def obj(params):
                    if time.time() > deadline:
                        return 1e10
                    inner = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
                    oc = np.array([0.0, 0.0])
                    oa = np.radians(rot)
                    o_angles = oa + np.arange(6) * np.pi / 3.0
                    o_verts = oc + side_len * np.column_stack([np.cos(o_angles), np.sin(o_angles)])
                    
                    penalty = 0.0
                    inner_verts = []
                    for idx, (cx, cy, ang) in enumerate(inner):
                        v = _hex_vertices_local(ang, cx, cy)
                        inner_verts.append(v)
                        for i in range(6):
                            e = o_verts[(i + 1) % 6] - o_verts[i]
                            n = np.array([-e[1], e[0]])
                            nn = np.linalg.norm(n)
                            if nn > 1e-15:
                                n = n / nn
                            mid = (o_verts[i] + o_verts[(i+1)%6]) / 2
                            if np.dot(n, oc - mid) < 0:
                                n = -n
                            viol = np.max((v - o_verts[i]) @ n)
                            if viol > 0:
                                penalty += viol * viol
                                        
                    for i in range(N):
                        for j in range(i + 1, N):
                            v1, v2 = inner_verts[i], inner_verts[j]
                            min_gap = 0.0
                            axes = []
                            for v in (v1, v2):
                                for k in range(6):
                                    e = v[(k + 1) % 6] - v[k]
                                    nn = np.linalg.norm(e)
                                    if nn > 1e-15:
                                        n = np.array([-e[1], e[0]]) / nn
                                        axes.append(n)
                            for n in axes:
                                p1 = v1 @ n
                                p2 = v2 @ n
                                gap = min(p2.min() - p1.max(), p1.min() - p2.max())
                                if gap < 0:
                                    min_gap = min(min_gap, gap)
                            if min_gap < 0:
                                penalty += min_gap * min_gap * 10
                    return penalty
                
                result = minimize(obj, x0, method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                return result.fun < 1e-6, result.x
            
            # Binary search on side length
            lo, hi = 7.0, 20.0
            # First check if hi works
            ok, _ = _fit_check(hi)
            if not ok:
                continue
            
            best_x = x0.copy()
            for _ in range(15):
                if time.time() > deadline:
                    break
                mid = (lo + hi) / 2.0
                ok, x_mid = _fit_check(mid)
                if ok:
                    hi = mid
                    best_x = x_mid
                else:
                    lo = mid
            
            if hi < best_side:
                best_side = hi
                inner = [[best_x[3*i], best_x[3*i+1], best_x[3*i+2]] for i in range(N)]
                best_result = (inner, (0.0, 0.0), best_side, rot)
    
    if best_result is None:
        # Fallback: return a valid but not optimal configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final validation and slight refinement
    inner, center, side, angle = best_result
    
    # Try to tighten with a final polish
    if time.time() < deadline - 10.0:
        x0 = np.array([p for pos in inner for p in pos])
        def final_obj(params):
            inner_p = [[params[3*i], params[3*i+1], params[3*i+2]] for i in range(N)]
            oc = np.array(center)
            oa = np.radians(angle)
            o_angles = oa + np.arange(6) * np.pi / 3.0
            o_verts = oc + side * np.column_stack([np.cos(o_angles), np.sin(o_angles)])
            
            penalty = 0.0
            inner_verts = []
            for idx, (cx, cy, ang) in enumerate(inner_p):
                v = _hex_vertices_local(ang, cx, cy)
                inner_verts.append(v)
                for i in range(6):
                    e = o_verts[(i + 1) % 6] - o_verts[i]
                    n = np.array([-e[1], e[0]])
                    nn = np.linalg.norm(n)
                    if nn > 1e-15:
                        n = n / nn
                    mid = (o_verts[i] + o_verts[(i+1)%6]) / 2
                    if np.dot(n, oc - mid) < 0:
                        n = -n
                    viol = np.max((v - o_verts[i]) @ n)
                    if viol > 0:
                        penalty += viol * viol
            for i in range(N):
                for j in range(i + 1, N):
                    v1, v2 = inner_verts[i], inner_verts[j]
                    min_gap = 0.0
                    axes = []
                    for v in (v1, v2):
                        for k in range(6):
                            e = v[(k + 1) % 6] - v[k]
                            nn = np.linalg.norm(e)
                            if nn > 1e-15:
                                n = np.array([-e[1], e[0]]) / nn
                                axes.append(n)
                    for n in axes:
                        p1 = v1 @ n
                        p2 = v2 @ n
                        gap = min(p2.min() - p1.max(), p1.min() - p2.max())
                        if gap < 0:
                            min_gap = min(min_gap, gap)
                    if min_gap < 0:
                        penalty += min_gap * min_gap * 10
            return penalty
        
        result = minimize(final_obj, x0, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.fun < 1e-8:
            inner = [[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]
    
    return inner, center, side, angle
