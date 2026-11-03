import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0 # Safety margin

def _hex_vertices(cx, cy, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = math.cos(theta)
        y = math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _get_axes(verts):
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12: continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _min_gap(vertsA, vertsB, axes):
    """
    Returns the minimum separation distance.
    Positive if separated, negative if overlapping.
    """
    min_gap = float('inf')
    for ax, ay in axes:
        projA = [vx * ax + vy * ay for vx, vy in vertsA]
        projB = [vx * ax + vy * ay for vx, vy in vertsB]
        maxA, minA = max(projA), min(projA)
        maxB, minB = max(projB), min(projB)
        
        gap1 = minB - maxA
        gap2 = minA - maxB
        gap = max(gap1, gap2)
        
        if gap < min_gap:
            min_gap = gap
        if min_gap > 0:
            # Can't get better than 0 if we just want penalty, 
            # but for exact distance we need min. 
            # Actually if min_gap > 0, we can early exit if we only care about penalty?
            # No, we need the actual min gap to know how far apart they are for smoothness.
            # But for penalty max(0, -gap)^2, if gap > 0, penalty is 0.
            # So if we find any axis with gap > 0, we know they are separated?
            # NO. SAT says if there is ANY separating axis, they are disjoint.
            # So if we find ONE axis with gap > 0, they are disjoint.
            # We can return a large positive number or just 1.0 for penalty purposes.
            return 1.0 
    return min_gap

def _collision_penalty(verts_list):
    penalty = 0.0
    n = len(verts_list)
    # Precompute axes for all
    axes_list = [_get_axes(v) for v in verts_list]
    
    for i in range(n):
        for j in range(i + 1, n):
            # Quick distance check
            cx1, cy1 = sum(v[0] for v in verts_list[i])/6, sum(v[1] for v in verts_list[i])/6
            cx2, cy2 = sum(v[0] for v in verts_list[j])/6, sum(v[1] for v in verts_list[j])/6
            if (cx1-cx2)**2 + (cy1-cy2)**2 > (2*1.5)**2: # Approx radius 1.5
                continue
            
            axes = axes_list[i] + axes_list[j]
            gap = _min_gap(verts_list[i], verts_list[j], axes)
            if gap < 0:
                penalty += gap * gap
    return penalty

def _outer_side(verts_list, center, angle_deg):
    cx, cy = center
    a = math.radians(angle_deg)
    apothem = math.sqrt(3) / 2.0
    max_need = 0.0
    for i in range(len(verts_list)):
        for vx, vy in verts_list[i]:
            for k in range(6):
                na = a + math.radians(90 + 60 * k)
                nx, ny = math.cos(na), math.sin(na)
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem
                if val > max_need:
                    max_need = val
    return max_need

def construct():
    t0 = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Initializers
    def init_hex_lattice():
        pts = []
        # 1 in center, 6 around, 5 more
        # Hex grid coords
        # (0,0)
        pts.append([0,0,0])
        r1 = 2 * math.sqrt(3)
        for i in range(6):
            a = math.radians(60*i)
            pts.append([r1*math.cos(a), r1*math.sin(a), 0])
        # 5 more in second ring
        # Positions in second ring: (2,0), (1,1), (-1,2), (-2,0), (-1,-1), (1,-2) in axial
        # Let's just pick 5 random spots in a larger ring
        r2 = r1 * 1.8
        for i in range(5):
            a = math.radians(60*i + 30)
            pts.append([r2*math.cos(a), r2*math.sin(a), 0])
        return pts

    def init_random():
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 60)])
        return pts

    initializers = [init_hex_lattice, init_random, init_random, init_random]
    
    for init_func in initializers:
        if time.time() - t0 > TIME_LIMIT - 30:
            break
        
        cfg = init_func()
        if len(cfg) != N: continue
        
        # Flatten for NM
        x0 = np.array([p[0] for p in cfg] + [p[1] for p in cfg] + [p[2] for p in cfg])
        
        def objective(params):
            cx = params[:N]
            cy = params[N:2*N]
            ang = params[2*N:]
            
            verts_list = []
            for i in range(N):
                verts_list.append(_hex_vertices(cx[i], cy[i], ang[i]))
            
            side = _outer_side(verts_list, (0,0), 0)
            pen = _collision_penalty(verts_list)
            
            return side + 100.0 * pen
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True})
            opt_params = res.x
            
            cx = opt_params[:N]
            cy = opt_params[N:2*N]
            ang = opt_params[2*N:]
            
            # Verify validity
            verts_list = [_hex_vertices(cx[i], cy[i], ang[i]) for i in range(N)]
            axes_list = [_get_axes(v) for v in verts_list]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    axes = axes_list[i] + axes_list[j]
                    gap = _min_gap(verts_list[i], verts_list[j], axes)
                    if gap < 1e-6: # Allow tiny overlap for numerical stability? No, must be disjoint.
                        valid = False
                        break
                if not valid: break
            
            if valid:
                # Optimize outer hex center and angle
                # For a fixed set of inner hexes, the minimal enclosing hexagon side
                # is minimized when the center is the centroid? Not necessarily.
                # But we can just compute the side for center (0,0) and angle 0.
                # Actually, the problem allows moving the outer hexagon.
                # The "side" is the raw value.
                # We can optimize the outer center and angle as well.
                # Let's do a quick local search on outer params.
                
                best_local_side = _outer_side(verts_list, (0,0), 0)
                best_local_center = (0,0)
                best_local_angle = 0
                
                # Grid search for outer angle and center
                for ang_off in range(-10, 11, 5):
                    a_test = ang_off
                    for dx in [-0.5, 0, 0.5]:
                        for dy in [-0.5, 0, 0.5]:
                            s = _outer_side(verts_list, (dx, dy), a_test)
                            if s < best_local_side:
                                best_local_side = s
                                best_local_center = (dx, dy)
                                best_local_angle = a_test
                                
                if best_local_side < best_side:
                    best_side = best_local_side
                    best_cfg = [[cx[i], cy[i], ang[i]] for i in range(N)]
                    best_center = best_local_center
                    best_angle = best_local_angle
                    
        except Exception:
            pass

    if best_cfg is None:
        # Fallback
        cfg = init_hex_lattice()
        # Scale up to ensure disjoint
        scale = 1.5
        for p in cfg:
            p[0] *= scale
            p[1] *= scale
        best_cfg = cfg
        verts_list = [_hex_vertices(p[0], p[1], p[2]) for p in cfg]
        best_side = _outer_side(verts_list, (0,0), 0)
        best_center = (0,0)
        best_angle = 0

    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, best_center, best_side, best_angle

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
