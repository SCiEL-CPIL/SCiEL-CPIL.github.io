import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)

    def hex_vertices(cx, cy, side, rot_deg):
        rot = math.radians(rot_deg)
        verts = []
        for k in range(6):
            ang = rot + k * math.pi / 3.0
            verts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, rot_deg):
        rot = math.radians(rot_deg)
        dx = px - cx
        dy = py - cy
        dx_local = dx * math.cos(rot) + dy * math.sin(rot)
        dy_local = -dx * math.sin(rot) + dy * math.cos(rot)
        apothem = side * APOTHEM
        for k in range(6):
            ang = k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            dot = dx_local * nx + dy_local * ny
            if dot > apothem + 1e-10:
                return False
        return True

    def seg_intersect(p1, p2, p3, p4):
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def hex_overlap(c1, c2, s1, r1_deg, s2, r2_deg, eps=1e-9):
        v1 = hex_vertices(c1[0], c1[1], s1, r1_deg)
        v2 = hex_vertices(c2[0], c2[1], s2, r2_deg)
        
        # Bounding box
        x1_min = min(v[0] for v in v1); x1_max = max(v[0] for v in v1)
        y1_min = min(v[1] for v in v1); y1_max = max(v[1] for v in v1)
        x2_min = min(v[0] for v in v2); x2_max = max(v[0] for v in v2)
        y2_min = min(v[1] for v in v2); y2_max = max(v[1] for v in v2)
        if x1_max < x2_min - eps or x2_max < x1_min - eps or y1_max < y2_min - eps or y2_max < y1_min - eps:
            return False
        
        # SAT
        axes = []
        for k in range(6):
            for i, vlist in enumerate([v1, v2]):
                a = vlist[k]
                b = vlist[(k+1)%6]
                ex = b[0]-a[0]; ey = b[1]-a[1]
                elen = math.sqrt(ex*ex+ey*ey)
                if elen > 1e-12:
                    axes.append((-ey/elen, ex/elen))
        
        for nx, ny in axes:
            proj1 = [v[0]*nx+v[1]*ny for v in v1]
            proj2 = [v[0]*nx+v[1]*ny for v in v2]
            if max(proj1) < min(proj2) - eps or max(proj2) < min(proj1) - eps:
                return False
        return True

    def calc_S(pts, beta, cx, cy):
        """Compute S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_c = np.array(pts) - np.array([cx, cy])
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def optimize_orientation(pts, cx, cy):
        """Minimize S over beta (outer rotation)."""
        best_S = float('inf')
        best_beta = 0.0
        
        # Coarse grid
        n_grid = 37
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi/6) / (n_grid - 1)
            S = calc_S(pts, beta, cx, cy)
            if S < best_S:
                best_S = S
                best_beta = beta
        
        # Local refinement
        try:
            def obj(b):
                b = ((b % (math.pi/6)) + math.pi/6) % (math.pi/6)
                return calc_S(pts, b, cx, cy)
            res = minimize(obj, best_beta, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = ((res.x % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except:
            pass
            
        return best_S, best_beta

    def optimize_center(pts, beta):
        """Minimize S over center (cx, cy) for fixed beta."""
        cx = float(np.mean([p[0] for p in pts]))
        cy = float(np.mean([p[1] for p in pts]))
        best_S, best_beta = optimize_orientation(pts, cx, cy)
        
        # Try to optimize center for the best beta
        try:
            def obj_c(c):
                cx, cy = c
                S, b = optimize_orientation(pts, cx, cy)
                return S
            res = minimize(obj_c, [cx, cy], method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                # Re-evaluate best beta with new center
                cx_new, cy_new = res.x
                _, best_beta = optimize_orientation(pts, cx_new, cy_new)
                cx, cy = cx_new, cy_new
        except:
            pass
            
        return best_S, best_beta, (cx, cy)

    # --- Strategy 1: Lattice based (aligned hexagons) ---
    # Generate many candidate configurations
    candidates = []
    
    # Standard hexagonal packing patterns
    # 1. Central + 6 ring + 4 more
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    
    # 2. Linear / Strip patterns
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    
    # 3. Triangular / Compact
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    
    # 4. Ring patterns
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(-1,1),(0,1),(1,1),(2,1),(3,1),(4,1)])
    candidates.append([(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(2,-2)])
    
    # 5. More diverse
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(0,1),(0,-1),(1,1)])
    candidates.append([(2,0),(-2,0),(0,2),(0,-2),(2,-2),(-2,2),(1,0),(-1,0),(1,1),(-1,1),(0,1)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(-1,1),(0,1),(1,1),(2,1),(3,1),(4,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3)])
    candidates.append([(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(2,-2)])

    best_global_S = float('inf')
    best_global_pts = None
    best_global_rotations = None
    best_global_beta = 0.0
    best_global_center = None

    # Evaluate lattice candidates
    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        pts = [axial_to_cartesian(q, r) for q, r in axial]
        rotations = [0.0] * len(pts)
        
        # Optimize center and orientation for this fixed shape
        S, beta, center = optimize_center(pts, 0.0)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_rotations = rotations
            best_global_beta = beta
            best_global_center = center

    # --- Strategy 2: Force-Directed Relaxation with Rotations ---
    # Start from the best lattice configuration and try to improve by moving/rotating hexagons
    
    if best_global_pts is not None:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            # Deep copy
            pts = [p[:] for p in best_global_pts]
            rots = best_global_rotations[:]
            
            # Parameters for relaxation
            # We want to minimize S, but S depends on the outer container.
            # Instead, we can try to minimize the "tightness" of the packing.
            # A simpler heuristic: minimize the sum of squared distances to the "ideal" boundary?
            # Or, directly minimize S using the current best center/beta as a guide.
            
            # Let's try a local search: perturb positions/rotations and see if S decreases.
            # S is determined by the convex hull of the points (for aligned hexagons, it's the hull of centers).
            # If we rotate, the effective radius changes, but for regular hexagons it's constant.
            # So moving centers is the main lever.
            
            # We can use the gradient of S with respect to center positions.
            # S = (D + APOTHEM) / APOTHEM, where D = max_k max_i |proj_k(pi - c)|.
            # This is a max of linear functions, so it's convex.
            # We can use subgradient methods or just simple hill climbing.
            
            # Let's do a simple simulated annealing / hill climbing step
            current_S = best_global_S
            current_pts = [p[:] for p in pts]
            current_rots = rots[:]
            current_center = best_global_center
            current_beta = best_global_beta
            
            step_size = 0.05
            n_steps = 200
            
            for step in range(n_steps):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                # Pick a random hexagon
                i = np.random.randint(N)
                
                # Perturb position
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                dr = np.random.uniform(-5, 5) # degrees
                
                old_x, old_y = current_pts[i]
                old_r = current_rots[i]
                
                new_x = old_x + dx
                new_y = old_y + dy
                new_r = old_r + dr
                
                # Check validity: no overlap with others, and ideally still "fits" in a reasonable bound
                # Overlap check
                valid = True
                for j in range(N):
                    if i == j:
                        continue
                    if hex_overlap((new_x, new_y), current_pts[j], 1.0, new_r, 1.0, current_rots[j]):
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Compute new S
                # We need to re-optimize center and beta? That's expensive.
                # Approximation: Keep center and beta fixed, just compute new D.
                # If D increases, S increases. If D decreases, S might decrease.
                # But center might shift. Let's just compute S with fixed center/beta first as a filter.
                
                new_pts = [p[:] for p in current_pts]
                new_pts[i] = [new_x, new_y]
                
                # Fixed center/beta S
                S_fixed = calc_S(new_pts, current_beta, current_center[0], current_center[1])
                
                # If fixed S is significantly better, re-optimize center/beta
                if S_fixed < current_S * 0.99: # 1% improvement threshold
                    S_new, beta_new, center_new = optimize_center(new_pts, current_beta)
                    
                    if S_new < current_S:
                        current_S = S_new
                        current_pts = new_pts
                        current_rots = rots[:] # rots didn't change in this simple version, but we tried dr
                        current_rots[i] = new_r
                        current_beta = beta_new
                        current_center = center_new
                        
                        # Update global best
                        if current_S < best_global_S:
                            best_global_S = current_S
                            best_global_pts = [p[:] for p in current_pts]
                            best_global_rotations = current_rots[:]
                            best_global_beta = current_beta
                            best_global_center = current_center
                else:
                    # Revert
                    pass # We didn't modify current_pts in place, so nothing to revert
            
            # Final polish on the best found
            if best_global_pts is not None:
                # One more optimization pass
                S, beta, center = optimize_center(best_global_pts, best_global_beta)
                if S < best_global_S:
                    best_global_S = S
                    best_global_beta = beta
                    best_global_center = center

    # --- Strategy 3: Randomized Restart ---
    # Try a few random initial configurations and relax them
    
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        for restart in range(5):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Generate random positions
            pts = []
            rots = []
            for i in range(N):
                x = np.random.uniform(-3, 3)
                y = np.random.uniform(-3, 3)
                r = np.random.uniform(0, 60)
                pts.append([x, y])
                rots.append(r)
            
            # Relax
            # This is a simple local search
            for step in range(100):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                i = np.random.randint(N)
                dx = np.random.uniform(-0.1, 0.1)
                dy = np.random.uniform(-0.1, 0.1)
                dr = np.random.uniform(-2, 2)
                
                new_x = pts[i][0] + dx
                new_y = pts[i][1] + dy
                new_r = rots[i] + dr
                
                valid = True
                for j in range(N):
                    if i == j: continue
                    if hex_overlap((new_x, new_y), pts[j], 1.0, new_r, 1.0, rots[j]):
                        valid = False
                        break
                
                if not valid: continue
                
                new_pts = [p[:] for p in pts]
                new_pts[i] = [new_x, new_y]
                
                # Quick S check with fixed center/beta from previous best
                S_fixed = calc_S(new_pts, best_global_beta, best_global_center[0], best_global_center[1])
                if S_fixed < best_global_S * 0.99:
                    S_new, beta_new, center_new = optimize_center(new_pts, best_global_beta)
                    if S_new < best_global_S:
                        best_global_S = S_new
                        best_global_pts = new_pts
                        best_global_rotations = rots[:]
                        best_global_rotations[i] = new_r
                        best_global_beta = beta_new
                        best_global_center = center_new
                        pts = new_pts
                        rots = best_global_rotations

    # Final Output
    if best_global_pts is None:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    inner = []
    for i in range(len(best_global_pts)):
        x, y = best_global_pts[i]
        r = best_global_rotations[i]
        inner.append([float(x), float(y), float(r)])
    
    center = [float(cx), float(cy)]
    
    # Calculate final side length S
    S = calc_S(best_global_pts, best_global_beta, cx, cy)
    side = float(S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

