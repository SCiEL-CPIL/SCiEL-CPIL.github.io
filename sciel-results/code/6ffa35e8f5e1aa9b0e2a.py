import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute unit hexagon vertices for angle 0
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Normals for outer hexagon (angle 0)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(30, 360, 60)])

    def get_hex_verts(cx, cy, angle_rad):
        """
        Returns vertices of a unit hexagon at (cx, cy) with rotation angle_rad.
        """
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T
        verts += np.array([cx, cy])
        return verts

    def calc_outer_side(params):
        """
        params: [x0..x11, y0..y11, a0..a11, xc, yc]
        Returns min side length of outer hexagon (angle 0) centered at (xc, yc)
        containing all inner hexagons.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:3*N]
        xc, yc = params[3*N], params[3*N+1]
        
        oc = np.array([xc, yc])
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang_rad[i]
            verts = get_hex_verts(cx, cy, a_rad)
            verts_rel = verts - oc
            projs = verts_rel @ outer_normals.T # (6, 6)
            # We need max absolute projection? 
            # The outer hexagon is defined by |p . n_k| <= S * sqrt(3)/2
            # So S = max_k max_i | (p_i - c) . n_k | * 2 / sqrt(3)
            # Note: outer_normals are unit vectors.
            abs_projs = np.abs(projs)
            max_proj_i = np.max(abs_projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
                
        side = max_proj * 2.0 / SQRT3
        return side

    def sat_overlap_depth(poly1, poly2):
        """
        Returns minimum penetration depth if poly1 and poly2 overlap.
        Returns 0.0 if disjoint.
        """
        min_overlap = 0.0
        axes = []
        
        # Axes from poly1 edges
        for k in range(6):
            edge = poly1[(k+1)%6] - poly1[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n/norm)
        
        # Axes from poly2 edges
        for k in range(6):
            edge = poly2[(k+1)%6] - poly2[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n/norm)
                
        for axis in axes:
            proj1 = np.dot(poly1, axis)
            proj2 = np.dot(poly2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0
            if overlap > min_overlap:
                min_overlap = overlap
                
        return min_overlap

    def calc_penalty(params):
        """
        Returns sum of squared overlap depths for all pairs.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:3*N]
        
        polys = []
        for i in range(N):
            cx, cy = pos[i]
            verts = get_hex_verts(cx, cy, ang_rad[i])
            polys.append(verts)
            
        penalty = 0.0
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]
                
                # Quick bounding box check
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue
                
                depth = sat_overlap_depth(p1, p2)
                if depth > 0:
                    penalty += depth * depth * 100.0 # High weight
                    
        return penalty

    def objective(params):
        side = calc_outer_side(params)
        penalty = calc_penalty(params)
        return side + penalty

    # --- GENERATE INITIAL CONFIGURATIONS ---
    
    inits = []
    
    # 1. Honeycomb lattice (3x4 grid)
    # Lattice vectors for hexagons:
    # v1 = (1.5, sqrt(3)/2), v2 = (0, sqrt(3))
    # To fit 12, we can use a 3x4 arrangement.
    # Let's generate a 3x4 grid of lattice points and center it.
    lattice_points = []
    for i in range(3):
        for j in range(4):
            x = i * 1.5
            y = i * SQRT3 / 2.0 + j * SQRT3
            lattice_points.append((x, y))
    
    init_pos = np.array(lattice_points)
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    init_center = np.array([0.0, 0.0])
    inits.append(np.concatenate([init_pos.flatten(), init_ang, init_center]))
    
    # 2. Ring configuration (1 center, 6 inner ring, 5 outer?) 
    # Let's try 1 center, 6 around, 5 more in a ring.
    # Actually, just a ring of 12 is a good start.
    ring_pos = []
    ring_ang = []
    r = 2.5
    for i in range(N):
        a = 2 * math.pi * i / N
        ring_pos.append([r * math.cos(a), r * math.sin(a)])
        ring_ang.append(a)
    ring_pos = np.array(ring_pos)
    ring_pos -= np.mean(ring_pos, axis=0)
    inits.append(np.concatenate([ring_pos.flatten(), np.array(ring_ang), np.array([0.0, 0.0])]))
    
    # 3. Random starts
    np.random.seed(42)
    for _ in range(10):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, 2*math.pi, N)
        center = np.random.uniform(-1, 1, 2)
        inits.append(np.concatenate([pos.flatten(), ang, center]))
        
    # 4. Distorted lattice
    for _ in range(5):
        pos = np.array(lattice_points)
        pos += np.random.normal(0, 0.2, pos.shape)
        pos -= np.mean(pos, axis=0)
        ang = np.random.normal(0, 0.1, N)
        center = np.random.uniform(-0.5, 0.5, 2)
        inits.append(np.concatenate([pos.flatten(), ang, center]))

    best_side = float('inf')
    best_params = None

    # --- OPTIMIZATION ---
    
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_side:
                # Check validity
                penalty = calc_penalty(res.x)
                if penalty < 1e-4:
                    best_side = calc_outer_side(res.x)
                    best_params = res.x
        except Exception as e:
            pass

    # Fallback
    if best_params is None:
        # Use the initial lattice
        best_params = inits[0]
        best_side = calc_outer_side(best_params)

    # Construct return values
    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:3*N]
    final_center = best_params[3*N:3*N+2]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(final_center), best_side, 0.0

BET: {mechanism: "Replaces slow soft-penalty SA with fast Nelder-Mead local search on high-quality lattice initializations, directly minimizing outer side length with hard-constraint verification, likely achieving side length ~4.5-5.0 (Score ~1.6-1.8) vs Seed 8.0 (Score 1.0) if Score=8/Value, or similar improvement ratio.", q: 0.9, delta: [0.5, 2.0]}
