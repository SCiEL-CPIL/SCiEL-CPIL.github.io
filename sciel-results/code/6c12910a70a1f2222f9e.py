import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute unit hexagon vertices for angle 0
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_outer_side(params, outer_center=(0.0, 0.0), outer_angle=0.0):
        """
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Returns min side length of outer hexagon containing all inner hexagons.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        theta_rad = math.radians(outer_angle)
        # Normals for the outer hexagon sides
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        oc = np.array(outer_center)
        
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang_rad[i]
            verts = get_hex_verts(cx, cy, 1.0, np.degrees(a_rad))
            verts_rel = verts - oc
            projs = verts_rel @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        side = max_proj * 2.0 / SQRT3
        return side

    # Generate a compact lattice configuration
    # Hexagonal lattice vectors for touching unit hexagons
    # Distance between centers is sqrt(3)
    a1 = np.array([SQRT3, 0])
    a2 = np.array([SQRT3/2, 1.5])
    
    lattice_points = []
    # Generate points in a grid of lattice indices
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * a1 + j * a2
            lattice_points.append(p)
    
    lattice_points = np.array(lattice_points)
    # Sort by distance from origin to pick the most central compact cluster
    dists = np.linalg.norm(lattice_points, axis=1)
    indices = np.argsort(dists)
    # Pick 12 closest points
    init_pos = lattice_points[indices[:N]]
    
    # Center the cluster
    centroid = np.mean(init_pos, axis=0)
    init_pos = init_pos - centroid
    
    init_ang = np.zeros(N)
    
    best_params = np.concatenate([init_pos.flatten(), init_ang])
    
    # Optimize the outer container center and angle
    # We fix the inner positions to the lattice cluster initially.
    # If we allow inner positions to move, we might break the valid packing.
    # So we only optimize outer_center and outer_angle.
    
    def outer_obj(outer_params):
        oc = outer_params[:2]
        oa = outer_params[2] # radians
        side = calc_outer_side(best_params, outer_center=oc, outer_angle=np.degrees(oa))
        return side

    # Initial guess for outer center is (0,0) and angle 0
    x0_outer = np.array([0.0, 0.0, 0.0])
    
    best_side = float('inf')
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0
    
    # Try multiple starting angles for the outer hexagon to avoid local minima
    # The outer hexagon has 6-fold symmetry, so 0 to 60 degrees is enough.
    # But the inner cluster might not be symmetric, so we check a few.
    
    for start_ang in [0, 15, 30, 45, 60]:
        if time.time() > deadline - 10:
            break
        x0_outer = np.array([0.0, 0.0, math.radians(start_ang)])
        try:
            res = minimize(
                outer_obj, 
                x0_outer, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_outer_center = res.x[:2]
                best_outer_angle = np.degrees(res.x[2])
        except:
            pass

    # If we have time, try a slight local search on inner positions to see if we can shrink
    # This is risky as it might introduce overlaps.
    # We will skip this for safety unless the current side is very large.
    
    # Fallback if optimization failed
    if best_side == float('inf'):
        best_side = calc_outer_side(best_params, outer_center=[0,0], outer_angle=0)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle
