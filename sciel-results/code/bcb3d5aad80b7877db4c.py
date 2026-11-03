"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import random

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

# Precompute unit hexagon vertices and edge normals (angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 300
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Edge normals for unit hexagon at angle 0
# Edges connect v[i] to v[i+1]
# Normal is perpendicular to edge, pointing outwards
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = np.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()

    unit_hex_verts = _UNIT_HEX_VERTS
    edge_normals = _EDGE_NORMALS

    def get_all_vertices(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        verts_list = []
        centers = np.zeros((N, 2))
        angles_rad = np.zeros(N)
        
        for i in range(N):
            x, y, a = params[i*3], params[i*3+1], params[i*3+2]
            centers[i] = [x, y]
            rad = np.deg2rad(a)
            angles_rad[i] = rad
            rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            verts = unit_hex_verts @ rot.T + np.array([x, y])
            verts_list.append(verts)
        return verts_list, centers, angles_rad

    def hexagons_disjoint_fast(verts_list, centers, angles_rad, eps=1e-7):
        # Precompute rotated normals for each hexagon
        # normals[i] is (6, 2)
        normals_list = []
        for i in range(N):
            rad = angles_rad[i]
            rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            normals = edge_normals @ rot.T
            normals_list.append(normals)

        for i in range(N):
            v1 = verts_list[i]
            n1 = normals_list[i]
            c1 = centers[i]
            
            # Bounding box of v1
            min1 = np.min(v1, axis=0)
            max1 = np.max(v1, axis=0)

            for j in range(i+1, N):
                v2 = verts_list[j]
                n2 = normals_list[j]
                c2 = centers[j]
                
                # Quick reject using center distance
                # Max distance between centers for touching hexagons is sqrt(3) * 2? 
                # No, distance between centers of touching hexagons is sqrt(3).
                # If dist > sqrt(3) + 1 (margin), they are definitely disjoint?
                # Actually, the max extent is 1. So if dist > sqrt(3) + 2, disjoint.
                # But we can use a tighter bound.
                # If dist > 2.0, they are disjoint (since radius is 1).
                # Wait, radius (center to vertex) is 1.
                # So if dist > 2.0, they are disjoint.
                dist = np.linalg.norm(c1 - c2)
                if dist > 2.0 + eps:
                    continue

                # Bounding box check for v2
                min2 = np.min(v2, axis=0)
                max2 = np.max(v2, axis=0)
                
                # If bounding boxes don't overlap, disjoint
                if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
                    continue

                # SAT check
                overlap = True
                # Check normals of v1
                for k in range(6):
                    n = n1[k]
                    proj1 = v1 @ n
                    proj2 = v2 @ n
                    min1p, max1p = np.min(proj1), np.max(proj1)
                    min2p, max2p = np.min(proj2), np.max(proj2)
                    if max1p <= min2p + eps or max2p <= min1p + eps:
                        overlap = False
                        break
                
                if not overlap:
                    continue

                # Check normals of v2
                for k in range(6):
                    n = n2[k]
                    proj1 = v1 @ n
                    proj2 = v2 @ n
                    min1p, max1p = np.min(proj1), np.max(proj1)
                    min2p, max2p = np.min(proj2), np.max(proj2)
                    if max1p <= min2p + eps or max2p <= min1p + eps:
                        overlap = False
                        break
                
                if overlap:
                    return False
        return True

    def compute_min_outer_side(verts_list):
        sqrt3 = np.sqrt(3)
        dirs = np.array([
            [1.0, 0.0],
            [sqrt3/2.0, 0.5],
            [-sqrt3/2.0, 0.5]
        ])
        
        max_a = 0.0
        for verts in verts_list:
            dots = verts @ dirs.T
            abs_dots = np.abs(dots)
            max_a = max(max_a, np.max(abs_dots))
            
        S = 2.0 * max_a / sqrt3
        return S

    def objective(params):
        verts_list, centers, angles_rad = get_all_vertices(params)
        
        if not hexagons_disjoint_fast(verts_list, centers, angles_rad, eps=1e-7):
            return 1000.0 + compute_min_outer_side(verts_list)
        
        return compute_min_outer_side(verts_list)

    def generate_lattice_config():
        # Generate 12 closest points in hex lattice to origin
        # Lattice: x = sqrt(3)(q + r/2), y = 1.5 r
        # Distance squared: 3(q^2 + qr + r^2)
        points = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                d2 = 3 * (q*q + q*r + r*r)
                x = np.sqrt(3) * (q + r/2.0)
                y = 1.5 * r
                points.append((d2, x, y))
        
        points.sort(key=lambda x: x[0])
        init_points = points[:N]
        
        params = []
        for _, x, y in init_points:
            params.extend([x, y, 0.0])
        return np.array(params)

    def generate_flower_config():
        # 1 center, 6 neighbors, 5 from next ring
        params = [0.0, 0.0, 0.0]
        # 6 neighbors at distance sqrt(3)
        for k in range(6):
            ang = np.deg2rad(k * 60)
            x = np.sqrt(3) * np.cos(ang)
            y = np.sqrt(3) * np.sin(ang)
            params.extend([x, y, 0.0])
        
        # 5 from next ring (distance 3? No, ring 2 distance is sqrt(3)*sqrt(3)=3? 
        # Axial (1,1) dist is sqrt(3). (2,0) dist is 2*sqrt(3).
        # Let's just pick 5 random points from ring 2 or extend the lattice
        # Better: pick 5 more from the lattice that are not in the first 7
        # We already have 7. Let's add 5 more from the lattice list
        lattice_params = generate_lattice_config()
        # The lattice config has 12 points. The flower has 7.
        # Let's just use the lattice config for simplicity and robustness
        return lattice_params

    best_result = None
    best_score = float('inf')
    
    # Initialize with lattice config
    init_params = generate_lattice_config()
    best_result = init_params
    best_score = objective(init_params)
    
    # Run Differential Evolution
    def time_check():
        return (time.time() - t0) < DEADLINE

    try:
        bounds = [(-5, 5), (-5, 5), (0, 60)] * N
        
        # Use a callback to stop if time runs out
        # DE doesn't have a direct time limit, so we wrap it or use a small maxiter
        # Let's estimate: 20 pop * 100 iter = 2000 evals.
        # If each eval is 1ms, that's 2s.
        # We can do more.
        
        de_result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=30, 
            tol=1e-4, 
            seed=42,
            polish=False,
            update_interval=10,
            callback=lambda xk, convergence: False if not time_check() else None,
            init='sobol' # Better initialization
        )
        
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_result = de_result.x
            
    except Exception as e:
        pass

    # Polish with Nelder-Mead
    if best_result is not None and time_check():
        try:
            nm_result = minimize(
                objective, 
                best_result, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
            )
            if nm_result.fun < best_score:
                best_score = nm_result.fun
                best_result = nm_result.x
        except:
            pass

    # Final Check
    verts_list, centers, angles_rad = get_all_vertices(best_result)
    if not hexagons_disjoint_fast(verts_list, centers, angles_rad, eps=1e-6):
        # Fallback to lattice if invalid
        best_result = generate_lattice_config()
        best_score = objective(best_result)

    inner = []
    for i in range(N):
        x = best_result[i*3]
        y = best_result[i*3+1]
        a = best_result[i*3+2]
        inner.append([float(x), float(y), float(a)])
        
    S_final = compute_min_outer_side(get_all_vertices(best_result)[0])
    
    center = (0.0, 0.0)
    side = float(S_final)
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
