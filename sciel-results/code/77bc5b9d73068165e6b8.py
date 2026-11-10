import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    def get_hex_vertices(cx, cy, s, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        vertices = []
        for i in range(6):
            a = angle_rad + i * np.pi / 3
            vertices.append([cx + s * np.cos(a), cy + s * np.sin(a)])
        return np.array(vertices)

    def get_hex_normals(angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        normals = []
        for i in range(6):
            a = angle_rad + np.pi/6 + i * np.pi / 3
            normals.append([np.cos(a), np.sin(a)])
        return np.array(normals)

    def point_in_hex_signed_dist(p, hex_normals, apothem):
        dists = apothem - np.dot(hex_normals, p)
        min_dist = np.min(dists)
        if min_dist < 0:
            return min_dist
        else:
            return 0.0

    def unpack(x):
        ic = x[:22].reshape(11, 2)
        ia = x[22:33]
        oc = x[33:35]
        oa = x[35:36][0]
        S = x[36:37][0]
        return ic, ia, oc, oa, S

    def penalty(x):
        if time.time() > deadline:
            return 1e100
        
        ic, ia, oc, oa, S = unpack(x)
        if S < np.sqrt(11):
            return 1e100 + (np.sqrt(11) - S) * 1000
            
        outer_normals = get_hex_normals(oa)
        outer_apothem = S * np.sqrt(3) / 2
        
        total_pen = 0.0
        
        # Containment penalty
        for i in range(11):
            inner_verts = get_hex_vertices(ic[i,0], ic[i,1], 1.0, ia[i])
            for v in inner_verts:
                v_rel = v - oc
                d = point_in_hex_signed_dist(v_rel, outer_normals, outer_apothem)
                if d < 0:
                    total_pen += d * d
        
        # Separation penalty
        for i in range(11):
            for j in range(i+1, 11):
                vi = get_hex_vertices(ic[i,0], ic[i,1], 1.0, ia[i])
                vj = get_hex_vertices(ic[j,0], ic[j,1], 1.0, ia[j])
                ni = get_hex_normals(ia[i])
                nj = get_hex_normals(ia[j])
                ai = np.sqrt(3) / 2
                aj = np.sqrt(3) / 2
                
                # Check vertices of i against j
                for v in vi:
                    v_rel = v - ic[j]
                    d = point_in_hex_signed_dist(v_rel, nj, aj)
                    if d < 0:
                        total_pen += d * d
                        
                # Check vertices of j against i
                for v in vj:
                    v_rel = v - ic[i]
                    d = point_in_hex_signed_dist(v_rel, ni, ai)
                    if d < 0:
                        total_pen += d * d
                        
        return S + 100.0 * total_pen

    best_x = None
    best_val = float('inf')
    
    # Initial guesses
    guesses = []
    
    # Guess 1: Hex grid
    centers = [[0, 0]]
    for k in range(6):
        angle = k * np.pi / 3
        centers.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle)])
    for k in range(4):
        angle = k * np.pi / 2 + np.pi/4
        centers.append([2 * np.sqrt(3) * np.cos(angle), 2 * np.sqrt(3) * np.sin(angle)])
    ic0 = np.array(centers)
    ia0 = np.zeros(11)
    oc0 = np.array([0, 0])
    oa0 = 0.0
    S0 = 5.0
    x0 = np.concatenate([ic0.flatten(), ia0, oc0, [oa0], [S0]])
    guesses.append(x0)
    
    # Guess 2: Random perturbation of Guess 1
    np.random.seed(42)
    x1 = x0 + np.random.normal(0, 0.5, len(x0))
    guesses.append(x1)
    
    # Guess 3: More random
    x2 = x0 + np.random.normal(0, 1.0, len(x0))
    guesses.append(x2)
    
    for x_init in guesses:
        if time.time() > deadline:
            break
            
        try:
            result = minimize(penalty, x_init, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            pass

    if best_x is None:
        # Fallback to seed
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    ic, ia, oc, oa, S = unpack(best_x)
    if S < np.sqrt(11):
        S = np.sqrt(11)
        
    inner_hex_data = np.column_stack((ic[:,0], ic[:,1], ia))
    outer_hex_data = np.array([oc[0], oc[1], oa])
    outer_hex_side_length = S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
