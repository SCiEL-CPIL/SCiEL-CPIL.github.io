"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    def support_unit(n_vec):
        # n_vec is a numpy array [nx, ny]
        # Returns max dot product of unit_verts with n_vec
        return np.max(unit_verts @ n_vec)
        
    def calc_S(positions, alpha_deg, beta_deg):
        # positions: numpy array of shape (N, 2)
        # alpha_deg: rotation of inner hexagons
        # beta_deg: rotation of outer hexagon
        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)
        
        # Outer normals
        # Outer hexagon angle beta.
        # Normals are at 30 + beta + k*60
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30) + beta_rad + k * math.radians(60)
            n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # Support of inner hexagons in direction n
            # Inner hex i is at P_i, rotated by alpha.
            # Support = P_i . n + h_unit(R_-alpha n)
            
            # Rotate n by -alpha
            c = math.cos(-alpha_rad)
            s = math.sin(-alpha_rad)
            n_rot = np.array([c*n[0] - s*n[1], s*n[0] + c*n[1]])
            
            h_unit_val = support_unit(n_rot)
            
            # Vectorized projection of all centers
            # positions is (N, 2), n is (2,)
            projs = positions @ n + h_unit_val
            max_p_proj = np.max(projs)
            
            if max_p_proj > max_proj:
                max_proj = max_p_proj
                
        return max_proj * 2.0 / sqrt3

    # Define topologies
    # Axial to Cartesian: x = s(q+r/2), y = s(r*sqrt3/2)
    def get_positions(topo, scale=2.0):
        pts = []
        for q, r in topo:
            x = scale * (q + r/2.0)
            y = scale * (r * sqrt3/2.0)
            pts.append((x, y))
        # Center
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        return np.array([(x-cx, y-cy) for x, y in pts])
        
    topologies = []
    
    # Topo 1: 3x4 Parallelogram
    t1 = [(q, r) for q in range(3) for r in range(4)]
    topologies.append(t1)
    
    # Topo 2: 4x3 Parallelogram
    t2 = [(q, r) for q in range(4) for r in range(3)]
    topologies.append(t2)
    
    # Topo 3: 2x6 Parallelogram
    t3 = [(q, r) for q in range(2) for r in range(6)]
    topologies.append(t3)
    
    # Topo 4: 6x2 Parallelogram
    t4 = [(q, r) for q in range(6) for r in range(2)]
    topologies.append(t4)
    
    # Topo 5: Compact "Hexagon-like" cross
    t5 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2), (2,-1)
    ]
    if len(set(t5)) == 12:
        topologies.append(t5)
        
    # Topo 6: Hexagonal cluster
    t6 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    if len(set(t6)) == 12:
        topologies.append(t6)
    
    # Topo 7: Linear chain
    t7 = [(q, 0) for q in range(12)]
    topologies.append(t7)
    
    # Topo 8: 3x4 with offset
    t8 = [(q, r) for q in range(3) for r in range(4)]
    topologies.append(t8)
    
    # Topo 9: 4x3 with offset  
    t9 = [(q, r) for q in range(4) for r in range(3)]
    topologies.append(t9)
    
    # Topo 10: Diamond shape
    t10 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1), (2,-1)
    ]
    if len(set(t10)) == 12:
        topologies.append(t10)
        
    best_S = float('inf')
    best_positions = None
    best_alpha = 0.0
    best_beta = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Multi-start SLSQP optimization
    # Parameters: [cx, cy, alpha, beta]
    # We fix the topology positions and optimize center, alpha, beta
    
    def objective(params):
        cx, cy, alpha, beta = params
        pos_shifted = base_positions + np.array([cx, cy])
        return calc_S(pos_shifted, alpha, beta)
    
    for t_idx, topo in enumerate(topologies):
        if time.time() > deadline - 10:
            break
            
        base_positions = get_positions(topo, scale=2.0)
        
        # Multi-start: try several initial guesses for alpha, beta
        initial_guesses = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 30.0, 30.0],
            [0.0, 0.0, 15.0, 15.0],
            [0.0, 0.0, 45.0, 45.0],
            [0.0, 0.0, 10.0, 20.0],
            [0.0, 0.0, 20.0, 10.0],
            [0.0, 0.0, 0.0, 30.0],
            [0.0, 0.0, 30.0, 0.0],
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(objective, guess, method='SLSQP', 
                              options={'maxiter': 500, 'ftol': 1e-10})
                if res.success and res.fun < best_S:
                    best_S = res.fun
                    best_positions = base_positions.copy()
                    best_cx, best_cy, best_alpha, best_beta = res.x
            except:
                pass
    
    # Fallback if SLSQP failed
    if best_positions is None:
        base_positions = get_positions(topologies[0], scale=2.0)
        best_S = calc_S(base_positions, 0, 0)
        best_cx, best_cy, best_alpha, best_beta = 0.0, 0.0, 0.0, 0.0
    
    # Final high-precision polish with Nelder-Mead
    if time.time() < deadline - 2 and best_positions is not None:
        def obj_all(params):
            cx, cy, alpha, beta = params
            pos_shifted = best_positions + np.array([cx, cy])
            return calc_S(pos_shifted, alpha, beta)
            
        try:
            res_all = minimize(obj_all, [best_cx, best_cy, best_alpha, best_beta], 
                               method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_all.fun < best_S:
                best_S = res_all.fun
                best_cx, best_cy, best_alpha, best_beta = res_all.x
        except:
            pass
            
    # Construct final positions
    final_positions = best_positions + np.array([best_cx, best_cy])
    
    inner = []
    for x, y in final_positions:
        inner.append([x, y, best_alpha])
        
    return inner, (0.0, 0.0), best_S, best_beta

# EVOLVE-BLOCK-END
