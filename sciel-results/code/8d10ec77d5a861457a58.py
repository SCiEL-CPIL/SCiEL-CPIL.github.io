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
    
    # Topo 5: Compact "Hexagon-like"
    # Center (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: Pick 5. Let's pick the ones that extend the shape evenly.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) -- this is a bit asymmetric.
    # Let's try a "Cross"
    t5 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (-2,0), (0,2), (0,-2), (2,-1) # 5 more
    ]
    # Check uniqueness
    if len(set(t5)) == 12:
        topologies.append(t5)
        
    # Topo 6: Another variation
    t6 = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    if len(set(t6)) == 12:
        topologies.append(t6)
        
    best_S = float('inf')
    best_params = None
    best_topology_idx = 0
    
    # Strategy:
    # 1. For each topology, do a coarse grid search on alpha, beta.
    # 2. Keep the top 3 candidates from the grid.
    # 3. Refine the top candidates with Nelder-Mead on alpha, beta.
    # 4. Refine the best alpha, beta with Nelder-Mead on cx, cy.
    
    for t_idx, topo in enumerate(topologies):
        if time.time() > deadline - 20:
            break
            
        positions = get_positions(topo, scale=2.0)
        
        # Coarse Grid Search
        # Symmetry: period 60 degrees
        # Step 1.0 degree for coarse, then refine
        step = 1.0
        candidates = [] # list of (S, alpha, beta)
        
        # We can optimize the grid by noting that alpha and beta are somewhat independent in the projection sum, 
        # but they interact. 
        # Let's just do a full grid.
        for a in np.arange(0, 60, step):
            if time.time() > deadline - 10:
                break
            for b in np.arange(0, 60, step):
                S = calc_S(positions, a, b)
                candidates.append((S, a, b))
        
        if not candidates:
            continue
            
        # Sort and take top 5
        candidates.sort(key=lambda x: x[0])
        
        for S_cand, a_cand, b_cand in candidates[:5]:
            if time.time() > deadline - 5:
                break
                
            # Refine alpha, beta
            def obj_ab(params):
                a, b = params
                return calc_S(positions, a, b)
            
            try:
                res_ab = minimize(obj_ab, [a_cand, b_cand], method='Nelder-Mead', 
                                  options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                if res_ab.fun < S_cand:
                    S_cand = res_ab.fun
                    a_cand, b_cand = res_ab.x
            except:
                pass
            
            # Refine center cx, cy
            def obj_c(params):
                cx, cy = params
                # Shift positions
                pos_shifted = positions + np.array([cx, cy])
                return calc_S(pos_shifted, a_cand, b_cand)
            
            try:
                res_c = minimize(obj_c, [0.0, 0.0], method='Nelder-Mead', 
                                 options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
                if res_c.fun < S_cand:
                    S_cand = res_c.fun
            except:
                pass
            
            if S_cand < best_S:
                best_S = S_cand
                best_params = (positions, a_cand, b_cand, res_c.x[0], res_c.x[1])
                best_topology_idx = t_idx

    # If we didn't find anything (should not happen), fallback
    if best_params is None:
        # Fallback to a simple configuration
        positions = get_positions(topologies[0], scale=2.0)
        best_S = calc_S(positions, 0, 0)
        best_params = (positions, 0.0, 0.0, 0.0, 0.0)
        
    positions, alpha, beta, cx, cy = best_params
    
    # Final polish with very tight tolerance on all parameters if time allows
    if time.time() < deadline - 2:
        def obj_all(params):
            cx, cy, alpha, beta = params
            pos_shifted = positions + np.array([cx, cy])
            return calc_S(pos_shifted, alpha, beta)
            
        try:
            res_all = minimize(obj_all, [cx, cy, alpha, beta], method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res_all.fun < best_S:
                best_S = res_all.fun
                cx, cy, alpha, beta = res_all.x
        except:
            pass
            
    # Construct final positions
    final_positions = positions + np.array([cx, cy])
    
    inner = []
    for x, y in final_positions:
        inner.append([x, y, alpha])
        
    return inner, (0.0, 0.0), best_S, beta

# EVOLVE-BLOCK-END
