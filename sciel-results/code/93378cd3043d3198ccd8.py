import numpy as np
from scipy.optimize import minimize
import time

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Support function of a unit regular hexagon
    def h_unit(phi):
        phi = phi % (2 * np.pi)
        angles = np.arange(6) * np.pi / 3
        return np.max(np.cos(phi - angles))

    # Outer normals for Theta = 0
    # Outer hexagon side S, center 0.
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    outer_normals = np.array([
        [np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))] for k in range(6)
    ])
    
    def compute_constraints_and_penalty(vars, beta=10.0):
        # vars: [x1, y1, t1, x2, y2, t2, ..., x12, y12, t12, S]
        S = vars[-1]
        if S <= 0:
            return 1e10, None
        
        centers = vars[:2*N].reshape(N, 2)
        thetas = vars[2*N:3*N]
        
        penalty = 0.0
        
        # 1. Containment Penalty
        # For each i, for each outer normal k
        # c_i . n_k + h_i(n_k) <= S * h_outer(n_k)
        # h_outer(n_k) = S * (SQRT3/2) is not right.
        # Outer hexagon support in direction n_k (normal) is S * (SQRT3/2) / (SQRT3/2) ?
        # Side length S. Apothem = S * SQRT3 / 2.
        # Support function of outer hexagon in direction n (unit normal) is Apothem = S * SQRT3 / 2.
        # Wait, the support function h(n) is the max dot product.
        # For a regular hexagon of side S centered at origin:
        # Vertices at distance R = S.
        # Normals at angles 30, 90...
        # h(n) for n being a normal is the apothem = S * SQRT3 / 2.
        # So constraint: c_i . n_k + h_unit(n_k - theta_i) <= S * SQRT3 / 2
        
        for i in range(N):
            c_i = centers[i]
            t_i = thetas[i]
            for k in range(6):
                n = outer_normals[k]
                lhs = np.dot(c_i, n) + h_unit(np.arctan2(n[1], n[0]) - t_i)
                rhs = S * SQRT3 / 2.0
                if lhs > rhs:
                    penalty += (lhs - rhs) ** 2
        
        # 2. Non-Overlap Penalty
        # For each pair i, j
        # Check separation in directions of normals of i and j
        for i in range(N):
            for j in range(i + 1, N):
                c_i = centers[i]
                c_j = centers[j]
                t_i = thetas[i]
                t_j = thetas[j]
                
                diff = c_i - c_j
                
                # Normals for i
                normals_i = np.array([
                    [np.cos(t_i + np.radians(30 + 60*l)), np.sin(t_i + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                # Normals for j
                normals_j = np.array([
                    [np.cos(t_j + np.radians(30 + 60*l)), np.sin(t_j + np.radians(30 + 60*l))] 
                    for l in range(6)
                ])
                
                all_normals = np.vstack([normals_i, normals_j])
                
                # Calculate gaps
                gaps = []
                for n in all_normals:
                    # Gap = (c_i - c_j) . n - h_i(n) - h_j(-n)
                    # h_i(n) = h_unit(angle(n) - t_i)
                    # h_j(-n) = h_unit(angle(-n) - t_j) = h_unit(angle(n) + pi - t_j)
                    # Due to symmetry h_unit(x + pi) = h_unit(x) for hexagon?
                    # h_unit(phi) = max cos(phi - k*60).
                    # h_unit(phi + pi) = max cos(phi + pi - k*60) = max -cos(phi - k*60) = -min cos(phi - k*60).
                    # This is NOT symmetric. h(n) != h(-n) generally?
                    # Wait, regular hexagon is centrally symmetric. h(-n) = h(n).
                    # Let's check: h_unit(0) = 1. h_unit(pi) = max cos(pi - 0), cos(pi-60)...
                    # cos(pi) = -1. cos(120) = -0.5. Max is -0.5? No.
                    # cos(180-0) = -1. cos(180-60) = cos(120) = -0.5.
                    # cos(180-120) = cos(60) = 0.5.
                    # Max is 0.5?
                    # Let's re-evaluate h_unit.
                    # Vertices at 0, 60, 120, 180, 240, 300.
                    # h(0) = max(1, 0.5, -0.5, -1, -0.5, 0.5) = 1.
                    # h(pi) = max(-1, -0.5, 0.5, 1, 0.5, -0.5) = 1.
                    # Yes, h(-n) = h(n).
                    
                    angle_n = np.arctan2(n[1], n[0])
                    h_i_val = h_unit(angle_n - t_i)
                    h_j_val = h_unit(angle_n - t_j) # h_j(-n) = h_j(n)
                    
                    gap = np.dot(diff, n) - h_i_val - h_j_val
                    gaps.append(gap)
                
                gaps = np.array(gaps)
                min_gap = np.min(gaps)
                if min_gap < 0:
                    penalty += min_gap ** 2
        
        return penalty, (centers, thetas, S)

    def objective(vars):
        if time.time() > deadline:
            return 1e10
        penalty, _ = compute_constraints_and_penalty(vars)
        S = vars[-1]
        return S + 100.0 * penalty

    def is_valid(vars):
        penalty, (centers, thetas, S) = compute_constraints_and_penalty(vars)
        return penalty < 1e-6 and S > 0

    best_score = np.inf
    best_vars = None

    # Initializations
    inits = []
    
    # 1. Hexagonal Lattice
    # Approximate side length for 12 hexes in a 4x3 grid?
    # Distance between centers is SQRT3.
    # Rows: 4, 3, 4, 1? No, 12 hexes.
    # Standard hex packing:
    # Row 0: 4 hexes
    # Row 1: 3 hexes (shifted)
    # Row 2: 4 hexes
    # Row 3: 1 hex? No.
    # Let's just use a 4x3 rectangular grid of centers with hex spacing.
    # Spacing x: SQRT3, y: 1.5
    centers_init = []
    for r in range(3):
        for c in range(4):
            x = c * SQRT3
            y = r * 1.5
            if r % 2 == 1:
                x += SQRT3 / 2.0
            centers_init.append([x, y])
    centers_init = np.array(centers_init)
    centers_init -= centers_init.mean(axis=0)
    
    thetas_init = np.zeros(N)
    S_init = 5.0 # Guess
    
    vars_init = np.concatenate([centers_init.flatten(), thetas_init, [S_init]])
    inits.append(vars_init)
    
    # 2. Random starts
    for _ in range(5):
        centers_rand = np.random.uniform(-3, 3, (N, 2))
        thetas_rand = np.random.uniform(0, np.pi/3, N)
        S_rand = 4.0
        vars_rand = np.concatenate([centers_rand.flatten(), thetas_rand, [S_rand]])
        inits.append(vars_rand)

    for v0 in inits:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                objective,
                v0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if is_valid(res.x):
                S_val = res.x[-1]
                if S_val < best_score:
                    best_score = S_val
                    best_vars = res.x.copy()
                    
        except Exception:
            pass

    if best_vars is None:
        # Fallback: Return a known valid configuration if optimization failed
        # Use the lattice init and manually adjust S to be large enough
        centers, thetas, S = compute_constraints_and_penalty(inits[0])[1]
        # Calculate required S for this config
        max_req = 0
        for i in range(N):
            c_i = centers[i]
            t_i = thetas[i]
            for k in range(6):
                n = outer_normals[k]
                lhs = np.dot(c_i, n) + h_unit(np.arctan2(n[1], n[0]) - t_i)
                req_s = lhs / (SQRT3 / 2.0)
                if req_s > max_req:
                    max_req = req_s
        S = max_req * 1.01 # Add margin
        best_vars = np.concatenate([centers.flatten(), thetas, [S]])
        best_score = S

    centers, thetas, S = compute_constraints_and_penalty(best_vars)[1]
    
    inner = []
    for i in range(N):
        inner.append([centers[i, 0], centers[i, 1], np.degrees(thetas[i])])
    
    outer_center = (0.0, 0.0)
    outer_side = S
    outer_angle_deg = 0.0

    return inner, outer_center, outer_side, outer_angle_deg
