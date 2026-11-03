import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute normals for inner hexagons (relative to each other)
    inner_normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])
    
    # Precompute base outer normals (for beta=0)
    base_outer_normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])

    def get_outer_normals(beta):
        # Rotate base normals by beta
        c = math.cos(beta)
        s = math.sin(beta)
        R = np.array([[c, -s], [s, c]])
        return base_outer_normals @ R.T

    def calc_exact_S(pts, beta, cx, cy):
        """
        Calculate exact minimum side length S for given points, rotation beta, and center.
        Returns (S, cx, cy) or (inf, None, None) if infeasible.
        """
        normals = get_outer_normals(beta)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                S = (D + APOTHEM) / APOTHEM
                return S, res.x[0], res.x[1]
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def check_overlap_exact(pts):
        """
        Check if any two hexagons overlap.
        Returns True if no overlap.
        """
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ inner_normals.T)
                if np.min(projs) < math.sqrt(3) - 1e-9:
                    return False
        return True

    def smooth_objective(params, temperature):
        """
        Smooth surrogate for the optimization.
        params: [x1, y1, ..., xN, yN, cx, cy, beta]
        """
        n_pts = N
        pts = params[:2*n_pts].reshape(n_pts, 2)
        cx = params[2*n_pts]
        cy = params[2*n_pts+1]
        beta = params[2*n_pts+2]
        
        # Normalize beta to [0, pi/6)
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        normals = get_outer_normals(beta)
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        
        # Max norm approximation using log-sum-exp
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return 1e10
            
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D_smooth = temperature * log_sum
        
        S_smooth = (D_smooth + APOTHEM) / APOTHEM
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs_in = np.abs(d @ inner_normals.T)
                min_proj = np.min(projs_in)
                if min_proj < math.sqrt(3):
                    violation = math.sqrt(3) - min_proj
                    penalty += violation * 100.0
                    
        return S_smooth + penalty

    def run_annealing(initial_params, T_start, T_end, steps):
        """
        Run simulated annealing.
        Returns best params found.
        """
        current_params = initial_params.copy()
        current_obj = smooth_objective(current_params, T_start)
        
        best_params = current_params.copy()
        best_obj = current_obj
        
        time_per_step = (TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)) / steps
        if time_per_step <= 0:
            return best_params
            
        start_time = time.time()
        
        for i in range(steps):
            if time.time() - start_time > (TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)):
                break
                
            # Annealing schedule (geometric)
            T = T_start * (T_end / T_start) ** (i / steps)
            
            # Perturbation
            # Scale perturbation with temperature
            scale = 0.1 * T / T_start + 0.01
            noise = np.random.normal(0, scale, size=current_params.shape)
            
            # For beta, use smaller noise
            noise[-1] = np.random.normal(0, scale * 0.5)
            
            new_params = current_params + noise
            
            new_obj = smooth_objective(new_params, T)
            
            if new_obj < current_obj or np.random.rand() < math.exp((current_obj - new_obj) / T):
                current_params = new_params
                current_obj = new_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_params = current_params.copy()
                    
        return best_params

    # Initial configurations
    # 1. Lattice cluster
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    pts_init_1 = axial_to_cartesian(axial_1)
    cx_init_1 = np.mean(pts_init_1[:, 0])
    cy_init_1 = np.mean(pts_init_1[:, 1])
    init_1 = np.concatenate([pts_init_1.ravel(), [cx_init_1, cy_init_1, 0.0]])

    # 2. Random start
    np.random.seed(42)
    pts_init_2 = np.random.uniform(-2, 2, (N, 2))
    # Force some separation roughly
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts_init_2[i] - pts_init_2[j]) < 2.0:
                pts_init_2[j] += np.array([2.0, 0.0])
    cx_init_2 = np.mean(pts_init_2[:, 0])
    cy_init_2 = np.mean(pts_init_2[:, 1])
    init_2 = np.concatenate([pts_init_2.ravel(), [cx_init_2, cy_init_2, 0.0]])

    # 3. Another lattice variant
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    pts_init_3 = axial_to_cartesian(axial_3)
    cx_init_3 = np.mean(pts_init_3[:, 0])
    cy_init_3 = np.mean(pts_init_3[:, 1])
    init_3 = np.concatenate([pts_init_3.ravel(), [cx_init_3, cy_init_3, 0.0]])

    best_global_S = float('inf')
    best_global_params = None

    candidates = [init_1, init_2, init_3]
    
    # Allocate time to each candidate
    time_per_candidate = (TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)) / len(candidates)
    
    for idx, init in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Run annealing
        # T_start should be high enough to explore, T_end low to converge
        # Objective scale is roughly O(1) for S, penalty can be large.
        # Let's start T around 1.0 and go down to 0.01
        T_start = 1.0
        T_end = 0.01
        
        # Estimate steps based on time
        # smooth_objective is O(N^2) ~ 121 ops + LP? No, smooth uses vectorized numpy.
        # Let's aim for ~1000-5000 steps depending on speed.
        # We'll run in chunks to check time.
        
        steps = 2000
        if idx == 0:
            steps = 3000 # Give more to the first good candidate
        
        best_params = run_annealing(init, T_start, T_end, steps)
        
        # Extract and verify
        n_pts = N
        pts = best_params[:2*n_pts].reshape(n_pts, 2)
        cx = best_params[2*n_pts]
        cy = best_params[2*n_pts+1]
        beta = best_params[2*n_pts+2]
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        # Local polish with Nelder-Mead on exact S? 
        # Or just use the smooth result and verify with LP.
        # Let's do a quick local polish on the exact objective using the smooth params as start.
        
        def exact_obj(params):
            n_pts = N
            pts = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            S, c_x, c_y = calc_exact_S(pts, beta, cx, cy)
            if S == float('inf'):
                return 1e10
            
            # Penalty for overlap
            penalty = 0.0
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    d = pts[i] - pts[j]
                    projs_in = np.abs(d @ inner_normals.T)
                    min_proj = np.min(projs_in)
                    if min_proj < math.sqrt(3):
                        violation = math.sqrt(3) - min_proj
                        penalty += violation * 100.0
                        
            return S + penalty

        try:
            from scipy.optimize import minimize
            res = minimize(exact_obj, best_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_global_S + 0.1:
                best_params = res.x
                pts = best_params[:2*n_pts].reshape(n_pts, 2)
                cx = best_params[2*n_pts]
                cy = best_params[2*n_pts+1]
                beta = best_params[2*n_pts+2]
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                S, c_x, c_y = calc_exact_S(pts, beta, cx, cy)
                if S < float('inf') and check_overlap_exact(pts):
                    if S < best_global_S:
                        best_global_S = S
                        best_global_params = best_params.copy()
        except Exception:
            pass

    # Final verification and formatting
    if best_global_params is None:
        # Fallback to initial lattice
        best_global_params = init_1
        pts = init_1[:2*N].reshape(N, 2)
        cx = init_1[2*N]
        cy = init_1[2*N+1]
        beta = init_1[2*N+2]
        S, c_x, c_y = calc_exact_S(pts, beta, cx, cy)
        if S < float('inf'):
            best_global_S = S
            best_global_params = np.concatenate([pts.ravel(), [cx, cy, beta]])

    if best_global_params is None:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Extract final best
    pts = best_global_params[:2*N].reshape(N, 2)
    cx = best_global_params[2*N]
    cy = best_global_params[2*N+1]
    beta = best_global_params[2*N+2]
    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
    
    # Recalculate exact S to be safe
    S, c_x, c_y = calc_exact_S(pts, beta, cx, cy)
    if S == float('inf') or not check_overlap_exact(pts):
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Apply tiny safety scale
    SCALE = 1.0 + 1e-12
    pts_scaled = []
    for x, y in pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    # Recalculate S for scaled points
    S_scaled, _, _ = calc_exact_S(pts_scaled, beta, cx, cy)
    if S_scaled == float('inf'):
         S_scaled = S

    inner = []
    angle_deg = math.degrees(beta)
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

