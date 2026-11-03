import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple hexagonal lattice
        pts = _fallback_lattice()
        cx, cy = 0.0, 0.0
        beta = 0.0
        S = _calc_S(pts, beta, cx, cy)
        angle_deg = 0.0
        inner = [[float(x), float(y), angle_deg] for x, y in pts]
        center = [float(cx), float(cy)]
        side = float(S)
        rotation = float(angle_deg)
        return inner, center, side, rotation

    def _fallback_lattice():
        # Standard hexagonal packing of 11 hexagons
        pts = []
        # Center
        pts.append((0, 0))
        # Ring 1: 6 hexagons
        for k in range(6):
            ang = k * math.pi / 3
            pts.append((math.sqrt(3) * math.cos(ang), math.sqrt(3) * math.sin(ang)))
        # Ring 2: 4 more to make 11 (partial ring)
        for k in [0, 1, 4, 5]:
            ang = k * math.pi / 3
            r = 2 * math.sqrt(3)
            pts.append((r * math.cos(ang), r * math.sin(ang)))
        return np.array(pts)

    def _calc_S(pts, beta, cx, cy):
        """Calculate the side length S of the smallest enclosing regular hexagon."""
        if len(pts) == 0:
            return 0.0
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
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
                cx_opt = res.x[0]
                cy_opt = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx_opt, cy_opt), D
            else:
                return float('inf'), (cx, cy), float('inf')
        except Exception:
            return float('inf'), (cx, cy), float('inf')

    def _calc_S_fast(pts, beta, cx, cy):
        """Fast calculation of S assuming center is given (for optimization loops)."""
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def _overlap_penalty(pts, eps=1e-4):
        """Calculate overlap penalty between all pairs of hexagons."""
        normals = np.array([
            [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        n_pts = len(pts)
        penalty = 0.0
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                min_proj = np.min(projs)
                if min_proj < math.sqrt(3) - eps:
                    violation = (math.sqrt(3) - eps) - min_proj
                    penalty += violation * 100.0
        return penalty

    def _optimize_from_start(pts_init, beta_init=0.0):
        """Optimize positions, center, and rotation from a given starting point."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None, None, None
        
        pts = pts_init.copy()
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        beta = beta_init
        
        # Step 1: Optimize beta for fixed positions
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            def obj_beta(b):
                b = ((b % (math.pi/6)) + math.pi/6) % (math.pi/6)
                # Find optimal center for this beta
                S_res, center, D = _calc_S(pts, b, cx, cy)
                return S_res
            
            try:
                res_b = minimize_scalar(obj_beta, bounds=(0, math.pi/6), method='bounded',
                                       options={'maxiter': 100, 'xatol': 1e-10})
                if res_b.fun < float('inf'):
                    beta = res_b.x
            except Exception:
                pass
        
        # Step 2: Joint optimization of positions, center, and beta
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            def obj(params):
                n = len(pts)
                pts_curr = params[:2*n].reshape(n, 2)
                cx_c = params[2*n]
                cy_c = params[2*n+1]
                beta_c = ((params[2*n+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                
                # Smooth S approximation
                S_smooth = _calc_S_fast(pts_curr, beta_c, cx_c, cy_c)
                
                # Overlap penalty
                penalty = _overlap_penalty(pts_curr)
                
                return S_smooth + penalty
            
            init_params = np.concatenate([pts.ravel(), [cx, cy, beta]])
            
            try:
                res = minimize(obj, init_params, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9,
                                        'adaptive': True})
                if res.fun < float('inf'):
                    pts = res.x[:2*N].reshape(N, 2)
                    cx = res.x[2*N]
                    cy = res.x[2*N+1]
                    beta = ((res.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
            except Exception:
                pass
            
            # Step 3: Final refinement with tighter settings
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                try:
                    res2 = minimize(obj, init_params, method='Powell',
                                   options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res2.fun < _calc_S_fast(pts, beta, cx, cy) + _overlap_penalty(pts):
                        pts = res2.x[:2*N].reshape(N, 2)
                        cx = res2.x[2*N]
                        cy = res2.x[2*N+1]
                        beta = ((res2.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                except Exception:
                    pass
        
        # Final exact calculation
        S, center, D = _calc_S(pts, beta, cx, cy)
        if S < float('inf'):
            return S, pts, beta, center[0], center[1]
        return float('inf'), pts, beta, cx, cy

    def _generate_starts():
        """Generate diverse starting configurations."""
        starts = []
        
        # Start 1: Standard hexagonal lattice (1+6+4)
        pts = np.array([
            [0, 0],
            [math.sqrt(3), 0],
            [math.sqrt(3)/2, 1.5],
            [-math.sqrt(3)/2, 1.5],
            [-math.sqrt(3), 0],
            [-math.sqrt(3)/2, -1.5],
            [math.sqrt(3)/2, -1.5],
            [2*math.sqrt(3), 0],
            [math.sqrt(3), 1.5*1.732],  # Approx
            [0, 1.5*2],
            [-math.sqrt(3), 1.5*1.732]
        ])
        starts.append(pts)
        
        # Start 2: Linear chain
        pts = np.array([[i * math.sqrt(3), 0] for i in range(N)])
        starts.append(pts)
        
        # Start 3: Triangular arrangement
        pts = []
        for i in range(3):
            for j in range(4):
                pts.append([j * math.sqrt(3) + i * math.sqrt(3)/2, i * 1.5])
        if len(pts) > N:
            pts = pts[:N]
        starts.append(np.array(pts))
        
        # Start 4: Random perturbed lattice
        rng = np.random.default_rng(42)
        pts = np.array([
            [0, 0],
            [math.sqrt(3), 0],
            [math.sqrt(3)/2, 1.5],
            [-math.sqrt(3)/2, 1.5],
            [-math.sqrt(3), 0],
            [-math.sqrt(3)/2, -1.5],
            [math.sqrt(3)/2, -1.5],
            [2*math.sqrt(3), 0],
            [math.sqrt(3), 3.0],
            [0, 3.0],
            [-math.sqrt(3), 3.0]
        ])
        pts += rng.normal(0, 0.1, pts.shape)
        starts.append(pts)
        
        # Start 5: Compact cluster
        pts = np.array([
            [0, 0],
            [1.732, 0],
            [0.866, 1.5],
            [-0.866, 1.5],
            [-1.732, 0],
            [-0.866, -1.5],
            [0.866, -1.5],
            [2.6, 0.5],
            [1.3, 2.5],
            [-1.3, 2.5],
            [-2.6, 0.5]
        ])
        starts.append(pts)
        
        return starts

    def _global_search():
        """Use differential evolution for global optimization."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        def obj(params):
            n = N
            pts = params[:2*n].reshape(n, 2)
            cx = params[2*n]
            cy = params[2*n+1]
            beta = ((params[2*n+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
            
            S = _calc_S_fast(pts, beta, cx, cy)
            penalty = _overlap_penalty(pts)
            
            return S + penalty
        
        # Bounds: positions in [-5, 5], center in [-2, 2], beta in [0, pi/6]
        bounds = []
        for _ in range(2*N):
            bounds.append((-5.0, 5.0))
        bounds.append((-2.0, 2.0))
        bounds.append((-2.0, 2.0))
        bounds.append((0.0, math.pi/6))
        
        try:
            # Use a limited number of evaluations to stay within time budget
            remaining_time = TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)
            max_iter = max(10, int(remaining_time / 0.5))  # Rough estimate
            
            res = differential_evolution(obj, bounds, maxiter=max_iter, tol=1e-8,
                                         polish=True, seed=42, workers=1)
            
            if res.success and res.fun < float('inf'):
                pts = res.x[:2*N].reshape(N, 2)
                cx = res.x[2*N]
                cy = res.x[2*N+1]
                beta = ((res.x[2*N+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                return pts, cx, cy, beta
        except Exception:
            pass
        
        return None

    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Try multiple starting points with local optimization
    starts = _generate_starts()
    for i, pts_init in enumerate(starts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        S, pts, beta, cx, cy = _optimize_from_start(pts_init)
        if S < best_S:
            best_S = S
            best_pts = pts
            best_beta = beta
            best_cx = cx
            best_cy = cy

    # Try global search if time permits
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 30:
        result = _global_search()
        if result is not None:
            pts, cx, cy, beta = result
            S, center, D = _calc_S(pts, beta, cx, cy)
            if S < best_S:
                best_S = S
                best_pts = pts
                best_beta = beta
                best_cx = center[0]
                best_cy = center[1]

    # Final verification and scaling
    if best_pts is None or best_S == float('inf'):
        best_pts = _fallback_lattice()
        best_cx = 0.0
        best_cy = 0.0
        best_beta = 0.0
        best_S, _, _ = _calc_S(best_pts, best_beta, best_cx, best_cy)

    # Ensure no overlap with tiny scale
    overlap = _overlap_penalty(best_pts)
    if overlap > 1e-6:
        center = np.mean(best_pts, axis=0)
        scale = 1.0 - overlap * 0.001
        best_pts = center + (best_pts - center) * scale
        best_S, _, _ = _calc_S(best_pts, best_beta, best_cx, best_cy)

    angle_deg = math.degrees(best_beta)
    inner = [[float(x), float(y), angle_deg] for x, y in best_pts]
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
