import numpy as np
import time
from scipy.optimize import minimize

# Grid definition
# x: -3 to 1.5 (exclusive of 1.5? np.arange(-3, 1.5, 0.01) -> 450 points)
# y: -1.5 to 1.5 (exclusive of 1.5? np.arange(-1.5, 1.5, 0.01) -> 300 points)
X, Y = np.meshgrid(np.arange(-3, 1.5, 0.01), np.arange(-1.5, 1.5, 0.01))
N_POINTS = X.size

# Pre-flatten for vectorized ops
X_flat = X.ravel()
Y_flat = Y.ravel()

_DEADLINE = 610.0  # Safety margin before 630s kill
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _base_inside(u, v):
    """Boolean: is (u,v) in the base L-shaped corridor?"""
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return horiz | vert

def _eval_path_vectorized(path):
    """
    Vectorized evaluation: returns number of sofa grid points inside EVERY placement.
    path: np.array of shape (n, 3)
    """
    # Add start and end points
    start = np.array([[0.0, 0.0, 0.0]])
    end = np.array([[-2.0, 0.0, 90.0]])
    p = np.vstack([start, path, end])
    
    # Densify path per evaluator rules
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1 = p[i, 2]
        a2 = p[i + 1, 2]
        # m = max(1, ceil(|a2 - a1|))
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        if m > 1:
            # Interpolate between p[i] and p[i+1]
            # We need m steps, so we generate points at t = 1/m, 2/m, ..., 1
            # But the evaluator adds linear interpolations. 
            # "Between consecutive rows it adds linear interpolations of (x, y, a) in m = max(1, ceil(|a2 - a1|)) equal steps"
            # This usually means we check the intermediate points too.
            # Let's generate all points in the segment including endpoints to be safe, 
            # but the start/end are already handled.
            # Actually, the evaluator says: "puts (0,0,0) before ... and (-2,0,90) after. 
            # Between consecutive rows it adds linear interpolations ... in m equal steps."
            # This implies the sequence is: Row_i, Interp_1, ..., Interp_{m-1}, Row_{i+1}
            # Wait, "in m equal steps" usually means m intervals, so m-1 intermediate points? 
            # Or does it mean m points total?
            # "adds linear interpolations ... in m equal steps"
            # If m=1, no intermediate points? Or 1 intermediate?
            # Let's look at the current program's logic:
            # for k in range(1, m + 1): t = k / m ... 
            # This generates m points. One of them is t=1.0 which is the next row.
            # So the evaluator effectively checks: Row_i, Interp(1/m), ..., Interp(m/m=Row_{i+1})
            # So we need to check all k=1..m for each segment.
            
            ts = np.linspace(0.0, 1.0, m + 1)
            # We skip t=0 because p[i] is already in dense
            for t in ts[1:]:
                row = p[i] * (1.0 - t) + p[i + 1] * t
                dense.append(row)
        else:
            dense.append(p[i + 1])
            
    # Remove duplicates if any (shouldn't be any with this logic, but safe)
    # Actually, the last point of segment i is the first of segment i+1.
    # My loop appends p[i+1] at the end of segment i.
    # Next segment starts with p[i+1]. 
    # So dense contains: p[0], seg1_interps, p[1], seg2_interps, p[2]...
    # This is correct.
    
    dense = np.array(dense)
    n_placements = len(dense)
    
    # Process in chunks to manage memory (135k points * 4 bytes * n_placements)
    # 135,000 * 4 = 540 KB per mask. 100 placements = 54 MB. Safe.
    chunk_size = 100
    inside = np.ones(X.shape, dtype=bool)
    
    for start_idx in range(0, n_placements, chunk_size):
        end_idx = min(start_idx + chunk_size, n_placements)
        chunk = dense[start_idx:end_idx]
        n_chunk = end_idx - start_idx
        
        cx = chunk[:, 0]  # (n_chunk,)
        cy = chunk[:, 1]
        a_rad = np.deg2rad(chunk[:, 2])
        c = np.cos(a_rad)  # (n_chunk,)
        s = np.sin(a_rad)  # (n_chunk,)
        
        # Rotate and translate
        # dx = X - cx, dy = Y - cy
        # u = c*dx - s*dy ? 
        # Problem says: R(-a)(w - (x, y))
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        # So u = cos(a)*dx + sin(a)*dy
        #    v = -sin(a)*dx + cos(a)*dy
        
        dx = X[None, :, :] - cx[:, None, None]
        dy = Y[None, :, :] - cy[:, None, None]
        
        u = c[:, None, None] * dx + s[:, None, None] * dy
        v = -s[:, None, None] * dx + c[:, None, None] * dy
        
        # Check base inside
        horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask = horiz | vert
        
        # Point is inside if inside ALL placements in chunk
        chunk_inside = mask.all(axis=0)
        inside &= chunk_inside
        
        # Early exit if no points remain
        if not inside.any():
            break
            
    return int(inside.sum()) * 0.0001

def _generate_path(params):
    """
    Generate a path from parametric representation.
    params: [cx, cy, R, phase, k1, k2, n_points]
    We use a spline-like interpolation or a simple parametric curve.
    Let's use a cubic Bezier or a simple polynomial in t for x and y, 
    with angle determined by the tangent or fixed linear progression.
    
    To keep it robust, let's stick to optimizing N points but with a smoother initialization 
    and a "smoothing" penalty in the objective? 
    No, the score is strictly the area.
    
    Let's use a parametric curve:
    x(t) = cx + A * cos(t * pi/2 + phase) + B * sin(t * pi)
    y(t) = cy + C * sin(t * pi/2 + phase) + D * cos(t * pi)
    a(t) = linear from 0 to 90? No, the path defines the angle.
    
    Actually, the simplest robust improvement is:
    1. Use the vectorized evaluator.
    2. Use Nelder-Mead on the 16 (x,y) variables but with a better initial guess 
       derived from a "hinge" motion.
    """
    n_points = 8
    # Fixed angles for now, as optimizing them adds 8 more dims and they are strongly coupled.
    # The inspiration optimized angles too. Let's try optimizing angles as well.
    # Total 24 params.
    
    x = params[0:8]
    y = params[8:16]
    a = params[16:24]
    
    # Clip angles to valid range
    a = np.clip(a, 1.0, 89.0)
    
    path = np.column_stack([x, y, a])
    return path

def _objective(params):
    if _elapsed() > _DEADLINE - 5:
        return 0.0
    path = _generate_path(params)
    area = _eval_path_vectorized(path)
    return -area

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    
    # Initial guesses
    # Strategy: "Hinge" motion. The sofa rotates around the inner corner.
    # Inner corner of base is (0,0).
    # Start (0,0,0). End (-2,0,90).
    # A common good path is to rotate around (0,0) then translate?
    # Or rotate around a moving point.
    
    # Guess 1: Linear interpolation of the "center of mass" of the L-shape
    # L-shape center of mass approx (0.5, 0.5) in local coords?
    # Let's try a few random starts and a few heuristic starts.
    
    best_params = None
    best_score = -np.inf
    
    # Heuristic Start 1: Arc
    # Rotate around (0,0) for first half, then translate?
    # Let's just use a smooth curve from (0,0) to (-2,0) with angle 0->90
    t = np.linspace(0, 1, 8)
    x1 = -2.0 * t
    y1 = 0.0
    a1 = 90.0 * t
    p1 = np.concatenate([x1, y1, a1])
    
    # Heuristic Start 2: Quarter Circle
    # Center at (0, -2)? Radius 2?
    # If we rotate around (0, -2), the point (0,0) moves to (-2, 0) at 90 deg.
    # Let's check: (0,0) relative to (0,-2) is (0,2). Rot 90 -> (-2, 0). Abs pos (-2, -2).
    # We want end at (-2, 0).
    # So maybe center at (0, 2)? (0,0) rel (0,2) is (0,-2). Rot 90 -> (2,0). Abs (2,2). No.
    
    # Let's try: Center of rotation moves.
    # Let's just use random restarts with the fast evaluator.
    
    np.random.seed(42)
    
    candidates = []
    
    # Candidate 1: Linear
    candidates.append(p1)
    
    # Candidate 2: S-Curve Y
    x2 = -2.0 * t
    y2 = 0.5 * np.sin(np.pi * t)
    a2 = 90.0 * t
    candidates.append(np.concatenate([x2, y2, a2]))
    
    # Candidate 3: S-Curve Y (negative)
    y3 = -0.5 * np.sin(np.pi * t)
    candidates.append(np.concatenate([x2, y3, a2]))
    
    # Candidate 4: Random
    for _ in range(5):
        x_r = np.linspace(0, -2, 8) + np.random.normal(0, 0.2, 8)
        y_r = np.random.normal(0, 0.5, 8)
        a_r = np.linspace(10, 80, 8) + np.random.normal(0, 5, 8)
        candidates.append(np.concatenate([x_r, y_r, a_r]))
    
    # Evaluate candidates
    for cand in candidates:
        if _elapsed() > _DEADLINE - 50:
            break
        score = -_objective(cand)
        if score > best_score:
            best_score = score
            best_params = cand.copy()
            
    # Refine the best candidate with Nelder-Mead
    if best_params is not None and _elapsed() < _DEADLINE - 100:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
        
        # Second refinement from a slightly perturbed best
        if _elapsed() < _DEADLINE - 50:
            try:
                perturbed = best_params + np.random.normal(0, 0.1, len(best_params))
                res2 = minimize(_objective, perturbed, method='Nelder-Mead', 
                                options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
                if res2.fun < -best_score:
                    best_score = -res2.fun
                    best_params = res2.x
            except Exception:
                pass
                
    # Final check and return
    path = _generate_path(best_params)
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(path)):
        # Fallback to linear
        path = np.column_stack([np.linspace(0, -2, 8), np.zeros(8), np.linspace(10, 80, 8)])
        
    return path

# EVOLVE-BLOCK-END
