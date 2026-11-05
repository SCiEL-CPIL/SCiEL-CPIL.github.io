"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import brentq
import time
import sys

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Uses dynamic quadratic interpolation with Illinois safeguarding.
    """
    x0 = float(x0)
    deadline = time.time() + 115  # 5s safety margin before 120s kill
    
    # State
    eval_count = 0
    max_evals = 500
    
    def f_counted(x):
        nonlocal eval_count
        if time.time() > deadline:
            raise TimeoutError
        eval_count += 1
        return f(x)
    
    best_x = x0
    fx0 = f_counted(x0)
    best_f = abs(fx0)
    if best_f <= 1e-8:
        return best_x
    
    # Quadratic interpolation solver with Illinois safeguarding
    def quad_illinois_solve(a, b, fa, fb):
        """
        Solve for root in [a,b] given f(a)=fa, f(b)=fb with sign change.
        Uses quadratic interpolation when 3 points available, with Illinois safeguard.
        """
        nonlocal best_x, best_f
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb > 0:
            return None
        
        try:
            lo, hi = a, b
            flo, fhi = fa, fb
            
            # Track the three most recent distinct points for quadratic interp
            # Start with the two bracket endpoints
            pts = [(lo, flo), (hi, fhi)]
            
            for _ in range(120):
                if time.time() > deadline:
                    break
                if flo == 0:
                    return lo
                if fhi == 0:
                    return hi
                if flo * fhi > 0:
                    break
                
                # Compute next candidate
                xs = None
                
                # Try quadratic interpolation if we have 3 distinct points
                if len(pts) >= 3:
                    # Use the 3 most recent points
                    (x1, y1), (x2, y2), (x3, y3) = pts[-3:]
                    # Check for distinct x values
                    if abs(x1 - x2) > 1e-15 and abs(x2 - x3) > 1e-15 and abs(x1 - x3) > 1e-15:
                        try:
                            # Lagrange interpolation for quadratic y = a*x^2 + b*x + c
                            A = np.array([[x1**2, x1, 1],
                                          [x2**2, x2, 1],
                                          [x3**2, x3, 1]], dtype=float)
                            b_vec = np.array([y1, y2, y3], dtype=float)
                            coeffs = np.linalg.solve(A, b_vec)
                            qa, qb, qc = coeffs
                            
                            # Find roots of quadratic
                            if abs(qa) < 1e-300:
                                if abs(qb) > 1e-300:
                                    xr = -qc / qb
                                    if lo < xr < hi:
                                        xs = xr
                            else:
                                disc = qb*qb - 4*qa*qc
                                if disc >= 0:
                                    sqrt_disc = np.sqrt(disc)
                                    r1 = (-qb + sqrt_disc) / (2*qa)
                                    r2 = (-qb - sqrt_disc) / (2*qa)
                                    # Choose root in bracket
                                    if lo < r1 < hi:
                                        xs = r1
                                    elif lo < r2 < hi:
                                        xs = r2
                        except (np.linalg.LinAlgError, ZeroDivisionError, OverflowError):
                            pass
                
                # Fallback: regula falsi
                if xs is None:
                    if abs(flo - fhi) > 1e-300:
                        xs = hi - fhi * (hi - lo) / (fhi - flo)
                    else:
                        xs = (lo + hi) / 2.0
                    if not (lo < xs < hi):
                        xs = (lo + hi) / 2.0
                
                fx = f_counted(xs)
                if abs(fx) < best_f:
                    best_x = xs
                    best_f = abs(fx)
                if fx == 0:
                    return xs
                
                # Update bracket
                if flo * fx < 0:
                    pts.append((hi, fhi))
                    hi = xs
                    fhi = fx
                else:
                    pts.append((lo, flo))
                    lo = xs
                    flo = fx
                
                # Keep only last 3 points
                if len(pts) > 3:
                    pts = pts[-3:]
                
                # Illinois safeguard: shrink the endpoint with larger |f|
                if abs(flo) > abs(fhi):
                    lo_new = (lo + hi) / 2.0
                    flo_new = f_counted(lo_new)
                    if abs(flo_new) < best_f:
                        best_x = lo_new
                        best_f = abs(flo_new)
                    pts.append((hi, fhi))
                    lo = lo_new
                    flo = flo_new
                    if len(pts) > 3:
                        pts = pts[-3:]
                else:
                    hi_new = (lo + hi) / 2.0
                    fhi_new = f_counted(hi_new)
                    if abs(fhi_new) < best_f:
                        best_x = hi_new
                        best_f = abs(fhi_new)
                    pts.append((lo, flo))
                    hi = hi_new
                    fhi = fhi_new
                    if len(pts) > 3:
                        pts = pts[-3:]
                
                if abs(hi - lo) < 1e-14 * max(1.0, abs(lo), abs(hi)):
                    break
            
            # Return the better endpoint
            if abs(flo) < abs(fhi):
                return lo
            else:
                return hi
        except Exception:
            return None
    
    # Collect points for model fitting
    points = [(x0, fx0)]
    
    # Phase 1: Local expansion from x0 to find a bracket
    scales = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    for s in scales:
        if time.time() > deadline or eval_count >= max_evals - 10:
            break
        a = x0 - s
        b = x0 + s
        if eval_count >= max_evals - 5:
            break
        fa = f_counted(a)
        fb = f_counted(b)
        points.append((a, fa))
        points.append((b, fb))
        if abs(fa) < best_f:
            best_x = a
            best_f = abs(fa)
        if abs(fb) < best_f:
            best_x = b
            best_f = abs(fb)
        if best_f <= 1e-8:
            return best_x
        # Check for sign change with x0 or between a and b
        if fa * fx0 < 0:
            root = quad_illinois_solve(a, x0, fa, fx0)
            if root is not None:
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    return root
                if abs(fr) < best_f:
                    best_x = root
                    best_f = abs(fr)
            break
        if fb * fx0 < 0:
            root = quad_illinois_solve(x0, b, fx0, fb)
            if root is not None:
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    return root
                if abs(fr) < best_f:
                    best_x = root
                    best_f = abs(fr)
            break
        if fa * fb < 0:
            root = quad_illinois_solve(a, b, fa, fb)
            if root is not None:
                fr = f_counted(root)
                if abs(fr) <= 1e-8:
                    return root
                if abs(fr) < best_f:
                    best_x = root
                    best_f = abs(fr)
            break
    
    # Phase 2: Model-guided local search using quadratic prediction
    for iteration in range(40):
        if time.time() > deadline or eval_count >= max_evals - 10:
            break
        
        # Get model prediction from last 3 points
        x_pred = None
        if len(points) >= 3:
            (x1, y1), (x2, y2), (x3, y3) = points[-3:]
            if abs(x1 - x2) > 1e-15 and abs(x2 - x3) > 1e-15 and abs(x1 - x3) > 1e-15:
                try:
                    A = np.array([[x1**2, x1, 1],
                                  [x2**2, x2, 1],
                                  [x3**2, x3, 1]], dtype=float)
                    b_vec = np.array([y1, y2, y3], dtype=float)
                    coeffs = np.linalg.solve(A, b_vec)
                    qa, qb, qc = coeffs
                    if abs(qa) < 1e-300:
                        if abs(qb) > 1e-300:
                            xr = -qc / qb
                            if abs(xr - best_x) < 100 * max(1.0, abs(best_x)):
                                x_pred = xr
                    else:
                        disc = qb*qb - 4*qa*qc
                        if disc >= 0:
                            sqrt_disc = np.sqrt(disc)
                            r1 = (-qb + sqrt_disc) / (2*qa)
                            r2 = (-qb - sqrt_disc) / (2*qa)
                            # Choose root closer to best_x
                            if abs(r1 - best_x) < abs(r2 - best_x):
                                if abs(r1 - best_x) < 100 * max(1.0, abs(best_x)):
                                    x_pred = r1
                            else:
                                if abs(r2 - best_x) < 100 * max(1.0, abs(best_x)):
                                    x_pred = r2
                except (np.linalg.LinAlgError, ZeroDivisionError, OverflowError):
                    pass
        
        if x_pred is None:
            # Fall back to local perturbation
            xbase = best_x
            x_pred = xbase + 0.1 * max(1.0, abs(xbase)) * np.random.uniform(-1, 1)
        
        # Evaluate at predicted point
        if eval_count >= max_evals - 5:
            break
        fx = f_counted(x_pred)
        if abs(fx) < best_f:
            best_x = x_pred
            best_f = abs(fx)
        if best_f <= 1e-8:
            return best_x
        points.append((x_pred, fx))
        
        # Check if we can form a bracket with any pair of recent points
        for i in range(len(points) - 1, 0, -1):
            xi, fi = points[i]
            xj, fj = points[i-1]
            if fi * fj < 0:
                root = quad_illinois_solve(xj, xi, fj, fi)
                if root is not None:
                    fr = f_counted(root)
                    if abs(fr) <= 1e-8:
                        return root
                    if abs(fr) < best_f:
                        best_x = root
                        best_f = abs(fr)
                break
        
        # Keep points list manageable
        if len(points) > 20:
            points = points[-10:]
    
    # Phase 3: Broader multi-scale exploration
    np.random.seed(42)
    for radius in [0.5, 1, 2, 5, 10, 50, 100, 500]:
        if time.time() > deadline or eval_count >= max_evals - 20:
            break
        for _ in range(3):
            if time.time() > deadline or eval_count >= max_evals - 10:
                break
            x_try = x0 + radius * np.random.uniform(-1, 1)
            a = x_try - radius * 0.1
            b = x_try + radius * 0.1
            if eval_count >= max_evals - 5:
                break
            fa = f_counted(a)
            fb = f_counted(b)
            points.append((a, fa))
            points.append((b, fb))
            if abs(fa) < best_f:
                best_x = a
                best_f = abs(fa)
            if abs(fb) < best_f:
                best_x = b
                best_f = abs(fb)
            if best_f <= 1e-8:
                return best_x
            # Check brackets
            if fa * fb < 0:
                root = quad_illinois_solve(a, b, fa, fb)
                if root is not None:
                    fr = f_counted(root)
                    if abs(fr) <= 1e-8:
                        return root
                    if abs(fr) < best_f:
                        best_x = root
                        best_f = abs(fr)
                break
            # Also check against best_x
            fbx = f_counted(best_x)
            if fa * fbx < 0:
                root = quad_illinois_solve(a, best_x, fa, fbx)
                if root is not None:
                    fr = f_counted(root)
                    if abs(fr) <= 1e-8:
                        return root
                    if abs(fr) < best_f:
                        best_x = root
                        best_f = abs(fr)
                break
    
    # Phase 4: Final refinement
    for iteration in range(20):
        if time.time() > deadline or eval_count >= max_evals - 5:
            break
        x_pred = None
        if len(points) >= 3:
            (x1, y1), (x2, y2), (x3, y3) = points[-3:]
            if abs(x1 - x2) > 1e-15 and abs(x2 - x3) > 1e-15 and abs(x1 - x3) > 1e-15:
                try:
                    A = np.array([[x1**2, x1, 1],
                                  [x2**2, x2, 1],
                                  [x3**2, x3, 1]], dtype=float)
                    b_vec = np.array([y1, y2, y3], dtype=float)
                    coeffs = np.linalg.solve(A, b_vec)
                    qa, qb, qc = coeffs
                    if abs(qa) < 1e-300:
                        if abs(qb) > 1e-300:
                            xr = -qc / qb
                            x_pred = xr
                    else:
                        disc = qb*qb - 4*qa*qc
                        if disc >= 0:
                            sqrt_disc = np.sqrt(disc)
                            r1 = (-qb + sqrt_disc) / (2*qa)
                            r2 = (-qb - sqrt_disc) / (2*qa)
                            x_pred = r1 if abs(r1 - points[-1][0]) < abs(r2 - points[-1][0]) else r2
                except (np.linalg.LinAlgError, ZeroDivisionError, OverflowError):
                    pass
        
        if x_pred is None:
            break
        if abs(x_pred - points[-1][0]) < 1e-15:
            break
        fx = f_counted(x_pred)
        if abs(fx) < best_f:
            best_x = x_pred
            best_f = abs(fx)
        if best_f <= 1e-8:
            return best_x
        points.append((x_pred, fx))
        if len(points) > 20:
            points = points[-10:]
    
    # Final check: try solver on any sign-change pair in points
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if points[i][1] * points[j][1] < 0:
                a, b = min(points[i][0], points[j][0]), max(points[i][0], points[j][0])
                fa, fb = points[i][1] if points[i][0] == a else points[j][1]
                fb = points[j][1] if points[j][0] == b else fa
                root = quad_illinois_solve(a, b, fa, fb)
                if root is not None:
                    fr = f_counted(root)
                    if abs(fr) <= 1e-8:
                        return root
                    if abs(fr) < best_f:
                        best_x = root
                        best_f = abs(fr)
    
    return best_x
# EVOLVE-BLOCK-END
