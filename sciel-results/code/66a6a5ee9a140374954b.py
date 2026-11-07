import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hermite_even_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 in ascending power order."""
    H0 = np.array([1.0])
    H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
    H8 = np.array([1680.0, 0.0, -13440.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0])
    H12 = np.array([10321920.0, 0.0, -5529600.0, 0.0, 8709120.0, 0.0, -4915200.0, 0.0, 1032192.0, 0.0, -98304.0, 0.0, 4096.0])
    return H0, H4, H8, H12

def _build_q(c0, c1, c2):
    """
    Build coefficients of Q(y) = g(sqrt(y))/y in ascending power order.
    g(x) = c0 H0(x) + c1 H4(x) + c2 H8(x) + c3 H12(x)
    g(x) is even, so g(x) = sum a_k x^{2k}.
    q(x) = g(x)/x^2 = sum a_k x^{2(k-1)} = sum a_k (x^2)^{k-1}.
    Let y = x^2. Q(y) = sum a_k y^{k-1}.
    """
    # Calculate c3
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 10321920.0
    
    # Coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 4915200x^6 + 8709120x^4 - 5529600x^2 + 10321920
    
    # a_k is coeff of x^{2k}
    # k=1: x^2
    a1 = c1 * (-48.0) + c2 * (-13440.0) + c3 * (-5529600.0)
    
    # k=2: x^4
    a2 = c1 * 16.0 + c2 * 11520.0 + c3 * 8709120.0
    
    # k=3: x^6
    a3 = c2 * (-3072.0) + c3 * (-4915200.0)
    
    # k=4: x^8
    a4 = c2 * 256.0 + c3 * 1032192.0
    
    # k=5: x^10
    a5 = c3 * (-98304.0)
    
    # k=6: x^12
    a6 = c3 * 4096.0
    
    # Q(y) = a1 + a2*y + a3*y^2 + a4*y^3 + a5*y^4 + a6*y^5
    return np.array([a1, a2, a3, a4, a5, a6])

def _largest_sign_change_root_y(Q_coeffs):
    """
    Find the largest positive real root y of Q(y) where Q changes sign.
    """
    if len(Q_coeffs) == 0:
        return None
        
    # Trim trailing zeros
    while len(Q_coeffs) > 1 and abs(Q_coeffs[-1]) < 1e-15:
        Q_coeffs = Q_coeffs[:-1]
        
    if len(Q_coeffs) == 1:
        return None

    roots = np.roots(Q_coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            real_roots.append(float(r.real))
            
    real_roots.sort()
    
    best_y = None
    for y in real_roots:
        if y <= 0:
            continue
            
        # Check sign change
        found_sign_change = False
        for eps in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            vm = np.polyval(Q_coeffs, y - eps)
            vp = np.polyval(Q_coeffs, y + eps)
            if vm * vp < 0:
                found_sign_change = True
                break
        
        if found_sign_change:
            if best_y is None or y > best_y:
                best_y = y
                
    return best_y

def _objective(c):
    """
    Compute the value r^2/(2*pi) for coefficient vector c.
    r^2 = y_max.
    """
    c0, c1, c2 = c
    Q_coeffs = _build_q(c0, c1, c2)
    y_max = _largest_sign_change_root_y(Q_coeffs)
    if y_max is None:
        return 1e12
    return y_max / (2.0 * np.pi)

def _local_refine(c_start, max_iter=2000, tol=1e-12):
    """Local refinement using L-BFGS-B with gradient, then Nelder-Mead polish."""
    c = np.array(c_start, dtype=float)
    best_val = _objective(c)
    
    # Try L-BFGS-B with gradient first
    try:
        result = minimize(_objective, c, method='L-BFGS-B', 
                        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        if result.fun < best_val:
            c = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Nelder-Mead polish
    try:
        result = minimize(_objective, c, method='Nelder-Mead', 
                        options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if result.fun < best_val:
            c = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Coordinate descent polish for final precision
    for _ in range(50):
        improved = False
        for i in range(3):
            for scale in [1e-6, 1e-7, 1e-8, 1e-9]:
                for direction in [-1, 1]:
                    c_new = c.copy()
                    c_new[i] += direction * scale
                    val = _objective(c_new)
                    if val < best_val - 1e-14:
                        c = c_new
                        best_val = val
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
            
    return c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # 15s safety margin before 210s limit
    
    # Seed
    seed_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed_c.copy()
    best_val = _objective(best_c)
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        bounds = [
            (-2.0, 2.0),   # c0
            (-0.5, 0.5),   # c1
            (-0.05, 0.05), # c2
        ]
        
        try:
            de_result = differential_evolution(
                _objective, 
                bounds, 
                seed=42,
                maxiter=50,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement from multiple starting points
    # Start from seed, DE result, and some random points
    start_points = [seed_c.copy(), best_c.copy()]
    
    # Add some diverse random starting points
    np.random.seed(123)
    for _ in range(10):
        c_rand = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.02, 0.02),
        ])
        start_points.append(c_rand)
    
    for pt in start_points:
        if time.time() > deadline - 30:
            break
        c_ref, v_ref = _local_refine(pt, max_iter=1000, tol=1e-12)
        if v_ref < best_val:
            best_val = v_ref
            best_c = c_ref.copy()
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        c_final, v_final = _local_refine(best_c, max_iter=5000, tol=1e-14)
        if v_final < best_val:
            best_val = v_final
            best_c = c_final.copy()
    
    return best_c.tolist()
