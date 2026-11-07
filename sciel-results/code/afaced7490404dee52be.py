import numpy as np
import time
from scipy.optimize import minimize, approx_fprime

def _hermite_even_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 in ascending power order."""
    H0 = np.array([1.0])
    H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
    H8 = np.array([1680.0, 0.0, -13440.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0])
    H12 = np.array([10321920.0, 0.0, -5529600.0, 0.0, 8709120.0, 0.0, -4915200.0, 0.0, 1032192.0, 0.0, -98304.0, 0.0, 4096.0])
    return H0, H4, H8, H12

def _build_Q_coeffs(c0, c1, c2):
    """
    Build coefficients of Q(y) = g(sqrt(y))/y in ascending power order.
    g(x) = c0 H0(x) + c1 H4(x) + c2 H8(x) + c3 H12(x)
    g(x) is even, so g(x) = sum a_k x^{2k}.
    q(x) = g(x)/x^2 = sum a_k x^{2(k-1)} = sum a_k (x^2)^{k-1}.
    Let y = x^2. Q(y) = sum a_k y^{k-1}.
    """
    H0, H4, H8, H12 = _hermite_even_coeffs()
    
    # Calculate c3
    # g(0) = c0*1 + c1*12 + c2*1680 + c3*10321920 = 0
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 10321920.0
    
    # g coefficients (even powers only, index i corresponds to x^i)
    # H0: x^0
    # H4: x^0, x^2, x^4
    # H8: x^0, x^2, x^4, x^6, x^8
    # H12: x^0, x^2, x^4, x^6, x^8, x^10, x^12
    
    # We need a_k for x^{2k}
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
    # H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 4915200x^6 + 8709120x^4 - 5529600x^2 + 10321920
    
    # Coefficients for x^0, x^2, x^4, x^6, x^8, x^10, x^12
    # Note: The arrays in _hermite_even_coeffs are in ascending power order of x.
    # H0: [1] -> x^0: 1
    # H4: [12, 0, -48, 0, 16] -> x^0: 12, x^2: -48, x^4: 16
    # H8: [1680, 0, -13440, 0, 11520, 0, -3072, 0, 256] -> x^0: 1680, x^2: -13440, x^4: 11520, x^6: -3072, x^8: 256
    # H12: [10321920, 0, -5529600, 0, 8709120, 0, -4915200, 0, 1032192, 0, -98304, 0, 4096]
    #      x^0: 10321920, x^2: -5529600, x^4: 8709120, x^6: -4915200, x^8: 1032192, x^10: -98304, x^12: 4096
    
    # a_k is coeff of x^{2k}
    # k=0: x^0
    a0 = c0 * 1.0 + c1 * 12.0 + c2 * 1680.0 + c3 * 10321920.0
    # By definition of c3, a0 should be 0.
    
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
    # Note: a0 is 0, so the constant term of Q(y) is a1.
    # Wait, q(x) = a1 + a2 x^2 + ...
    # Q(y) = a1 + a2 y + a3 y^2 + a4 y^3 + a5 y^4 + a6 y^5
    
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
        # Constant polynomial
        if abs(Q_coeffs[0]) < 1e-15:
            # Q(y) = 0 for all y?
            # This implies g(x) is divisible by x^2 for all x?
            # If Q is identically zero, q is identically zero.
            # Roots are everywhere?
            # This is a degenerate case.
            return None
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
        # Evaluate Q(y-eps) and Q(y+eps)
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
    Q_coeffs = _build_Q_coeffs(c0, c1, c2)
    y_max = _largest_sign_change_root_y(Q_coeffs)
    if y_max is None:
        # If no valid root, return a large penalty
        return 1e12
    return y_max / (2.0 * np.pi)

def _objective_grad(c, eps=1e-7):
    """Compute gradient of objective using finite differences."""
    return approx_fprime(c, _objective, eps)

def _local_refine(c_start, max_iter=2000, tol=1e-12):
    """Local refinement using L-BFGS-B with gradient, then Nelder-Mead polish."""
    c = np.array(c_start, dtype=float)
    best_val = _objective(c)
    
    # Try L-BFGS-B with gradient first
    try:
        result = minimize(_objective, c, jac=_objective_grad, method='L-BFGS-B', 
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
    
    # Phase 1: Multi-scale grid exploration around seed
    base = seed_c.copy()
    
    # Coarse grid
    ranges_coarse = [
        (base[0] - 1.0, base[0] + 1.0, 11),
        (base[1] - 0.1, base[1] + 0.1, 11),
        (base[2] - 0.005, base[2] + 0.005, 11),
    ]
    
    grid_results = []
    for d0 in np.linspace(ranges_coarse[0][0], ranges_coarse[0][1], ranges_coarse[0][2]):
        for d1 in np.linspace(ranges_coarse[1][0], ranges_coarse[1][1], ranges_coarse[1][2]):
            for d2 in np.linspace(ranges_coarse[2][0], ranges_coarse[2][1], ranges_coarse[2][2]):
                if time.time() > deadline - 30:
                    break
                pt = np.array([d0, d1, d2])
                val = _objective(pt)
                grid_results.append((val, pt))
    
    grid_results.sort(key=lambda x: x[0])
    
    # Refine top candidates from coarse grid
    top_k = min(20, len(grid_results))
    for val, pt in grid_results[:top_k]:
        if time.time() > deadline - 20:
            break
        c_ref, v_ref = _local_refine(pt, max_iter=500, tol=1e-10)
        if v_ref < best_val:
            best_val = v_ref
            best_c = c_ref.copy()
    
    # Phase 2: Fine grid around best found
    if time.time() < deadline - 30:
        fine_ranges = [
            (best_c[0] - 0.05, best_c[0] + 0.05, 7),
            (best_c[1] - 0.005, best_c[1] + 0.005, 7),
            (best_c[2] - 0.0002, best_c[2] + 0.0002, 7),
        ]
        
        fine_results = []
        for d0 in np.linspace(fine_ranges[0][0], fine_ranges[0][1], fine_ranges[0][2]):
            for d1 in np.linspace(fine_ranges[1][0], fine_ranges[1][1], fine_ranges[1][2]):
                for d2 in np.linspace(fine_ranges[2][0], fine_ranges[2][1], fine_ranges[2][2]):
                    if time.time() > deadline - 15:
                        break
                    pt = np.array([d0, d1, d2])
                    val = _objective(pt)
                    fine_results.append((val, pt))
        
        fine_results.sort(key=lambda x: x[0])
        
        for val, pt in fine_results[:10]:
            if time.time() > deadline - 10:
                break
            c_ref, v_ref = _local_refine(pt, max_iter=1000, tol=1e-11)
            if v_ref < best_val:
                best_val = v_ref
                best_c = c_ref.copy()
    
    # Phase 3: Random restarts with L-BFGS-B
    np.random.seed(42)
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline - 5:
            break
        # Generate diverse random points
        scale0 = np.random.choice([0.1, 0.5, 1.0])
        scale1 = np.random.choice([0.01, 0.05, 0.1])
        scale2 = np.random.choice([0.001, 0.005, 0.01])
        
        c_rand = np.array([
            np.random.uniform(-scale0, scale0),
            np.random.uniform(-scale1, scale1),
            np.random.uniform(-scale2, scale2),
        ])
        
        c_ref, v_ref = _local_refine(c_rand, max_iter=300, tol=1e-10)
        if v_ref < best_val:
            best_val = v_ref
            best_c = c_ref.copy()
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 3:
        c_final, v_final = _local_refine(best_c, max_iter=5000, tol=1e-14)
        if v_final < best_val:
            best_val = v_final
            best_c = c_final.copy()
    
    return best_c.tolist()
