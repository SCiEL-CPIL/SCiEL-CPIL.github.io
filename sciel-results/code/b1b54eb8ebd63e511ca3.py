import numpy as np
import numpy.polynomial.laguerre as npl
import time
from scipy.optimize import minimize
from scipy.special import factorial

def construct():
    start_time = time.time()
    deadline = 600.0
    
    # Initial guess: Roots of L_12
    try:
        roots_init, _ = npl.laggauss(12)
        roots_init = np.sort(roots_init).astype(float)
    except Exception:
        roots_init = np.array([4.0 * (i + 1) for i in range(12)])

    # Proxy Objective: Minimize Variance of P(x)^2 under Laguerre weight
    # P(x) = prod (x - r_i)^2
    # P(x)^2 = prod (x - r_i)^4
    
    def compute_variance(roots):
        # Compute coefficients of P(x)^2
        # P(x) = prod (x^2 - 2 r_i x + r_i^2)
        # It's easier to compute P(x) first, then square it?
        # Or just compute the product of (x - r_i)^4 directly?
        # Let's use numpy.poly1d for simplicity, though it's not the most efficient.
        # For 12 roots, degree 48.
        
        # Efficient way:
        # Start with poly [1]
        # For each r, multiply by (x - r)^4 = x^4 - 4r x^3 + 6r^2 x^2 - 4r^3 x + r^4
        
        coeffs = np.array([1.0])
        for r in roots:
            # (x - r)^4 coefficients in descending order? 
            # numpy.poly1d uses descending.
            # Let's stick to numpy.poly1d for clarity, check speed.
            pass
            
        # Actually, let's use the moment property directly if we can.
        # But we need coefficients.
        
        # Let's use numpy.poly1d
        p = np.poly1d([1.0])
        for r in roots:
            # (x - r)^4
            q = np.poly1d([1.0, -4.0*r, 6.0*r*r, -4.0*r**3, r**4])
            p = p * q
            
        # p is P(x)^2
        # Coefficients in descending order: c_48 x^48 + ... + c_0
        # M_k = sum_{j=k}^{48} c_j * (j!) * (-1)^{j-k}? No.
        # Integral of x^j e^{-x} is j!.
        # So M_k = sum_{j=0}^{48} coeff[j] * j! where coeff[j] is coeff of x^j.
        # numpy.poly1d gives coeffs in descending order.
        # So coeff of x^j is p.c[48-j].
        
        # We need M_0, M_1, M_2
        # M_0 = sum c_j j!
        # M_1 = sum j c_j j!
        # M_2 = sum j^2 c_j j!
        
        # Note: p.c might be very small/large.
        # Let's compute.
        
        # Optimization: Precompute factorials?
        # j! for j up to 48 is large. 48! ~ 10^60.
        # This might overflow float64.
        # We can scale.
        # Or use log-sum-exp?
        # Or just normalize the polynomial?
        # The variance is scale-invariant if we scale P? No, P is fixed by roots.
        # But we can divide all moments by M_0.
        # Let's compute using logs or just careful scaling.
        
        # Alternative: Use Gauss-Laguerre quadrature to compute the integrals?
        # We have 12 roots. P(x)^2 is degree 48.
        # Gauss-Laguerre with N=50 nodes would be exact for polynomials up to degree 2*50-1 = 99.
        # This is much safer and faster to code.
        
        # Let's use npl.laggauss(50)
        # nodes, weights = npl.laggauss(50)
        # P_sq = prod (nodes - r_i)^4
        # M_0 = sum weights * P_sq
        # M_1 = sum weights * nodes * P_sq
        # M_2 = sum weights * nodes**2 * P_sq
        
        # This is O(50 * 12) per evaluation. Very fast.
        return None # Placeholder

    # Let's implement the quadrature-based objective
    
    # Precompute quadrature nodes and weights (N=50 is sufficient for degree 48)
    try:
        nodes, weights = npl.laggauss(50)
    except Exception:
        return roots_init.tolist()

    def objective(roots):
        # Compute P(x)^2 at nodes
        # P(x) = prod (x - r_i)^2
        # P(x)^2 = prod (x - r_i)^4
        
        # Vectorized computation
        # roots shape (12,), nodes shape (50,)
        # diff shape (50, 12)
        diff = nodes[:, np.newaxis] - roots[np.newaxis, :]
        p_sq = np.prod(diff**4, axis=1)
        
        # Moments
        m0 = np.dot(weights, p_sq)
        m1 = np.dot(weights, nodes * p_sq)
        m2 = np.dot(weights, nodes**2 * p_sq)
        
        # Variance
        if m0 <= 0:
            return 1e10
        mean = m1 / m0
        var = m2 / m0 - mean**2
        
        return var

    # Optimize
    # Bounds: roots must be positive and distinct.
    # We can optimize over log(roots) to ensure positivity.
    # And use a penalty for closeness? Or just rely on the landscape.
    # Let's optimize over log(roots).
    
    log_roots_init = np.log(roots_init)
    
    def obj_log(x):
        roots = np.exp(x)
        # Check for duplicates? 
        # If roots are too close, variance might be ill-defined or just large.
        # The objective function itself is smooth.
        return objective(roots)

    # Use L-BFGS-B
    try:
        res = minimize(obj_log, log_roots_init, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-10})
        
        if res.success:
            best_roots = np.exp(res.x)
            # Sort and ensure positive
            best_roots = np.sort(best_roots)
            
            # Verify against deadline
            if time.time() > deadline - 60:
                return roots_init.tolist()
                
            return best_roots.tolist()
        else:
            return roots_init.tolist()
    except Exception:
        return roots_init.tolist()
