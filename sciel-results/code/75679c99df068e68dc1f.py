"""Symbolic regression with 1 input(s)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill

    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    
    # Precompute basis functions
    # Clamped cubic B-spline basis for better boundary behavior
    # Knots: [-3, -2, -1, 0, 1, 2, 3] with clamped ends
    # This gives 6 cubic B-spline basis functions
    
    def clamp_cubic_bspline(x, knots):
        """Compute clamped cubic B-spline basis functions."""
        n = len(x)
        basis = np.zeros((n, len(knots) - 3))
        
        for i in range(len(knots) - 3):
            k1 = knots[i]
            k2 = knots[i + 1]
            k3 = knots[i + 2]
            k4 = knots[i + 3]
            
            # Cubic B-spline support
            mask = (x >= k1) & (x < k4)
            if not np.any(mask):
                continue
            
            xm = x[mask]
            h = k4 - k1
            
            # Piecewise cubic
            t1 = (xm - k1) / h
            t2 = (xm - k2) / h
            t3 = (xm - k3) / h
            
            # B-spline coefficients
            term1 = np.where(t1 >= 0, t1**3, 0.0)
            term2 = np.where(t2 >= 0, 4.0 * t2**3, 0.0)
            term3 = np.where(t3 >= 0, 6.0 * t3**2 * (t3 - 1.0), 0.0)
            term4 = np.where((t3 - 1.0) >= 0, 4.0 * (t3 - 1.0)**3, 0.0)
            
            basis[mask, i] = (term1 - term2 + term3 - term4) / 6.0
        
        return basis
    
    # Use knots that span [-3, 3] with clamped ends
    knots = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    bspline_basis = clamp_cubic_bspline(x, knots)
    # This gives 6 basis functions (len(knots) - 3 = 6)
    
    # Add polynomial terms for global trend
    b_poly0 = np.ones(n)
    b_poly1 = x
    b_poly2 = x**2
    b_poly3 = x**3
    
    # Combine: 4 polynomial + 6 B-spline = 10 total
    # But we need exactly 10, so use 4 poly + 6 bspline
    if bspline_basis.shape[1] >= 6:
        b4 = bspline_basis[:, 0]
        b5 = bspline_basis[:, 1]
        b6 = bspline_basis[:, 2]
        b7 = bspline_basis[:, 3]
        b8 = bspline_basis[:, 4]
        b9 = bspline_basis[:, 5]
    else:
        # Fallback to Fourier if B-spline computation fails
        b4 = np.sin(x)
        b5 = np.cos(x)
        b6 = np.sin(2.0 * x)
        b7 = np.cos(2.0 * x)
        b8 = np.sin(3.0 * x)
        b9 = np.cos(3.0 * x)
    
    # Stack into matrix (n, 10)
    basis = np.vstack([
        b_poly0, b_poly1, b_poly2, b_poly3,
        b4, b5, b6, b7,
        b8, b9
    ]).T
    
    # Linear combination: (n,)
    result = basis @ params
    
    # Check deadline (though this is very fast)
    if time.time() > deadline:
        # Return best effort (which is already computed)
        pass
        
    return result
# EVOLVE-BLOCK-END
