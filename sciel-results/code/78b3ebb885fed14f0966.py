import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a rank-3 separable Chebyshev tensor-product basis.
    
    The 10 parameters are linear weights over 10 Chebyshev tensor-product basis functions:
      T_i(x) * T_j(y) for i, j in {0, 1, 2}
    
    This gives a 3x3 grid of basis functions (9 terms) plus a constant-like term
    (which is T0(x)*T0(y) already included). Actually, 3x3 = 9 terms.
    We need 10 terms. Let's use:
      T_i(x) * T_j(y) for i in {0,1,2,3}, j in {0,1}  -> 8 terms
      plus T_0(x)*T_2(y) and T_0(x)*T_3(y) -> 2 more terms? 
      
    Let's stick to the directive: "first 5 Chebyshev polynomials for each variable independently 
    (i.e., T_i(x)T_j(y) for i,j in {0,1,2})" -> this is 3x3=9 terms.
    But we have 10 params. 
    The directive says: "Implement a separable tensor-product basis using the first 5 Chebyshev 
    polynomials for each variable independently (i.e., T_i(x)T_j(y) for i,j in {0,1,2})"
    This phrasing is slightly contradictory ("first 5" vs "i,j in {0,1,2}"). 
    {0,1,2} is 3 polynomials. 3x3=9.
    Maybe it means up to degree 2 in each, plus one higher order term?
    Or maybe the directive meant {0,1,2,3,4} but that's 25 terms.
    
    Let's interpret "first 5 Chebyshev polynomials" as T0..T4, but only use a subset.
    The current best score is from a degree-3 total polynomial (10 terms).
    The current program (score 36M) is a neural net.
    The inspirations (score 3.1B) are degree-3 total Chebyshev.
    
    The directive asks for a STRUCTURAL change. The neural net is structurally different 
    from the polynomial basis. The polynomial basis is well-conditioned and BFGS-friendly.
    The neural net has fixed internal weights, which might be poorly scaled or ill-conditioned.
    
    Let's combine the "separable tensor-product" idea with the "well-conditioned Chebyshev" idea.
    A 3x3 grid of Chebyshev polynomials (T0..T2 in x, T0..T2 in y) gives 9 terms.
    We need 10. We can add T3(x)*T0(y) to get the 10th term, making it slightly asymmetric 
    but capturing a bit more detail in x. Or add T0(x)*T3(y).
    Alternatively, use T0..T3 in x and T0..T1 in y (4*2=8) + T0(x)*T2(y) + T0(x)*T3(y)?
    
    Let's try a symmetric 3x3 grid (9 terms) + 1 extra term for a 4th order effect in one direction.
    Actually, the "degree 3 total" basis from inspirations is:
    1 (deg 0)
    x, y (deg 1)
    x^2, xy, y^2 (deg 2)
    x^3, x^2y, xy^2, y^3 (deg 3)
    Total 10 terms.
    
    The directive suggests "separable tensor-product basis ... T_i(x)T_j(y) for i,j in {0,1,2}".
    This is exactly the 3x3 grid.
    Terms:
    T0(x)T0(y)
    T0(x)T1(y)
    T0(x)T2(y)
    T1(x)T0(y)
    T1(x)T1(y)
    T1(x)T2(y)
    T2(x)T0(y)
    T2(x)T1(y)
    T2(x)T2(y)
    
    This is 9 terms. The 10th parameter can be used for a T3(x)T0(y) term or similar.
    Let's use T3(x)T0(y) as the 10th term to break symmetry slightly and add a cubic term in x.
    
    This basis is well-conditioned (Chebyshev) and separable (tensor product), which aligns with the directive.
    """
    t_start = time.time()
    TIME_LIMIT = 140.0
    
    # Chebyshev polynomials of the first kind
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x
    
    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y
    
    # Construct the 10 basis functions
    # 3x3 grid of T0..T2 for x and y, plus T3(x)*T0(y)
    f0 = T0_x * T0_y
    f1 = T0_x * T1_y
    f2 = T0_x * T2_y
    f3 = T1_x * T0_y
    f4 = T1_x * T1_y
    f5 = T1_x * T2_y
    f6 = T2_x * T0_y
    f7 = T2_x * T1_y
    f8 = T2_x * T2_y
    f9 = T3_x * T0_y  # Extra term to reach 10 params
    
    basis = np.column_stack((f0, f1, f2, f3, f4, f5, f6, f7, f8, f9))
    
    prediction = basis @ params
    
    elapsed = time.time() - t_start
    if elapsed > TIME_LIMIT:
        return prediction
    
    return prediction
