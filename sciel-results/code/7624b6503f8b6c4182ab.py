import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a piecewise-linear basis (linear splines).
    
    Strategy:
    - Use 10 parameters corresponding to 10 hat functions (and potentially edge handling).
    - To ensure local support and well-conditioning for BFGS, we define fixed knots 
      across the domain [-3, 3].
    - We use 10 basis functions. Let's place 10 knots evenly spaced.
    - To handle the edges of the domain robustly with only 10 params, we can use 
      'bump' functions or specific hat functions that cover the range.
    
    A simpler, robust approach for 10 params on [-3, 3]:
    Divide the domain into 9 intervals (10 knots).
    Knots: -3, -2.333, -1.666, ..., 3.
    Basis functions:
    - phi_0(x): ramp up from -3 to first interior knot, then 0? No, standard splines need care at boundaries.
    
    Let's use a set of triangular "hat" functions centered at specific points.
    If we use 10 parameters, we can define 10 basis functions.
    Let centers be c_i.
    To cover [-3, 3] well, let's use centers:
    c = np.linspace(-3, 3, 10)
    However, hats at the edges will be half-triangles if defined symmetrically.
    
    Alternative: Use a global basis but scaled and shifted to be well-conditioned?
    No, the inspiration suggests local basis is better for BFGS.
    
    Let's define 10 basis functions:
    We can use 10 "hats" with support width 2/3 (spacing between centers).
    Centers: -3, -2.33, ..., 3.
    Actually, let's just use 10 evenly spaced knots and construct the basis matrix directly.
    
    Knots: k_i = -3 + i * (6/9) for i=0..9.
    Basis functions:
    We need 10 parameters.
    If we use standard linear splines, we usually have n_knots - 1 or similar degrees of freedom.
    With 10 knots, we have 9 intervals.
    
    Let's try a different local basis: Gaussian-like bumps? No, linear is safer for "closed form" simplicity and BFGS.
    
    Let's define 10 basis functions as follows:
    We partition [-3, 3] into 9 equal sub-intervals of length 2/3.
    Knots: x_0 = -3, x_1, ..., x_9 = 3.
    
    We will use 10 basis functions:
    1. Left edge ramp: phi_0(x) = max(0, 1 - (x - x_0)/h) for x in [x_0, x_1], 0 else? 
       This is a triangle starting at x_0.
    Actually, a simpler and very robust "local" basis that BFGS loves is just the hat functions 
    associated with each interior knot plus edge ramps.
    
    Let's use 10 parameters for 10 basis functions:
    Basis 0: Linear ramp from -3 to -2.333... (value 1 at -3, 0 at -2.333)
    Basis 1: Hat centered at -2.333... (0 at -3, 1 at -2.333, 0 at -1.666)
    ...
    Basis 8: Hat centered at 2.333...
    Basis 9: Linear ramp down from 2.333... to 3 (value 0 at 2.333, 1 at 3)
    
    Wait, the centers need to be consistent.
    Let h = 6/9 = 2/3.
    Knots: -3, -7/3, -5/3, -1, -1/3, 1/3, 1, 5/3, 7/3, 3.
    
    Basis 0: Support [-3, -7/3]. phi_0(x) = 1 if x=-3, linearly goes to 0 at -7/3.
             phi_0(x) = 1 - (x - (-3))/h for x in [-3, -3+h]. 0 elsewhere.
             At x=-3, val=1. At x=-3+h, val=0.
             
    Basis 1: Support [-3, -3+2h]. Center -3+h.
             phi_1(x) = (x - (-3))/h for x in [-3, -3+h].
             phi_1(x) = 1 - (x - (-3+h))/h for x in [-3+h, -3+2h].
             0 elsewhere.
             
    ...
    Basis 8: Support [3-2h, 3]. Center 3-h.
             phi_8(x) = (x - (3-2h))/h for x in [3-2h, 3-h].
             phi_8(x) = 1 - (x - (3-h))/h for x in [3-h, 3].
             
    Basis 9: Support [3-h, 3].
             phi_9(x) = (x - (3-h))/h for x in [3-h, 3].
             0 elsewhere.
             At x=3, val=1.
             
    This creates a set of 10 overlapping triangular functions that span the domain.
    This basis is locally supported (each param only affects ~1/3 of the domain),
    which decouples the optimization problem significantly compared to global polynomials.
    BFGS should converge much faster and more reliably.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Domain [-3, 3]
    # 9 intervals, 10 knots
    h = 6.0 / 9.0
    knots = np.linspace(-3.0, 3.0, 10)
    
    n_basis = 10
    n_points = x.size
    T = np.zeros((n_points, n_basis), dtype=np.float64)
    
    # Basis 0: Left edge
    # Support: [knots[0], knots[1]]
    # Value 1 at knots[0], 0 at knots[1]
    mask0 = (x >= knots[0]) & (x <= knots[1])
    T[mask0, 0] = 1.0 - (x[mask0] - knots[0]) / h
    
    # Basis 1 to 8: Interior hats
    # i corresponds to knots[i] as center
    # Support: [knots[i-1], knots[i+1]]
    for i in range(1, 9):
        left = knots[i-1]
        center = knots[i]
        right = knots[i+1]
        
        # Left slope: [left, center]
        mask_left = (x >= left) & (x <= center)
        T[mask_left, i] = (x[mask_left] - left) / h
        
        # Right slope: [center, right]
        mask_right = (x > center) & (x <= right)
        T[mask_right, i] = 1.0 - (x[mask_right] - center) / h
        
    # Basis 9: Right edge
    # Support: [knots[8], knots[9]]
    # Value 0 at knots[8], 1 at knots[9]
    mask9 = (x >= knots[8]) & (x <= knots[9])
    T[mask9, 9] = (x[mask9] - knots[8]) / h
    
    return T @ params
