import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a separable tensor-product Chebyshev basis.
    
    We use T_i(x) * T_j(y) for i, j in {0, 1, 2, 3, 4} (5x5 = 25 terms),
    but we only have 10 parameters. To handle this, we use a structured
    decomposition:
    
    f(x, y) = sum_{i=0}^{4} a_i * T_i(x) * T_0(y)  +  sum_{j=1}^{4} b_j * T_0(x) * T_j(y)
    
    This gives 5 + 4 = 9 terms, plus 1 bias = 10 parameters.
    This captures rank-1 structure in each direction separately.
    
    However, a better approach: use a full 3x3 grid (i,j in {0,1,2}) = 9 terms + 1 bias = 10 params.
    This gives T_i(x) * T_j(y) for i,j in {0,1,2}, which is a full rank-3 tensor product.
    
    Chebyshev polynomials:
    T_0(t) = 1
    T_1(t) = t
    T_2(t) = 2t^2 - 1
    T_3(t) = 4t^3 - 3t
    T_4(t) = 8t^4 - 8t^2 + 1
    """
    start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Compute Chebyshev polynomials up to degree 4 for x and y
    # T_0 = 1
    # T_1 = t
    # T_2 = 2t^2 - 1
    # T_3 = 4t^3 - 3t
    # T_4 = 8t^4 - 8t^2 + 1
    
    x2 = x * x
    y2 = y * y
    
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2.0 * x2 - 1.0
    Tx3 = 4.0 * x2 * x - 3.0 * x
    Tx4 = 8.0 * x2 * x2 - 8.0 * x2 + 1.0
    
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2.0 * y2 - 1.0
    Ty3 = 4.0 * y2 * y - 3.0 * y
    Ty4 = 8.0 * y2 * y2 - 8.0 * y2 + 1.0
    
    # Use a 3x3 tensor product: T_i(x) * T_j(y) for i,j in {0,1,2}
    # That's 9 basis functions + 1 bias = 10 parameters
    # 
    # params[0] * T_0(x)*T_0(y) + params[1] * T_1(x)*T_0(y) + params[2] * T_2(x)*T_0(y)
    # + params[3] * T_0(x)*T_1(y) + params[4] * T_1(x)*T_1(y) + params[5] * T_2(x)*T_1(y)
    # + params[6] * T_0(x)*T_2(y) + params[7] * T_1(x)*T_2(y) + params[8] * T_2(x)*T_2(y)
    # + params[9] * 1 (bias)
    
    # Wait, T_0(x)*T_0(y) = 1, so params[0] and params[9] would be redundant.
    # Let me restructure: use 9 distinct basis functions + 1 bias
    # Actually T_0(x)*T_0(y) = 1, so I should exclude it and use 9 other terms + bias
    # Or better: use the 9 terms T_i(x)*T_j(y) for i,j in {0,1,2} where NOT (i=0 AND j=0),
    # plus one bias term.
    
    # Let me use a different split: 
    # Terms: T_i(x)*T_j(y) for (i,j) in {(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0)}
    # That's 9 terms + 1 bias = 10 params
    # 
    # Actually, let me just use a clean 3x3 grid but note that T_0(x)*T_0(y)=1
    # So I'll use 8 non-trivial terms from the 3x3 grid + 1 bias + 1 extra term
    # 
    # Simpler approach: use the full 3x3 grid (9 terms) where the (0,0) term is the bias
    # So params[0] is the bias (multiplied by T_0(x)*T_0(y)=1)
    
    result = (params[0] * Tx0 * Ty0 +
              params[1] * Tx1 * Ty0 +
              params[2] * Tx2 * Ty0 +
              params[3] * Tx0 * Ty1 +
              params[4] * Tx1 * Ty1 +
              params[5] * Tx2 * Ty1 +
              params[6] * Tx0 * Ty2 +
              params[7] * Tx1 * Ty2 +
              params[8] * Tx2 * Ty2)
    
    # That's 9 terms. I have 10 params. Let me add one more useful term.
    # Add T_3(x) * T_0(y) = T_3(x)
    result = result + params[9] * Tx3 * Ty0
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        # Fallback: simple linear
        result = params[0] + params[1] * x + params[2] * y
    
    return result
