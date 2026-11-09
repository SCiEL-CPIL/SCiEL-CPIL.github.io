import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: Use a 3-point support {0, 1, M} with large M.
    Optimize weights w_1, w_M.
    """
    
    def prob_3point(u, v):
        # u = w_1, v = w_M
        # w_0 = 1 - u - v
        if u < 0 or v < 0 or u + v > 1:
            return 0.0
        w0 = 1.0 - u - v
        w1 = u
        wM = v
        
        # P | X4=1: w0^3 + 3*w1*w0^2
        p_x4_1 = w0**3 + 3.0 * w1 * w0**2
        
        # P | X4=M: 1 - P(>=2 Ms)
        # P(2 Ms) = 3 * wM^2 * (w0+w1) = 3 * wM^2 * (1-wM)
        # P(3 Ms) = wM^3
        # P(valid) = 1 - (3 * wM^2 * (1-wM) + wM^3) = 1 - 3*wM^2 + 3*wM^3 - wM^3 = 1 - 3*wM^2 + 2*wM^3
        p_x4_M = 1.0 - 3.0 * wM**2 + 2.0 * wM**3
        
        # P | X4=0: 0
        
        return w1 * p_x4_1 + wM * p_x4_M

    def neg_prob(params):
        u, v = params
        return -prob_3point(u, v)

    # Initial guess
    # Based on intuition, maybe w0 is large, wM is moderate.
    # Let's try a few starting points.
    best_val = -1.0
    best_params = (0.5, 0.5)
    
    starts = [
        (0.1, 0.1),
        (0.2, 0.2),
        (0.5, 0.1),
        (0.1, 0.5),
        (0.3, 0.3),
        (0.0, 0.8), # w0=0.2
        (0.8, 0.0), # wM=0
    ]
    
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]},
        {'type': 'ineq', 'fun': lambda x: 1.0 - x[0] - x[1]}
    ]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    for start in starts:
        try:
            res = minimize(neg_prob, start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_params = res.x
        except Exception:
            pass

    u_opt, v_opt = best_params
    w0_opt = 1.0 - u_opt - v_opt
    
    # Ensure non-negative
    w0_opt = max(0.0, w0_opt)
    u_opt = max(0.0, u_opt)
    v_opt = max(0.0, v_opt)
    
    # Normalize just in case
    total = w0_opt + u_opt + v_opt
    if total == 0:
        w0_opt, u_opt, v_opt = 1.0, 0.0, 0.0
    else:
        w0_opt /= total
        u_opt /= total
        v_opt /= total

    M = 10.0 # Any M > 2 works
    
    support = np.array([0.0, 1.0, M], dtype=np.float64)
    weights = np.array([w0_opt, u_opt, v_opt], dtype=np.float64)
    
    return support, weights
