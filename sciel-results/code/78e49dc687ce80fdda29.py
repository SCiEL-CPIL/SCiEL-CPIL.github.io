import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Maximize P[X1 + X2 + X3 < 2*X4] using a 3-point discrete distribution.
    Support: {0, 1, 2}.
    Weights: p0, p1, p2.
    """
    
    def objective(params):
        p0, p1 = params
        p2 = 1.0 - p0 - p1
        # Ensure non-negative
        if p0 < -1e-9 or p1 < -1e-9 or p2 < -1e-9:
            return 1e9
        p0 = max(0.0, p0)
        p1 = max(0.0, p1)
        p2 = max(0.0, p2)
        
        # P = p2 * (1 - (3*p2^2 - 2*p2^3)) + p1 * (p0^3 + 3*p0^2*p1)
        # Term 1: X4=2 (p2). Success if <2 of X1..X3 are 2.
        # Fail if >=2 are 2.
        # P(fail|2) = 3*p2^2*(1-p2) + p2^3 = 3*p2^2 - 2*p2^3
        term1 = p2 * (1.0 - (3.0*p2**2 - 2.0*p2**3))
        
        # Term 2: X4=1 (p1). Success if X1..X3 in {0,1} and sum < 2.
        # P(all in {0,1}) = (p0+p1)^3 = (1-p2)^3
        # Given in {0,1}, let q1 = p1/(1-p2), q0 = p0/(1-p2)
        # P(sum<2) = q0^3 + 3*q0^2*q1
        # Term = (1-p2)^3 * ( (p0/(1-p2))^3 + 3*(p0/(1-p2))^2*(p1/(1-p2)) )
        #      = p0^3 + 3*p0^2*p1
        if p2 == 1.0:
            term2 = 0.0
        else:
            term2 = p1 * (p0**3 + 3.0*p0**2*p1)
            
        # Term 3: X4=0 (p0). Success prob 0.
        
        return -(term1 + term2) # Minimize negative probability

    # Initial guess based on analysis
    # p0 ~ 0.5, p1 ~ 0.3, p2 ~ 0.2
    x0 = np.array([0.5, 0.3])
    
    # Bounds
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    # Constraints: p0 + p1 <= 1
    constraints = [
        {'type': 'ineq', 'fun': lambda x: 1.0 - x[0] - x[1]}
    ]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    p0, p1 = res.x
    p2 = 1.0 - p0 - p1
    
    # Clip to ensure validity
    p0 = max(0.0, p0)
    p1 = max(0.0, p1)
    p2 = max(0.0, p2)
    
    # Normalize just in case
    total = p0 + p1 + p2
    if total <= 0:
        p0, p1, p2 = 1/3, 1/3, 1/3
    else:
        p0 /= total
        p1 /= total
        p2 /= total
        
    support = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    weights = np.array([p0, p1, p2], dtype=np.float64)
    
    return support, weights
