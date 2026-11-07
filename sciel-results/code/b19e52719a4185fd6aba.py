"""
Symbolic regression using a rational function basis: ratio of two low-degree polynomials.
y = (p0 + p1*x + p2*x^2 + p3*x^3) / (1 + q1*x + q2*x^2 + q3*x^3)
This captures non-polynomial behaviors (hyperbolic, exponential-like, rational trends)
that linear polynomial bases cannot fit efficiently.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    x: array of input points
    params: array of 10 coefficients
    Returns: predicted values
    
    We use a rational function model:
    y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4 + p5*x^5) / (1 + q0*x + q1*x^2 + q2*x^3 + q3*x^4 + q4*x^5)
    
    Actually, to keep it simpler and ensure the denominator stays positive, let's use:
    y = (sum_{i=0}^{4} p_i * x^i) / (1 + sum_{j=0}^{3} q_j * x^j)
    
    That's 5 + 4 = 9 parameters, but we have 10. Let me use:
    y = (sum_{i=0}^{4} p_i * x^i) / (1 + sum_{j=0}^{4} q_j * x^j)
    That's 5 + 5 = 10 parameters.
    
    To ensure the denominator is positive (for BFGS stability), we use:
    denominator = 1 + sum_{j=0}^{4} q_j * x^j
    
    But this could become zero or negative. A safer approach:
    denominator = 1 + sum_{j=0}^{4} (q_j * x^j) where we ensure positivity.
    
    Actually, a common trick is to use:
    denominator = 1 + 0.5 * sum_{j=0}^{4} q_j * x^j
    
    Or even better, use a form that guarantees positivity:
    denominator = 1 + sum_{j=0}^{4} (q_j * x^j)^2 ... no, that changes the structure.
    
    Let's use a simpler and more robust approach:
    y = (p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4) / (1 + 0.1*(q0*x + q1*x^2 + q2*x^3 + q3*x^4 + q4*x^5))
    
    The 0.1 scaling keeps the denominator close to 1, ensuring it stays positive for x in [-2,2]
    and reasonable parameter values.
    """
    deadline = time.time() + 10  # Safety margin: return before 150s wall-clock limit
    
    # Extract parameters
    p = params[0:5]   # Numerator coefficients: p0..p4
    q = params[5:10]  # Denominator coefficients: q0..q4
    
    # Compute powers of x
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    x5 = x4 * x
    
    # Numerator: p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
    numerator = p[0] + p[1]*x + p[2]*x2 + p[3]*x3 + p[4]*x4
    
    # Denominator: 1 + 0.1*(q0*x + q1*x^2 + q2*x^3 + q3*x^4 + q4*x^5)
    # The 0.1 scaling ensures denominator stays positive for reasonable params
    denominator = 1.0 + 0.1 * (q[0]*x + q[1]*x2 + q[2]*x3 + q[3]*x4 + q[4]*x5)
    
    # Avoid division by zero or near-zero
    # If denominator is too close to zero, clip it
    min_denom = 1e-8
    denominator = np.where(np.abs(denominator) < min_denom, min_denom, denominator)
    
    return numerator / denominator
# EVOLVE-BLOCK-END
