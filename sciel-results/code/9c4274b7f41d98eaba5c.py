import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Mixture of Experts (MoE) model with a logistic gate.
    
    Parameters:
    p[0], p[1]   : Linear expert coefficients (a0, a1)
    p[2], p[3], p[4], p[5] : Cubic expert coefficients (b0, b1, b2, b3)
    p[6], p[7]   : Gate parameters (scale k, center c)
    p[8], p[9]   : Global scale S, offset O
    
    Model:
    L(x) = p[0] + p[1]*x
    C(x) = p[2] + p[3]*x + p[4]*x^2 + p[5]*x^3
    gate = sigmoid(p[6]*(x - p[7]))
    y = p[8] * (gate * L(x) + (1 - gate) * C(x)) + p[9]
    """
    # Extract parameters
    a0, a1 = params[0], params[1]
    b0, b1, b2, b3 = params[2], params[3], params[4], params[5]
    k, c = params[6], params[7]
    S, O = params[8], params[9]
    
    # Compute Experts
    L = a0 + a1 * x
    C = b0 + b1 * x + b2 * x**2 + b3 * x**3
    
    # Compute Gate
    # Use stable sigmoid
    z = k * (x - c)
    # Clip to prevent overflow in exp
    z_clipped = np.clip(z, -50, 50)
    gate = 1.0 / (1.0 + np.exp(-z_clipped))
    
    # Combine
    y = S * (gate * L + (1.0 - gate) * C) + O
    
    return y
