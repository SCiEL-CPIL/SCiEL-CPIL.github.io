import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Scale x to [-1, 1] for Chebyshev polynomials
    t = x / 2.0
    
    # Chebyshev polynomials T_0 to T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    # T_4(t) = 8*t^4 - 8*t^2 + 1
    
    b0 = np.ones_like(x)
    b1 = t
    b2 = 2 * t**2 - 1
    b3 = 4 * t**3 - 3 * t
    b4 = 8 * t**4 - 8 * t**2 + 1
    
    # Gaussians with evenly spaced centers in [-1.5, 1.5]
    # Centers: -1.5, -0.75, 0, 0.75, 1.5
    # Width sigma = 0.5
    sigma = 0.5
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    
    # Compute Gaussians
    # x is (N, 1) or (N,), centers is (5,)
    # We need to broadcast correctly.
    # x shape is likely (N,). 
    # x[:, np.newaxis] - centers[np.newaxis, :]
    
    # Ensure x is 1D for calculation, then reshape if needed? 
    # The input x is an array of points.
    # Let's assume x is 1D.
    
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    gaussians = np.exp(-0.5 * (diff / sigma)**2)
    
    # gaussians shape is (N, 5)
    # b0..b4 are (N,)
    
    # Construct full basis matrix
    # B = [b0, b1, b2, b3, b4, g0, g1, g2, g3, g4]
    
    # Efficient calculation
    # y = params[0]*b0 + params[1]*b1 + params[2]*b2 + params[3]*b3 + params[4]*b4
    #   + params[5]*g0 + ... + params[9]*g4
    
    # Using dot product is cleaner and faster
    # Create basis matrix B of shape (N, 10)
    B = np.column_stack((b0, b1, b2, b3, b4, gaussians))
    
    return B @ params
