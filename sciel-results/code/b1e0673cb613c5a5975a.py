import numpy as np

def equation(x, y, params):
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Pre-compute trigonometric basis functions
    # Using pi * x and pi * y maps [-1, 1] to [-pi, pi], covering one full period for sin/cos of order 1.
    cx1 = np.cos(np.pi * x)
    sx1 = np.sin(np.pi * x)
    cy1 = np.cos(np.pi * y)
    sy1 = np.sin(np.pi * y)
    
    cx2 = np.cos(2 * np.pi * x)
    cy2 = np.cos(2 * np.pi * y)
    sx2 = np.sin(2 * np.pi * x)
    sy2 = np.sin(2 * np.pi * y)
    
    # Construct the linear combination
    # p0: DC term
    # p1, p2: First order cos
    # p3, p4: First order sin
    # p5: Product cos*cos (1,1)
    # p6: Product sin*sin (1,1)
    # p7, p8: Second order cos
    # p9: Product sin*sin (2,2) - or maybe cross terms?
    
    # Let's stick to a simple set of 10 basis functions.
    # 1. 1
    # 2. cos(pi*x)
    # 3. cos(pi*y)
    # 4. sin(pi*x)
    # 5. sin(pi*y)
    # 6. cos(pi*x)*cos(pi*y)
    # 7. sin(pi*x)*sin(pi*y)
    # 8. cos(2*pi*x)
    # 9. cos(2*pi*y)
    # 10. sin(2*pi*x)*sin(2*pi*y)
    
    term1 = 1.0
    term2 = cx1
    term3 = cy1
    term4 = sx1
    term5 = sy1
    term6 = cx1 * cy1
    term7 = sx1 * sy1
    term8 = cx2
    term9 = cy2
    term10 = sx2 * sy2
    
    return p0 * term1 + p1 * term2 + p2 * term3 + p3 * term4 + p4 * term5 + \
           p5 * term6 + p6 * term7 + p7 * term8 + p8 * term9 + p9 * term10
