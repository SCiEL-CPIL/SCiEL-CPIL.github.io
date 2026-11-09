import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1] range to improve conditioning
    # b: [1, 5]
    xb = (b - 1.0) / 4.0
    # s: [1, 5]
    xs = (s - 1.0) / 4.0
    # temp: [1, 40]
    xT = (temp - 1.0) / 39.0
    # pH: [2, 12]
    xPH = (pH - 2.0) / 10.0
    
    # Quadratic Additive Model + one interaction
    # y = p0 + p1*xb + p2*xs + p3*xT + p4*xPH 
    #   + p5*xb^2 + p6*xs^2 + p7*xT^2 + p8*xPH^2 
    #   + p9*xb*xs
    
    # This is linear in params, so BFGS will be very stable and fast.
    # It captures main effects, curvature, and b-s interaction.
    
    return (p[0] 
            + p[1] * xb 
            + p[2] * xs 
            + p[3] * xT 
            + p[4] * xPH 
            + p[5] * xb**2 
            + p[6] * xs**2 
            + p[7] * xT**2 
            + p[8] * xPH**2 
            + p[9] * xb * xs)
