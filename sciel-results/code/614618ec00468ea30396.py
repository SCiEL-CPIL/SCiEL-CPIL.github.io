import numpy as np

def equation(b, s, temp, pH, params):
    p = params
    
    # Normalize inputs to [0, 1] based on known ranges
    xb = (b - 1.0) / 4.0
    xs = (s - 1.0) / 4.0
    xT = (temp - 1.0) / 39.0
    xPH = (pH - 2.0) / 10.0
    
    # Hinge basis functions: max(0, x - c)
    # This creates a piecewise-linear basis that can capture non-smooth thresholds
    # while remaining convex in the parameters.
    
    # For each variable, we create a few hinge terms.
    # We use fixed knots to keep the number of parameters manageable (max 10).
    # Let's allocate parameters wisely.
    
    # We will use a linear combination of hinge functions for each input.
    # y = sum_i (a_i * max(0, xb - c_i) + b_i * max(0, xs - c_i) + ...) + offset
    
    # To fit within 10 params, we need to be strategic.
    # Let's use 2 hinges per variable for the most critical ones, or a shared structure.
    # Actually, the directive suggests "piecewise-linear basis function expansion".
    # Let's try a simple approach:
    # y = c0 + c1*xb + c2*max(0, xb-0.5) + c3*xs + c4*max(0, xs-0.5) + c5*xT + c6*max(0, xT-0.5) + c7*xPH + c8*max(0, xPH-0.5) + c9*max(0, xT-0.8)
    
    # This is a linear model in the parameters!
    # This makes the optimization problem convex (quadratic loss on linear model).
    # BFGS should converge very reliably and quickly.
    
    # Let's define the basis matrix implicitly.
    
    # Hinges for b (density)
    h_b1 = np.maximum(0.0, xb - 0.5)
    
    # Hinges for s (substrate)
    h_s1 = np.maximum(0.0, xs - 0.5)
    
    # Hinges for temp
    h_t1 = np.maximum(0.0, xT - 0.5)
    h_t2 = np.maximum(0.0, xT - 0.8) # Another knot for temperature peak/decay
    
    # Hinges for pH
    h_p1 = np.maximum(0.0, xPH - 0.5)
    
    # Linear terms
    l_b = xb
    l_s = xs
    l_t = xT
    l_p = xPH
    
    # Construct prediction
    # p[0]: offset
    # p[1]: coeff for l_b
    # p[2]: coeff for h_b1
    # p[3]: coeff for l_s
    # p[4]: coeff for h_s1
    # p[5]: coeff for l_t
    # p[6]: coeff for h_t1
    # p[7]: coeff for h_t2
    # p[8]: coeff for l_p
    # p[9]: coeff for h_p1
    
    pred = (p[0] + 
            p[1] * l_b + p[2] * h_b1 +
            p[3] * l_s + p[4] * h_s1 +
            p[5] * l_t + p[6] * h_t1 + p[7] * h_t2 +
            p[8] * l_p + p[9] * h_p1)
            
    return pred
