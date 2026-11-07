import numpy as np
import time

_START_TIME = time.time()

def equation(x, y, params):
    # Deadline guard: safety margin for the 150s limit
    if time.time() - _START_TIME > 145:
        return np.zeros_like(x)

    # Adaptive-frequency parametric sinusoids
    # Model: offset + A1*sin(w1*x + v1*y + p1) + A2*sin(w2*x + v2*y + p2) + linear_x
    # This allows BFGS to adjust frequencies and phases to fit sinusoidal targets.
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]

    theta1 = p2 * x + p3 * y + p4
    theta2 = p6 * x + p7 * y + p8

    term1 = p1 * np.sin(theta1)
    term2 = p5 * np.sin(theta2)
    term3 = p9 * x

    result = p0 + term1 + term2 + term3

    return result
