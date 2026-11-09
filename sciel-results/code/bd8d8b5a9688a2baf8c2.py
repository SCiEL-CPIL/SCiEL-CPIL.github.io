import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation modeled as a sum of two tanh-activated units.
    
    Parameters:
    t: Time
    x: Position
    v: Velocity
    params: Array of 10 fitted constants
        p[0]: Bias term
        p[1]: Weight for unit 1
        p[2]: Input weight t for unit 1
        p[3]: Input weight x for unit 1
        p[4]: Input weight v for unit 1
        p[5]: Bias for unit 1
        p[6]: Weight for unit 2
        p[7]: Input weight t for unit 2
        p[8]: Input weight x for unit 2
        p[9]: Input weight v for unit 2
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Unit 1: primarily state-dependent (x, v)
    # Note: p2 is weight for t in unit 1, but we can set it to 0 effectively if not needed, 
    # but here we use all params. Let's assign:
    # Unit 1: tanh(p2*t + p3*x + p4*v + p5)
    # Unit 2: tanh(p7*t + p8*x + p9*v) -- missing bias for unit 2? 
    # Let's restructure to use 10 params effectively.
    
    # Structure: 
    # y = p0 + p1 * tanh(p2*t + p3*x + p4*v + p5) + p6 * tanh(p7*t + p8*x + p9*v)
    # This uses 10 params.
    
    u1 = p2 * t + p3 * x + p4 * v + p5
    u2 = p7 * t + p8 * x + p9 * v
    
    return p0 + p1 * np.tanh(u1) + p6 * np.tanh(u2)
