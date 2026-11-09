import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs
    # b: [1, 5] -> center 3, scale 2 -> [-1, 1]
    b_norm = (b - 3.0) / 2.0
    # s: [1, 40] -> keep raw for Monod
    # temp: [1, 40] -> center 20, scale 20 -> [-1, 1]
    temp_norm = (temp - 20.0) / 20.0
    # pH: [1, 12] -> center 7, scale 5 -> [-1, 1]
    ph_norm = (pH - 7.0) / 5.0
    
    # Monod term for substrate limitation
    # p1 is Ks. Init 1.0. Range of s is 1-40.
    monod = s / (p1 + s)
    
    # Saturating response for other factors using tanh
    # This allows for S-shaped responses which are common in biology
    # instead of the parabolic shape of the quadratic term.
    resp_b = np.tanh(p3 * b_norm)
    resp_t = np.tanh(p4 * temp_norm)
    resp_p = np.tanh(p5 * ph_norm)
    
    # Interaction term between b and temp
    inter_bt = b_norm * temp_norm
    
    # Combine responses
    # p2 is a base offset
    # p6, p7, p8 weight the individual saturating responses
    # p9 weights the interaction
    modifier = (p2 + 
                p6 * resp_b + 
                p7 * resp_t + 
                p8 * resp_p + 
                p9 * inter_bt)
    
    # p0 is the overall scale
    # We use p1 for Monod Ks.
    # We have 10 params: p0..p9
    # p0: scale
    # p1: Ks
    # p2: offset
    # p3: slope for b in tanh
    # p4: slope for t in tanh
    # p5: slope for p in tanh
    # p6: weight for b resp
    # p7: weight for t resp
    # p8: weight for p resp
    # p9: weight for interaction
    
    return p0 * monod * modifier
