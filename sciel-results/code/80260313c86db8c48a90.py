import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Fourier series basis on [-2, 2] with orthogonal frequencies
    # p[0]: DC (constant)
    # p[1]: cos(pi*x/2)   - 1st harmonic
    # p[2]: sin(pi*x/2)
    # p[3]: cos(pi*x)     - 2nd harmonic
    # p[4]: sin(pi*x)
    # p[5]: cos(3*pi*x/2) - 3rd harmonic
    # p[6]: sin(3*pi*x/2)
    # p[7]: cos(2*pi*x)   - 4th harmonic
    # p[8]: sin(2*pi*x)
    # p[9]: cos(5*pi*x/2) - 5th harmonic
    
    half_pi_x = 0.5 * np.pi * x
    pi_x = np.pi * x
    three_half_pi_x = 1.5 * np.pi * x
    two_pi_x = 2.0 * np.pi * x
    five_half_pi_x = 2.5 * np.pi * x
    
    y_hat = (p[0] * np.ones_like(x) +
             p[1] * np.cos(half_pi_x) +
             p[2] * np.sin(half_pi_x) +
             p[3] * np.cos(pi_x) +
             p[4] * np.sin(pi_x) +
             p[5] * np.cos(three_half_pi_x) +
             p[6] * np.sin(three_half_pi_x) +
             p[7] * np.cos(two_pi_x) +
             p[8] * np.sin(two_pi_x) +
             p[9] * np.cos(five_half_pi_x))
    
    return y_hat
