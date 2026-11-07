import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Fourier series basis.
    
    The model is f(x) = a_0 + sum_{k=1}^{4} [a_k cos(2*pi*k*x/L) + b_k sin(2*pi*k*x/L)]
    where L is the domain length. This uses 10 parameters: a_0, a_1..a_4, b_1..b_4.
    
    Fourier basis functions are orthogonal on bounded intervals and can capture
    periodic or oscillatory behavior that polynomial bases may represent poorly.
    The loss is quadratic in the parameters (linear model), so BFGS should converge
    to the global minimum.
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has length 10
    if len(params) < 10:
        p = np.zeros(10, dtype=np.float64)
        p[:len(params)] = params
    else:
        p = params[:10]
    
    n_points = x.shape[0]
    result = np.zeros(n_points, dtype=np.float64)
    
    # Domain: x in [0.1, 6.0], so L = 5.9
    x_min = 0.1
    x_max = 6.0
    L = x_max - x_min
    
    # a_0 term (constant)
    result += p[0]
    
    # For k = 1 to 4: a_k * cos(2*pi*k*x/L) + b_k * sin(2*pi*k*x/L)
    # params layout: [a_0, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4]
    # That's only 9. Let me use: [a_0, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c]
    # where c is an additional linear term or we use 5 harmonics.
    
    # Better: use 5 harmonics with only cosine, or 4 harmonics with both.
    # Let's use: a_0, then for k=1..4: a_k*cos, b_k*sin -> that's 1 + 8 = 9
    # Add one more: use k=5 cosine term, or a linear term.
    # I'll use: a_0, a_1..a_5 (6 cosines) + b_1..b_4 (4 sines) = 10
    
    # Actually, simpler: use 4 harmonics (8 params) + a_0 + one extra
    # Let me do: a_0, a_1..a_4, b_1..b_4, and a_5 (5th cosine)
    
    # Recompute with clear indexing:
    # p[0] = a_0
    # p[1] = a_1, p[2] = a_2, p[3] = a_3, p[4] = a_4, p[5] = a_5
    # p[6] = b_1, p[7] = b_2, p[8] = b_3, p[9] = b_4
    
    for k in range(1, 6):
        result += p[k] * np.cos(2.0 * np.pi * k * x / L)
    
    for k in range(1, 5):
        result += p[5 + k] * np.sin(2.0 * np.pi * k * x / L)
    
    return result
