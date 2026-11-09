import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Map x from [0.1, 6] to z in [-1, 1]
    # Center: 3.05, Half-width: 2.95
    # z = (x - 3.05) / 2.95
    z = (x - 3.05) / 2.95
    
    # Build basis
    # 1. Chebyshev-like polynomials (just powers of z are fine if scaled, 
    #    but let's use explicit Chebyshev for orthogonality)
    # T0 = 1
    # T1 = z
    # T2 = 2z^2 - 1
    # T3 = 4z^3 - 3z
    # T4 = 8z^4 - 8z^2 + 1
    
    t0 = np.ones_like(z)
    t1 = z
    t2 = 2 * z**2 - 1
    t3 = 4 * z**3 - 3 * z
    t4 = 8 * z**4 - 8 * z**2 + 1
    
    # 2. Trigonometric terms on original x
    # x is in [0.1, 6] ~ [0.1, 2*pi]
    s1 = np.sin(x)
    c1 = np.cos(x)
    s2 = np.sin(2 * x)
    c2 = np.cos(2 * x)
    
    # 3. Exponential decay on scaled z (bounded)
    # exp(-z) ranges from exp(1) to exp(-1) ~ [0.37, 2.7]
    e1 = np.exp(-z)
    
    # Stack basis
    basis = np.vstack([
        t0, t1, t2, t3, t4,
        s1, c1, s2, c2,
        e1
    ]).T
    
    return basis @ params
