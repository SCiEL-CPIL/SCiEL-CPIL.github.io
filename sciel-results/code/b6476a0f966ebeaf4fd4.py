import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Structural change: Hybrid basis combining global smoothness (Chebyshev) 
    # with local adaptability (Gaussian RBFs).
    # This addresses the stagnation of pure polynomial or pure Fourier bases.
    # Chebyshev polynomials provide an efficient global approximation for smooth functions.
    # Gaussian RBFs allow the model to fit local features or discontinuities that 
    # global bases struggle with.
    
    # Parameters mapping:
    # p[0]: Chebyshev T0 coefficient (constant)
    # p[1]: Chebyshev T1 coefficient (linear)
    # p[2]: Chebyshev T2 coefficient (quadratic)
    # p[3]: Chebyshev T3 coefficient (cubic)
    # p[4]: RBF 1 amplitude
    # p[5]: RBF 2 amplitude
    # p[6]: RBF 3 amplitude
    # p[7]: RBF 4 amplitude
    # p[8]: RBF 5 amplitude
    # p[9]: RBF 6 amplitude
    
    # Fixed centers for RBFs distributed across the domain [-2, 2]
    # We use 6 RBFs with fixed centers and fixed width to keep the problem 
    # well-conditioned for BFGS.
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5, 2.0])
    width = 0.4  # Fixed width for stability
    
    # Compute Chebyshev polynomials T0(x), T1(x), T2(x), T3(x)
    # T0(x) = 1
    # T1(x) = x
    # T2(x) = 2x^2 - 1
    # T3(x) = 4x^3 - 3x
    
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2.0 * x * x - 1.0
    T3 = 4.0 * x * x * x - 3.0 * x
    
    # Compute RBF terms: exp(-(x - c)^2 / (2 * width^2))
    rbf_terms = []
    for c in centers:
        rbf = np.exp(-((x - c) ** 2) / (2.0 * width ** 2))
        rbf_terms.append(rbf)
    
    # Combine basis functions with learned parameters
    y_hat = (p[0] * T0 + 
             p[1] * T1 + 
             p[2] * T2 + 
             p[3] * T3 +
             p[4] * rbf_terms[0] + 
             p[5] * rbf_terms[1] + 
             p[6] * rbf_terms[2] + 
             p[7] * rbf_terms[3] + 
             p[8] * rbf_terms[4] + 
             p[9] * rbf_terms[5])
             
    return y_hat
