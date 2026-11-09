import numpy as np
import time

# Use a module-level start time for the deadline guard
_start_time = None

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Structural Change: 
    # Combine Chebyshev polynomials (global smoothness) with Gaussian RBFs (local adaptability).
    # This moves away from the pure polynomial/sigmoid mix which struggled with local features.
    # Chebyshev polynomials are well-conditioned on [-1, 1] (mapped from [-2, 2]).
    # Gaussian RBFs allow the model to capture local variations or sharp features that 
    # global polynomial bases might smooth over.
    
    # Mapping:
    # p[0]: T0 (constant)
    # p[1]: T1 (linear)
    # p[2]: T2 (quadratic)
    # p[3]: T3 (cubic)
    # p[4]: T4 (quartic)
    # p[5]: RBF at -1.5
    # p[6]: RBF at -0.5
    # p[7]: RBF at 0.5
    # p[8]: RBF at 1.5
    # p[9]: Sinusoidal term for potential periodicity
    
    # Scaled Chebyshev polynomials
    # Standard Chebyshev T_n(cos(theta)) = cos(n*theta)
    # For x in [-2, 2], let u = x/2, so u in [-1, 1].
    u = 0.5 * x
    u2 = u * u
    u3 = u2 * u
    u4 = u2 * u2
    
    T0 = np.ones_like(x)
    T1 = u
    T2 = 2.0 * u2 - 1.0
    T3 = 4.0 * u3 - 3.0 * u
    T4 = 8.0 * u4 - 8.0 * u2 + 1.0
    
    # Gaussian RBF terms
    # Fixed centers and widths to keep the problem linear in parameters (for BFGS)
    # and to provide a stable, differentiable basis.
    # Width sigma=0.6 provides moderate locality.
    sigma = 0.6
    sigma2 = 2.0 * sigma * sigma
    
    c1 = -1.5
    c2 = -0.5
    c3 = 0.5
    c4 = 1.5
    
    # Calculate RBFs
    # Using exp(-((x-c)^2) / (2*sigma^2))
    # To avoid underflow/overflow issues (though exp of negative is safe), 
    # we just compute directly. The values will be in [0, 1].
    g1 = np.exp(-((x - c1) ** 2) / sigma2)
    g2 = np.exp(-((x - c2) ** 2) / sigma2)
    g3 = np.exp(-((x - c3) ** 2) / sigma2)
    g4 = np.exp(-((x - c4) ** 2) / sigma2)
    
    # Sinusoidal term
    s = np.sin(x)
    
    # Combine
    y_hat = (p[0] * T0 + 
             p[1] * T1 + 
             p[2] * T2 + 
             p[3] * T3 + 
             p[4] * T4 + 
             p[5] * g1 + 
             p[6] * g2 + 
             p[7] * g3 + 
             p[8] * g4 + 
             p[9] * s)
             
    # Safety check for non-finite values
    if not np.all(np.isfinite(y_hat)):
        y_hat = np.where(np.isfinite(y_hat), y_hat, 0.0)
        
    return y_hat
