import numpy as np
import time

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Nonlinear oscillator acceleration using a physically-motivated basis
    with explicit scaling for BFGS convergence from params=1.
    
    The key insight is that the BFGS optimizer starts at params=all ones,
    so we want the optimal params to be close to 1. We scale each basis
    function by the expected magnitude of its coefficient so that when
    the true system has coefficients of order ~1 in our scaled basis,
    the optimal params will be near 1.
    
    We use a richer set of terms including:
    - Linear restoring and damping (dominant)
    - Cubic restoring (Duffing)
    - Nonlinear damping terms
    - Cross terms
    - Higher order corrections
    
    All terms are scaled so that for a "typical" oscillator with 
    coefficients in the range suggested by the data ranges, the 
    optimal params will be O(1).
    
    Inputs:
      x: position, in [-0.5728, 0.5275]
      v: velocity, in [-0.1754, 0.2821]
      params: array of 10 coefficients
      
    Returns:
      a: acceleration, in [-0.1047, 0.1106]
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).flatten()
    
    # Scale factors: we want each term * scale to be O(0.1) for typical inputs
    # x ~ 0.5, v ~ 0.2, a ~ 0.1
    
    # 1. Linear restoring: a ~ -k*x. For x~0.5, a~0.1, k~0.2
    #    So scale = 0.2, term = -x, optimal p[0] ~ 1
    b1 = -x * 0.2
    
    # 2. Linear damping: a ~ -c*v. For v~0.2, a~0.1, c~0.5
    b2 = -v * 0.5
    
    # 3. Cubic restoring (Duffing): a ~ -beta*x^3. For x~0.5, x^3~0.125, a~0.01, beta~0.1
    b3 = -x**3 * 0.1
    
    # 4. Quadratic velocity (damping correction): a ~ -gamma*v^2. For v~0.2, v^2~0.04, a~0.004, gamma~0.1
    b4 = -v**2 * 0.1
    
    # 5. VDP-like: a ~ mu*x*v. For x~0.5, v~0.2, xv~0.1, a~0.01, mu~0.1
    b5 = x * v * 0.1
    
    # 6. Cubic velocity: a ~ -delta*v^3. For v~0.2, v^3~0.008, a~0.0008, delta~0.1
    b6 = -v**3 * 0.1
    
    # 7. Quartic position: a ~ alpha*x^4. For x~0.5, x^4~0.0625, a~0.006, alpha~0.1
    b7 = x**4 * 0.1
    
    # 8. Cross term x^2*v: a ~ epsilon*x^2*v. For x^2~0.25, v~0.2, x^2*v~0.05, a~0.005, epsilon~0.1
    b8 = x**2 * v * 0.1
    
    # 9. Cross term x*v^2: a ~ zeta*x*v^2. For x~0.5, v^2~0.04, x*v^2~0.02, a~0.002, zeta~0.1
    b9 = x * v**2 * 0.1
    
    # 10. Quadratic position (bias): a ~ eta*x^2. For x^2~0.25, a~0.025, eta~0.1
    b10 = x**2 * 0.1
    
    a = (p[0] * b1 +
         p[1] * b2 +
         p[2] * b3 +
         p[3] * b4 +
         p[4] * b5 +
         p[5] * b6 +
         p[6] * b7 +
         p[7] * b8 +
         p[8] * b9 +
         p[9] * b10)
         
    return a
