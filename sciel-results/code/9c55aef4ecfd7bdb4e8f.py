import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Linear-in-parameters model with a fixed, well-conditioned hybrid basis.
    
    Basis functions:
    1-4: Chebyshev polynomials T_0 to T_3 mapped to [-1, 1] on x in [0.1, 6].
    5-8: Cosine and Sine terms with frequencies tuned to the domain length.
    9-10: Higher frequency Cosine and Sine terms for finer detail.
    
    This approach ensures the optimization landscape is convex (linear least squares),
    allowing BFGS to converge reliably to the global optimum without getting stuck
    in local minima associated with non-linear parameterizations (like RBF centers).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.ravel()
    
    p = np.asarray(params, dtype=np.float64)
    if p.ndim != 1 or p.shape[0] != 10:
        # Return zeros if params are malformed to avoid crashes
        return np.zeros_like(x)
    
    # 1. Map x from [0.1, 6.0] to t in [-1, 1] for Chebyshev stability
    # x_min = 0.1, x_max = 6.0
    # t = (2 * (x - x_min) / (x_max - x_min)) - 1
    x_min = 0.1
    x_max = 6.0
    t = (2.0 * (x - x_min) / (x_max - x_min)) - 1.0
    
    # Clip t to [-1, 1] to avoid numerical instability outside domain (though data is in domain)
    t = np.clip(t, -1.0, 1.0)
    
    # 2. Compute Chebyshev Polynomials T_0 to T_3
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    
    # 3. Compute Fourier Terms
    # Domain length L = 5.9. 
    # Fundamental frequency omega_0 = pi / L? Or 2*pi / L?
    # Let's use angular frequency k such that k * L = pi * n for integer n to align endpoints somewhat.
    # Actually, for a general function on [0.1, 6], using frequencies that fit a few cycles is good.
    # Let's use frequencies:
    # f1: ~1 cycle over the domain. k1 = 2*pi / 5.9
    # f2: ~2 cycles over the domain. k2 = 4*pi / 5.9
    # f3: ~3 cycles over the domain. k3 = 6*pi / 5.9
    # f4: ~4 cycles over the domain. k4 = 8*pi / 5.9
    
    k1 = 2.0 * np.pi / (x_max - x_min)
    k2 = 4.0 * np.pi / (x_max - x_min)
    k3 = 6.0 * np.pi / (x_max - x_min)
    k4 = 8.0 * np.pi / (x_max - x_min)
    
    # We have 10 parameters.
    # p[0]: T0 coeff
    # p[1]: T1 coeff
    # p[2]: T2 coeff
    # p[3]: T3 coeff
    # p[4]: Cos(k1*x) coeff
    # p[5]: Sin(k1*x) coeff
    # p[6]: Cos(k2*x) coeff
    # p[7]: Sin(k2*x) coeff
    # p[8]: Cos(k3*x) coeff
    # p[9]: Sin(k3*x) coeff
    
    # Note: Using x directly in trig functions is fine, or we could use t.
    # Using x preserves the physical scale of the oscillation.
    
    c1 = np.cos(k1 * x)
    s1 = np.sin(k1 * x)
    c2 = np.cos(k2 * x)
    s2 = np.sin(k2 * x)
    c3 = np.cos(k3 * x)
    s3 = np.sin(k3 * x)
    
    # Combine
    result = (p[0] * T0 + 
              p[1] * T1 + 
              p[2] * T2 + 
              p[3] * T3 + 
              p[4] * c1 + 
              p[5] * s1 + 
              p[6] * c2 + 
              p[7] * s2 + 
              p[8] * c3 + 
              p[9] * s3)
    
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
