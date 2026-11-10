"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers an unknown function using a Fourier series basis scaled to the domain [-3, 3].
    
    The model is: y = a_0 + sum_{k=1}^{5} [a_k * cos(k * pi * x / 3) + b_k * sin(k * pi * x / 3)]
    
    This basis is orthogonal on [-3, 3] for integer frequencies k, which provides
    good conditioning for BFGS optimization. It can capture periodic, oscillatory,
    and smooth non-periodic functions (via low-frequency components) effectively.
    
    params: array of length 10
      params[0]   -> a_0 (DC offset / bias)
      params[1]   -> a_1 (cos coefficient, k=1)
      params[2]   -> b_1 (sin coefficient, k=1)
      params[3]   -> a_2 (cos coefficient, k=2)
      params[4]   -> b_2 (sin coefficient, k=2)
      params[5]   -> a_3 (cos coefficient, k=3)
      params[6]   -> b_3 (sin coefficient, k=3)
      params[7]   -> a_4 (cos coefficient, k=4)
      params[8]   -> b_4 (sin coefficient, k=4)
      params[9]   -> a_5 (cos coefficient, k=5)
      Note: We use 10 params. To fit 5 full harmonics (cos+sin) plus DC, we need 11.
      Since we only have 10, we drop the last sine term (b_5) or combine differently.
      Let's use:
        p0: DC
        p1: cos(pi*x/3)
        p2: sin(pi*x/3)
        p3: cos(2*pi*x/3)
        p4: sin(2*pi*x/3)
        p5: cos(3*pi*x/3)
        p6: sin(3*pi*x/3)
        p7: cos(4*pi*x/3)
        p8: sin(4*pi*x/3)
        p9: cos(5*pi*x/3)
      This gives 5 cos terms and 4 sin terms. This is slightly asymmetric but fine.
      Alternatively, we could use a Chebyshev basis which is also orthogonal on [-1,1]
      and maps well to [-3,3]. The inspirations show Chebyshev scores well.
      However, the directive specifically suggests Fourier series.
      
      Let's stick to the Fourier suggestion but ensure we use all 10 params meaningfully.
      A common approach is to use the first 5 harmonics.
      DC + 5*cos + 4*sin = 10 params.
      Or DC + 4*cos + 5*sin = 10 params.
      Let's do:
      p0: 1
      p1: cos(pi*x/3)
      p2: sin(pi*x/3)
      p3: cos(2*pi*x/3)
      p4: sin(2*pi*x/3)
      p5: cos(3*pi*x/3)
      p6: sin(3*pi*x/3)
      p7: cos(4*pi*x/3)
      p8: sin(4*pi*x/3)
      p9: cos(5*pi*x/3)
    """
    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    
    # Ensure params has at least 10 elements
    if len(params) < 10:
        params = np.concatenate([params, np.ones(10 - len(params))])
    
    # Scale x to the fundamental frequency for the domain [-3, 3]
    # Fundamental period would be 6, so angular frequency w = pi/3
    # x_scaled = pi * x / 3
    x_scaled = np.pi * x / 3.0
    
    # Compute basis functions
    # k=1
    cos_1 = np.cos(x_scaled)
    sin_1 = np.sin(x_scaled)
    
    # k=2
    cos_2 = np.cos(2.0 * x_scaled)
    sin_2 = np.sin(2.0 * x_scaled)
    
    # k=3
    cos_3 = np.cos(3.0 * x_scaled)
    sin_3 = np.sin(3.0 * x_scaled)
    
    # k=4
    cos_4 = np.cos(4.0 * x_scaled)
    sin_4 = np.sin(4.0 * x_scaled)
    
    # k=5
    cos_5 = np.cos(5.0 * x_scaled)
    
    # Combine
    # p0 * 1 + p1*cos1 + p2*sin1 + ... + p9*cos5
    result = (
        params[0] * np.ones_like(x) +
        params[1] * cos_1 +
        params[2] * sin_1 +
        params[3] * cos_2 +
        params[4] * sin_2 +
        params[5] * cos_3 +
        params[6] * sin_3 +
        params[7] * cos_4 +
        params[8] * sin_4 +
        params[9] * cos_5
    )
    
    return result
# EVOLVE-BLOCK-END
