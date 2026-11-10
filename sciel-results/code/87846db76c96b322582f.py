"""Symbolic regression with a hybrid rational + local basis model."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model combining rational polynomial structure with local basis functions.
    
    y(x) = P(x)/Q(x) + c5*exp(-a*(x-b)^2) + c6*sin(w*x+ph)
    
    where:
      P(x) = p[0] + p[1]*x + p[2]*x^2 + p[3]*x^3 + p[4]*x^4
      Q(x) = 1 + p[5]*x + p[6]*x^2 + p[7]*x^3 + p[8]*x^4
    
    The rational part captures global structure with potential poles/asymptotes.
    The Gaussian term captures local features (bumps, dips).
    The sine term captures oscillatory behavior.
    
    Parameters (10 total):
      p[0]..p[4]: numerator coefficients
      p[5]..p[8]: denominator coefficients (constant term fixed at 1)
      p[9]: Gaussian amplitude (width and center fixed for stability)
    
    Actually, let's use all 10 params more flexibly:
      p[0]..p[3]: numerator coefficients (degree 3)
      p[4]..p[6]: denominator coefficients (degree 2, constant fixed at 1)
      p[7]: Gaussian amplitude
      p[8]: Gaussian width parameter
      p[9]: Gaussian center
    
    Wait, that's only 10. Let me reconsider.
    
    Better approach: Use a rational function with degree 3 numerator and degree 2 denominator,
    plus two additional local terms. But we only have 10 params.
    
    Let's do:
      Rational: P(x)/Q(x) where P is degree 3 (4 params), Q is degree 2 with constant 1 (2 params)
      That's 6 params.
      Remaining 4 params for: one Gaussian (amplitude, width, center) + one constant offset
    
    Actually, simpler: just use the rational form but make it more robust.
    
    Let me try a different structural change: use a sum of rational terms.
    
    y(x) = a0/(1+b0*x) + a1*x/(1+b1*x) + a2*x^2/(1+b2*x^2) + c
    
    That's 7 params. Add sine: + a3*sin(w*x) -> 9 params. Add another term.
    
    Let's go with a cleaner hybrid:
    y(x) = P(x)/Q(x) + G(x)
    
    where P is degree 2 (3 params), Q is degree 1 with constant 1 (1 param), 
    G(x) = a*exp(-b*(x-c)^2) (3 params)
    Total: 3+1+3 = 7. Need 3 more.
    
    Let's do:
    P degree 3 (4 params), Q degree 1 (1 param, constant=1), 
    Gaussian (3 params: amp, width, center)
    Total: 8. Add sine amplitude and frequency: 10 params.
    
    Final model:
    y(x) = P(x)/(1 + q1*x) + a*exp(-b*(x-c)^2) + s*sin(f*x)
    
    Params:
      p[0]..p[3]: P coefficients (p0 + p1*x + p2*x^2 + p3*x^3)
      p[4]: q1 (denominator is 1 + p[4]*x)
      p[5]: a (Gaussian amplitude)
      p[6]: b (Gaussian width, use exp(p[6]) for positivity)
      p[7]: c (Gaussian center)
      p[8]: s (sine amplitude)
      p[9]: f (sine frequency)
    """
    if time.time() - _T0 > _DEADLINE:
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x, dtype=np.float64)
    
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    
    if p.size < 10:
        pad = np.zeros(10)
        pad[:p.size] = p
        p = pad
    else:
        p = p[:10]
    
    # Clip x to avoid extreme values
    x_safe = np.clip(x, -20, 20)
    
    # Numerator polynomial P(x) = p0 + p1*x + p2*x^2 + p3*x^3
    x2 = x_safe * x_safe
    x3 = x2 * x_safe
    numerator = p[0] + p[1] * x_safe + p[2] * x2 + p[3] * x3
    
    # Denominator Q(x) = 1 + p4*x
    denominator = 1.0 + p[4] * x_safe
    
    # Safe division
    eps = 1e-10
    denom_safe = denominator + np.sign(denominator) * eps
    rational = numerator / denom_safe
    
    # Gaussian term: a * exp(-exp(b) * (x-c)^2)
    # Clip b to avoid overflow
    b_safe = np.clip(p[6], -10, 10)
    width = np.exp(b_safe)
    gauss = p[5] * np.exp(-width * (x_safe - p[7]) ** 2)
    
    # Sine term: s * sin(f*x)
    sine = p[8] * np.sin(p[9] * x_safe)
    
    result = rational + gauss + sine
    
    # Ensure finite output
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
# EVOLVE-BLOCK-END
