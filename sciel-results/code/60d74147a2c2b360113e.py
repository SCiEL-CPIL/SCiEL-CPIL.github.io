import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Piecewise-Linear Spline with Smooth Transitions (B-spline-like basis).
    
    The current program uses a global parametric form (Elastic + Power Law + Exponential).
    This approach is constrained by the specific functional form of each term.
    
    This new strategy uses a set of localized, smooth "bump" functions (Gaussian RBFs)
    superimposed on a linear elastic base. This allows the model to flexibly
    approximate the stress-strain curve, including the sharp yield point and
    subsequent hardening/softening, without being locked into a specific
    hardening law (like Ramberg-Osgood or exponential saturation).
    
    The 10 parameters are:
    p0: Elastic modulus (slope of initial linear part)
    p1: Temperature coefficient for elastic modulus
    p2: Offset (intercept)
    p3: Temperature coefficient for offset
    p4-p9: Amplitudes of 3 Gaussian bumps (centers and widths are fixed to known
           regions of interest: yield point, mid-hardening, and late hardening).
    
    Why fixed centers/widths?
    - BFGS struggles with highly non-linear parameters like centers and widths
      when starting from all ones.
    - By fixing them to physically plausible locations (based on the strain range
      [-0.002, 0.96]), we reduce the non-linearity of the optimization problem.
      The amplitudes are linear parameters, which BFGS handles very well.
    - This makes the problem "semi-linear": the amplitudes are linear, but the
      elastic slope and intercept still have temperature dependence (non-linear
      in the data if t varies, but linear in params for a given t).
    
    The Gaussian bumps are:
    1. Yield bump: centered at s=0.01, width=0.02 (sharp transition)
    2. Mid-hardening bump: centered at s=0.2, width=0.1
    3. Late-hardening bump: centered at s=0.5, width=0.2
    
    We also add a linear background term to capture the overall trend.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # Base linear elastic term
    # sigma_elastic = (p0 + p1*t) * s + (p2 + p3*t)
    sigma_elastic = (p0 + p1 * t) * s + (p2 + p3 * t)
    
    # Gaussian bumps
    # Fixed centers and widths based on expected strain range
    # s is in [-0.002, 0.96]
    
    # Bump 1: Yield region
    c1 = 0.01
    w1 = 0.02
    g1 = np.exp(-((s - c1) / w1) ** 2)
    
    # Bump 2: Mid-hardening region
    c2 = 0.2
    w2 = 0.1
    g2 = np.exp(-((s - c2) / w2) ** 2)
    
    # Bump 3: Late-hardening region
    c3 = 0.5
    w3 = 0.2
    g3 = np.exp(-((s - c3) / w3) ** 2)
    
    # Amplitudes of the bumps
    # We allow the amplitudes to depend on temperature linearly as well,
    # but to save parameters, we'll just use constant amplitudes for now.
    # If we need more flexibility, we can add temperature coefficients later.
    amp1 = p4
    amp2 = p5
    amp3 = p6
    
    # We have 4 more parameters (p7, p8, p9) unused.
    # Let's add a 4th bump for very late strain or a different region.
    # Or, let's make the amplitudes temperature-dependent.
    
    # Let's make the amplitudes temperature-dependent:
    # amp1 = p4 + p7*t
    # amp2 = p5 + p8*t
    # amp3 = p6 + p9*t
    
    # This uses all 10 parameters and allows the shape of the curve to change with temperature.
    
    amp1 = p4 + p7 * t
    amp2 = p5 + p8 * t
    amp3 = p6 + p9 * t
    
    sigma_bumps = amp1 * g1 + amp2 * g2 + amp3 * g3
    
    stress = sigma_elastic + sigma_bumps
    
    return stress
