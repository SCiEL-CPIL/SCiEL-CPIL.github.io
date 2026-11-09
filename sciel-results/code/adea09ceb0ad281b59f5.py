"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We keep the function signature: equation(strain, temp, params) -> np.ndarray
# The evaluator will fit `params` (length 10) using BFGS from all ones.
# We need to design a closed-form expression with 10 parameters that can capture
# the stress-strain curve of aluminium, which typically has:
# 1. Elastic region (linear)
# 2. Yield point
# 3. Plastic region (strain hardening, possibly power-law or exponential)
# 4. Temperature effects (softening at higher temperatures)

# A common model for metals is:
# stress = E * strain * (1 - strain/strain_max) for small strains (elastic with some hardening)
# or use a more flexible form like:
# stress = E * strain / (1 + A * strain^2)  (hyperbolic approximation)
# Or: stress = E * strain * exp(-B * strain)  (exponential decay of stiffness)

# Let's use a flexible model that can capture:
# - Linear elastic behavior at small strain
# - Strain hardening or saturation at large strain
# - Temperature dependence

# Model: stress = f(strain, temp)
# Let's try: stress = (E_eff * strain) / (1 + k * strain^n) + T_effect
# where E_eff depends on temperature

# More specifically, let's use:
# stress = (p0 * (p1 + p2*temp) * strain) / (1 + p3 * strain^p4) + p5*temp + p6
# plus additional terms for better fitting

# Actually, let's use a more standard approach for metal stress-strain:
# True stress-strain often follows: sigma = K * epsilon^n (Hollomon)
# But engineering stress has a different shape.

# Let's use a combination that's flexible:
# stress = p0 * strain * (1 + p1*strain) / (1 + p2*strain^2 + p3*strain^4) 
#         * (1 + p4*temp + p5*temp^2)
#         + p6*strain + p7*temp + p8

# That's 9 parameters. Let's add one more for a shift or additional nonlinearity.

# Better approach: use a rational function that can fit various curves.

# Let's define:
# stress = (p0 + p1*strain + p2*strain^2 + p3*temp + p4*temp^2 + p5*strain*temp)
#        / (1 + p6*strain + p7*strain^2 + p8*temp + p9*strain*temp)

# This is a bivariate rational function with 10 parameters. It's very flexible
# and can approximate many smooth functions. The denominator ensures it doesn't
# blow up if parameters are reasonable.

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a hybrid model with tanh saturation and rational correction.
    """
    # Ensure inputs are 1D arrays
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Core non-linear term: tanh captures yield and saturation
    # p1 controls stiffness/slope of tanh
    # p2 allows temperature to shift the yield point
    core = p0 * np.tanh(p1 * strain + p2 * temp)
    
    # Linear elastic/plastic correction
    linear = p3 * strain + p4
    
    # Numerator combines core and linear parts
    num = core + linear
    
    # Denominator: Rational structure to handle complex interactions and prevent blowup
    # p5, p6, p7 control the shape of the rational decay/correction
    den = 1.0 + p5 * strain**2 + p6 * temp + p7 * strain * temp
    
    # Avoid division by zero
    den_safe = np.where(np.abs(den) < 1e-10, 1e-10 * np.sign(den + 1e-300), den)
    
    # Main rational part
    main_part = num / den_safe
    
    # Additional temperature effects (softening/stiffening) outside the main rational
    temp_effect = p8 * temp + p9
    
    stress = main_part + temp_effect
    
    return stress
# EVOLVE-BLOCK-END
