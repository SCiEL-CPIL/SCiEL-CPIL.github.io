"""
Strategy: Non-linear Parameterized Physics-Informed Model.

Unlike the current program which uses a linear-in-parameters basis (fixed shape
polynomials and tanh), this approach uses a model where the parameters control
the SHAPE of the non-linear terms (slopes and shifts of saturation functions).
This creates a non-linear optimization landscape where BFGS can adapt the
"sharpness" of the yield point and the curvature of hardening to the data,
rather than just scaling fixed basis functions.

Model:
Stress = Offset 
       + Elastic_Term(Strain, Temp) 
       + Plastic_Yield_Term(Strain, Temp) 
       + Hardening_Term(Strain, Temp)

Parameters (10):
p[0]: Global Offset
p[1]: Elastic Modulus (Linear Strain Coeff)
p[2]: Thermal Expansion/Softening (Linear Temp Coeff)
p[3]: Yield Sharpness (Slope of first tanh)
p[4]: Yield Amplitude (Magnitude of first tanh)
p[5]: Yield Point (Shift of first tanh)
p[6]: Hardening Sharpness (Slope of second tanh)
p[7]: Hardening Amplitude (Magnitude of second tanh)
p[8]: Strain-Temp Interaction
p[9]: Temp-Squared Interaction
"""

import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses learnable tanh saturation functions to model plastic yield and hardening.
    """
    # Extract parameters
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Center temperature for numerical stability and to reduce correlation
    # Temp range is [0.066, 1.0]. Center around 0.5
    T_c = temp - 0.5
    
    # Strain terms
    s1 = strain
    s2 = strain * strain
    
    # 
    # 1. Elastic/Linear Component
    # Standard linear strain and linear temp effect
    linear_part = p1 * s1 + p2 * T_c
    
    # 
    # 2. Plastic Yield Component (First Saturation)
    # Models the transition from elastic to plastic.
    # tanh(p3 * (s1 - p5)) 
    # p3 controls sharpness (large p3 = sharp yield), p5 controls yield point location
    # p4 controls the magnitude of the plastic stress addition
    # We use a safe tanh to avoid overflow if p3*s is large
    arg_yield = p3 * (s1 - p5)
    # Clip argument to prevent numerical issues in tanh (though tanh is stable, 
    # gradients vanish for large args, which is good for BFGS stability)
    tanh_yield = np.tanh(arg_yield)
    yield_part = p4 * tanh_yield
    
    # 
    # 3. Hardening Component (Second Saturation)
    # Models strain hardening which eventually saturates.
    # Uses strain squared to capture curvature more strongly at higher strains
    # tanh(p6 * (s2 - p5*p5)) ... this is a bit weird dimensionally.
    # Better: tanh(p6 * (s1 - p5)) but with different slope?
    # Or: A separate tanh for higher order effects.
    # Let's use a tanh on the residual strain to model hardening saturation.
    # Hardening often looks like (strain - yield_strain)^n.
    # A tanh of the residual is a good smooth approximation.
    residual = s1 - p5
    arg_harden = p6 * residual
    tanh_harden = np.tanh(arg_harden)
    harden_part = p7 * tanh_harden
    
    # 
    # 4. Interaction Terms
    # Temperature affects the yield point and hardening.
    # Simple multiplicative/additive interactions with centered temp.
    interaction_part = p8 * s1 * T_c + p9 * T_c * T_c
    
    # Combine
    pred = p0 + linear_part + yield_part + harden_part + interaction_part
    
    return pred
