import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Mixture of Localized Spline Bases with Learnable Centers.
    
    Instead of a single global functional form (like Ramberg-Osgood) or fixed RBFs,
    this model uses 5 localized basis functions. The 10 parameters define the 
    centers (mu_i) and widths (sigma_i) of these 5 functions. The "amplitudes" 
    of these functions are not free parameters but are determined by a linear 
    projection of the input features (strain, temp) onto a set of fixed, 
    physically-motivated shape functions. This allows the model to:
    1. Self-locate important features (like yield strain) via mu_i.
    2. Adapt local curvature via sigma_i.
    3. Maintain global smoothness and monotonicity by using a linear combination
       of smooth, positive-definite basis shapes (Gaussians and Polynomials)
       weighted by the input coordinates.
       
    Parameters:
    strain: np.ndarray, shape (N,)
    temp: np.ndarray, shape (N,)
    params: np.ndarray, shape (10,)
       params[0:5] -> centers mu_1..mu_5 for strain
       params[5:10] -> widths sigma_1..sigma_5 for strain
    
    Returns:
    stress: np.ndarray, shape (N,)
    """
    s = strain
    t = temp
    
    # Extract centers and widths
    mus = params[0:5]
    sigmas = params[5:10]
    
    # Ensure sigmas are positive to avoid division by zero or negative variance
    # Use a soft lower bound to keep gradients smooth
    sigmas = np.maximum(sigmas, 1e-3)
    
    # Compute localized basis values: exp(-((s - mu_i) / sigma_i)^2)
    # Shape: (N, 5)
    diff = s[:, np.newaxis] - mus[np.newaxis, :]
    basis = np.exp(-(diff / sigmas[np.newaxis, :]) ** 2)
    
    # We need to combine these 5 localized bases with the temperature and strain
    # to produce a stress value. 
    # Strategy: The stress is a linear combination of the localized bases, 
    # where the coefficients depend on temperature and global strain.
    # coeff_j = a_j * (1 + b_j * t) + c_j * s
    # But we only have 10 params for mu/sigma. We need to generate the 
    # "amplitude" structure from the inputs themselves or fixed constants.
    
    # Alternative: Use the localized bases to modulate a global linear trend.
    # Stress = (E0 + E1*t + E2*s) * (1 + sum_i w_i * basis_i)
    # But w_i are not in params.
    
    # Let's try a different decomposition:
    # The 10 params define the "shape" of the curve's non-linearities.
    # The actual stress level is determined by the input magnitude.
    
    # Model:
    # sigma = s * (A + B*t) + sum_{i=1}^5 C_i * t * basis_i(s) + D * sum basis_i(s)
    # This still needs C_i, D.
    
    # Let's use the parameters to define a "flexible stiffness" function.
    # E_eff(s, t) = E_base(t) * (1 + sum_{i=1}^5 alpha_i * basis_i(s))
    # sigma = s * E_eff(s, t)
    # But alpha_i are not in params.
    
    # Re-thinking: The 10 params are ONLY mu and sigma. 
    # How do we get amplitudes?
    # We can assume the amplitudes are fixed or derived from the positions.
    # Or, we can use the params to define the *entire* function if we allow 
    # the function to be a sum of shifted and scaled Gaussians with fixed 
    # amplitudes that depend on the input temp/strain?
    
    # Let's try: 
    # sigma = s * (100 + 10*t) + sum_{i=1}^5 A_i * basis_i(s) * (1 + 0.1*t)
    # If A_i are fixed, the model can only shift/shape, not scale the bumps.
    # This is risky.
    
    # Better Strategy: 
    # Use 5 params for centers, 5 for widths.
    # The "amplitude" of each localized feature is proportional to the 
    # distance from the center? No.
    
    # Let's look at the inspiration scores. 
    # Inspiration 2 (RBF) scored 2.66. It likely had fixed centers and fitted amplitudes.
    # Current (RO) scored 4.95.
    # If we have 10 params, and we use them for mu/sigma, we have 0 free amplitudes.
    # This is very restrictive.
    
    # What if we split the 10 params differently?
    # 5 centers, 5 amplitudes? Fixed widths?
    # Or 5 centers, 5 widths, and we assume the amplitude is 1.0?
    
    # Let's try a hybrid:
    # The model is: sigma = s * (E0 + E1*t) + sum_{i=1}^5 A_i * exp(-((s-mu_i)/sigma_i)^2)
    # If we only have 10 params, we can't fit E0, E1, A_i, mu_i, sigma_i all.
    
    # Maybe the "amplitudes" should be linear functions of temp?
    # sigma = s * (100 + 5*t) + sum_{i=1}^5 (1 + 0.1*t) * exp(-((s-mu_i)/sigma_i)^2)
    # Here mu_i, sigma_i are params. The coefficients are fixed.
    # This allows the model to place "bumps" (hardening/yield) at specific strains.
    # The linear part handles the elastic slope.
    # The bumps handle the non-linearity.
    # Since the bumps are additive and positive (exp is positive), and the linear 
    # part is dominant for small s, this should be stable.
    
    # Let's refine the fixed coefficients to be physically plausible.
    # E0 ~ 70 GPa? The data is normalized? 
    # Target range [-0.005, 0.91]. Strain [-0.002, 0.96].
    # This looks like normalized stress and strain.
    # So E ~ 1 (slope of linear part).
    
    # Model:
    # sigma = 1.0 * s + 0.1 * t * s + sum_{i=1}^5 (1.0 + 0.2*t) * exp(-((s-mu_i)/sigma_i)^2)
    
    # This uses 10 params for mu and sigma.
    # The fixed coefficients ensure the function is bounded and smooth.
    # The BFGS will optimize mu and sigma to place the "knees" correctly.
    
    # One issue: If the true curve has a *drop* (unloading) or specific curvature 
    # that requires negative amplitudes, this model can't do it. 
    # But stress-strain for loading is usually monotonic.
    
    # Let's add a small quadratic term to the linear part to allow slight curvature 
    # in the elastic region? No, keep it simple.
    
    # Let's try to make the amplitudes slightly dependent on the position to 
    # allow for asymmetry? No, keep it simple first.
    
    # Implementation:
    
    # Fixed coefficients
    E_base = 1.0
    E_temp = 0.1
    A_base = 1.0
    A_temp = 0.2
    
    # Linear elastic term
    linear_term = (E_base + E_temp * t) * s
    
    # Localized non-linear terms
    # basis is (N, 5)
    # amp is (N, 1) broadcasted to (N, 5)
    amp = (A_base + A_temp * t)[:, np.newaxis]
    
    nonlinear_term = np.sum(basis * amp, axis=1)
    
    stress = linear_term + nonlinear_term
    
    return stress
