import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model for E. coli growth rate combining Monod kinetics,
    Gaussian-like temperature/pH optima, and density effects.
    
    Structural Design:
    1. Input Normalization: Scales inputs to ~[0, 1] for BFGS conditioning.
    2. Monod Kinetics for Substrate: Uses s/(Ks+s) form for biological saturation.
       Ks is parameterized as exp(p2 - 2) to ensure positivity.
    3. Gaussian Response for Temperature and pH: Uses exp(-((x-mu)/sigma)^2) form
       to capture optimal ranges, which is more biologically plausible than
       pure quadratics for growth rate optima.
    4. Linear/Quadratic Density: Retains flexibility for density effects.
    5. Interactions: Includes T*pH and Substrate*Temperature interactions.
    
    params layout (10 constants):
      p[0]: Bias
      p[1]: Monod scale (max growth contribution from substrate)
      p[2]: Log-Ks parameter for Monod (Ks = exp(p2 - 2))
      p[3]: Temperature mean (mu_T) in normalized space
      p[4]: Temperature width (sigma_T) in normalized space
      p[5]: pH mean (mu_pH) in normalized space
      p[6]: pH width (sigma_pH) in normalized space
      p[7]: Linear coeff for normalized Density (b)
      p[8]: Quadratic coeff for normalized Density (b)
      p[9]: Interaction coeff for normalized T * pH
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Normalize inputs to approximately [0, 1]
    # b: [1, 5] -> [0, 1]
    b_norm = (b - 1.0) / 4.0
    # s: [1, 5] -> [0, 1] (using upper bound 5 based on seed description, though inspiration 1 used 50)
    # The seed says [1, 4.998] for first input, [1, 4.999] for second.
    # Let's stick to the seed's implied range for s as well, or use a wider range if s goes up to 40?
    # Seed: [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
    # Input order: b, s, temp, pH
    # So b in [1, 5], s in [1, 5], temp in [1, 40], pH in [2, 12]
    s_norm = (s - 1.0) / 4.0
    # temp: [1, 40] -> [0, 1]
    T_norm = (temp - 1.0) / 39.0
    # pH: [2, 12] -> [0, 1]
    pH_norm = (pH - 2.0) / 10.0
    
    # Monod kinetics for substrate
    # p2 is initialized to 1.0. exp(1.0 - 2.0) = exp(-1) ~ 0.37.
    Ks = np.exp(p2 - 2.0)
    # Ensure Ks is not zero to avoid division by zero
    Ks_safe = np.maximum(Ks, 1e-8)
    monod_s = s_norm / (Ks_safe + s_norm)
    
    # Gaussian response for Temperature
    # mu_T (p3) and sigma_T (p4)
    # Ensure sigma is positive
    sigma_T = np.maximum(p4, 0.01)
    temp_term = np.exp(-((T_norm - p3) / sigma_T) ** 2)
    
    # Gaussian response for pH
    # mu_pH (p5) and sigma_pH (p6)
    sigma_pH = np.maximum(p6, 0.01)
    ph_term = np.exp(-((pH_norm - p5) / sigma_pH) ** 2)
    
    # Density terms
    density_term = p7 * b_norm + p8 * (b_norm ** 2)
    
    # Interaction term
    interaction = p9 * (T_norm * pH_norm)
    
    # Combine all terms
    # The substrate term scales the overall growth potential
    # The temperature and pH terms act as multipliers or additive factors?
    # Typically, growth rate is multiplicative: mu = mu_max * f(s) * f(T) * f(pH)
    # However, with only 10 params, a purely multiplicative model might be hard to fit
    # with BFGS starting from ones.
    # Let's try a semi-multiplicative approach:
    # y = p0 + p1 * monod_s * (temp_term + ph_term) + density_term + interaction
    # Or simpler additive with interactions:
    # y = p0 + p1 * monod_s + p1 * 0.5 * temp_term + p1 * 0.5 * ph_term + ...
    
    # Let's stick to a flexible additive form but use the Gaussian shapes
    # which capture the "optimum" better than quadratics.
    y = p0
    y += p1 * monod_s
    y += p1 * 0.5 * temp_term  # Scale temp contribution by substrate scale for consistency?
    y += p1 * 0.5 * ph_term
    y += density_term
    y += interaction
    
    return y
