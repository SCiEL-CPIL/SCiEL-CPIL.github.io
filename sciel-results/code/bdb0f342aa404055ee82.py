import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0

    p = params

    # Mechanistic-inspired model: Monod kinetics for substrate,
    # Gaussian for temperature, and log-normal for pH.
    # This uses 10 parameters:
    # p[0]: max growth rate (mu_max)
    # p[1]: half-saturation constant for substrate (Ks)
    # p[2]: temperature optimum (T_opt) in [0,1] scale
    # p[3]: temperature width (sigma_T)
    # p[4]: pH optimum (pH_opt) in [0,1] scale
    # p[5]: pH width (sigma_pH)
    # p[6]: biomass inhibition constant (Ki)
    # p[7]: biomass inhibition strength
    # p[8]: base growth rate offset
    # p[9]: additional scaling factor

    # Monod term for substrate: s / (Ks + s)
    # Ensure Ks is positive by using abs or sigmoid transformation
    Ks = np.abs(p[1]) + 1e-6
    monod_s = s_norm / (Ks + s_norm)

    # Gaussian for temperature: exp(-((t - T_opt)^2) / (2 * sigma_T^2))
    T_opt = p[2]
    sigma_T = np.abs(p[3]) + 1e-6
    gauss_t = np.exp(-((t_norm - T_opt) ** 2) / (2 * sigma_T ** 2))

    # Log-normal or Gaussian for pH: exp(-((pH - pH_opt)^2) / (2 * sigma_pH^2))
    pH_opt = p[4]
    sigma_pH = np.abs(p[5]) + 1e-6
    gauss_ph = np.exp(-((ph_norm - pH_opt) ** 2) / (2 * sigma_pH ** 2))

    # Biomass inhibition: 1 / (1 + (b/Ki)^n) or similar
    # Using a simple inhibition: exp(-p[7] * b_norm)
    # Or competitive inhibition: 1 / (1 + b_norm / Ki)
    Ki = np.abs(p[6]) + 1e-6
    inhib_b = 1.0 / (1.0 + b_norm / Ki)

    # Combine all terms multiplicatively
    y = p[0] * monod_s * gauss_t * gauss_ph * inhib_b + p[8] + p[9] * b_norm

    return y


