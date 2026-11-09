import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using a Hybrid Fourier-Response structure.
    Combines the strong prior multiplicative response of the inspirations (Hill, Gaussian)
    with a Fourier basis for capturing sharp transitions and periodic-like constraints.
    
    Structure:
    y = Amp * Hill(s) * (1 + C_b * b_norm) * (TempResp * PhResp + FourierTerm) + Bias
    
    The Fourier term allows the model to correct the smooth Gaussian priors with
    high-frequency components if the data exhibits sharp edges or complex oscillations.
    
    Parameters (10):
    0: log(Amp) - log of overall amplitude
    1: log(K_s) - log of substrate half-saturation constant
    2: hill_n - Hill coefficient (clipped to [0.5, 5.0])
    3: T_opt - Temperature optimum offset (added to prior 0.923)
    4: log(sigma_t) - log of temperature standard deviation
    5: pH_opt - pH optimum offset (added to prior 0.5)
    6: log(sigma_p) - log of pH standard deviation
    7: C_b - Density coefficient
    8: Bias - additive bias
    9: fourier_amp - Amplitude of the Fourier correction term (clipped to [-5, 5])
    """
    
    # Normalize inputs to [0, 1]
    # b: [1, 5] -> [0, 1]
    b_norm = np.clip((b - 1.0) / 4.0, 0.0, 1.0)
    # s: [1, 40] -> [0, 1]
    s_norm = np.clip((s - 1.0) / 39.0, 0.0, 1.0)
    # temp: [1, 40] -> [0, 1]
    temp_norm = np.clip((temp - 1.0) / 39.0, 0.0, 1.0)
    # pH: [2, 12] -> [0, 1]
    ph_norm = np.clip((pH - 2.0) / 10.0, 0.0, 1.0)
    
    # Unpack and transform parameters
    p0 = np.clip(params[0], -10, 10)
    p1 = np.clip(params[1], -10, 10)
    p2 = np.clip(params[2], 0.5, 5.0)
    p3 = np.clip(params[3], -0.5, 1.5)
    p4 = np.clip(params[4], -10, 10)
    p5 = np.clip(params[5], -0.5, 1.5)
    p6 = np.clip(params[6], -10, 10)
    p7 = params[7]
    p8 = params[8]
    p9 = np.clip(params[9], -5.0, 5.0)
    
    log_amp = p0
    log_ks = p1
    hill_n = p2
    T_opt_offset = p3
    log_sig_t = p4
    pH_opt_offset = p5
    log_sig_p = p6
    C_b = p7
    bias = p8
    fourier_amp = p9
    
    Amp = np.exp(log_amp)
    K_s = np.exp(log_ks)
    sigma_t = np.exp(log_sig_t)
    sigma_p = np.exp(log_sig_p)
    
    # Substrate: Hill Function
    # s_term = s^n / (K^n + s^n)
    s_n = s_norm ** hill_n
    K_n = K_s ** hill_n
    s_term = s_n / (K_n + s_n + 1e-9)
    
    # Temperature: Gaussian Response
    t_center = 0.923 + T_opt_offset
    t_diff = temp_norm - t_center
    t_exponent = -0.5 * (t_diff / sigma_t) ** 2
    t_exponent = np.clip(t_exponent, -100, 100)
    temp_term = np.exp(t_exponent)
    
    # pH: Gaussian Response
    ph_center = 0.5 + pH_opt_offset
    ph_diff = ph_norm - ph_center
    ph_exponent = -0.5 * (ph_diff / sigma_p) ** 2
    ph_exponent = np.clip(ph_exponent, -100, 100)
    ph_term = np.exp(ph_exponent)
    
    # Fourier Correction Term
    # We add a small set of fixed-frequency sine/cosine terms to allow the model
    # to capture sharp transitions or local variations not captured by the smooth Gaussians.
    # Frequencies are fixed to avoid optimization instability, only amplitude is learned.
    # This acts as a "local basis function" layer with fixed centers/frequencies.
    
    # Base frequencies: 2*pi * [1, 2, 3] for a few harmonics
    # We sum a few terms: sin(w * x + phase)
    # To keep it simple and differentiable, we use a fixed set of frequencies.
    
    # Temperature Fourier
    w_t = 2 * np.pi * np.array([1.0, 2.0, 3.0])
    ft_t = np.sin(w_t[0] * temp_norm) + 0.5 * np.sin(w_t[1] * temp_norm) + 0.25 * np.sin(w_t[2] * temp_norm)
    
    # pH Fourier
    w_p = 2 * np.pi * np.array([1.0, 2.0, 3.0])
    ft_p = np.sin(w_p[0] * ph_norm) + 0.5 * np.sin(w_p[1] * ph_norm) + 0.25 * np.sin(w_p[2] * ph_norm)
    
    # Combine Fourier terms:
    # We scale them by fourier_amp and add them to the multiplicative response.
    # The structure is: Amp * s_term * dens_term * (temp_term * ph_term + fourier_correction) + bias
    
    # The fourier_correction should be small and local.
    # Let's define the correction as: fourier_amp * (ft_t * temp_term + ft_p * ph_term)
    # This ensures the Fourier terms only modulate the regions where the base response is active.
    
    fourier_correction = fourier_amp * (ft_t * temp_term + ft_p * ph_term)
    
    # Density: Linear modulation
    dens_term = 1.0 + C_b * b_norm
    
    # Combine
    # The base response is temp_term * ph_term
    # The total environmental response is (temp_term * ph_term + fourier_correction)
    # We clip the environmental response to be non-negative to avoid unphysical negative growth rates
    # from the Fourier term overpowering the Gaussian in certain regions, or simply allow it to be signed
    # if the bias handles the offset. Given the target range includes negative values, signed is okay.
    
    env_response = temp_term * ph_term + fourier_correction
    
    result = Amp * s_term * dens_term * env_response + bias
    
    # Ensure finite
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return result
