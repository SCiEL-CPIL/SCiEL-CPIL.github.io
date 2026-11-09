"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a Learnable Skewed-Gaussian Product Mechanism.
    
    Strategy:
    1. Model the growth rate as a product of independent response functions for 
       Temperature (T), pH, Substrate (S), and Biomass (B). This multiplicative 
       structure is biologically plausible (growth is limited by the most 
       restrictive factor) and often easier for BFGS to fit than additive 
       mixtures because the parameters have clear physical interpretations 
       (optimum, width, amplitude).
    
    2. Use Skewed Gaussian (or generalized normal) functions for T and pH to 
       capture asymmetry. Thermal death is typically sharper (narrower) on the 
       high-temp side than thermal inhibition on the low-temp side. Similarly, 
       pH response may be asymmetric.
       Form: exp(-((x - mu) / sigma)^p) where p > 1 controls sharpness, but 
       to allow asymmetry, we use separate sigmas for left and right sides or 
       a skew parameter.
       
       A simpler robust approach for BFGS stability: 
       Use a "Gaussian with separate left/right widths" or a "Skew Normal".
       Let's use a parameterized skew normal approximation or simply two 
       half-gaussians joined at the mean.
       
       Actually, a very effective and differentiable form for asymmetric 
       peaks in optimization is:
       f(x) = exp(-0.5 * ((x - mu) / sigma)^2 * (1 + skew * (x - mu) / sigma))
       But this can be unstable.
       
       Alternative: Product of two Gaussians with different centers? No, that's a double peak.
       
       Let's use the "Generalized Normal Distribution" form:
       f(x) = exp(-| (x - mu) / sigma |^k)
       This captures sharpness (k) but is symmetric.
       
       To capture asymmetry (skew) with 10 params total:
       T-response: mu_T, sigma_T_left, sigma_T_right, amp_T (4 params)
       pH-response: mu_pH, sigma_pH_left, sigma_pH_right, amp_pH (4 params)
       S-response: amp_S, slope_S (2 params) - Monod-like or linear
       B-response: Fixed to 1 or shared? 
       We have 10 params.
       
       Let's allocate:
       p[0]: mu_T (Optimal Temp)
       p[1]: sigma_T_left (Width on low side)
       p[2]: sigma_T_right (Width on high side, usually smaller/sharper)
       p[3]: amp_T (Max growth factor from Temp)
       
       p[4]: mu_pH (Optimal pH)
       p[5]: sigma_pH_left
       p[6]: sigma_pH_right
       p[7]: amp_pH (Max growth factor from pH)
       
       p[8]: slope_S (Substrate effect)
       p[9]: bias (Global offset)
       
       Response Functions:
       R_T(t) = amp_T * exp(-0.5 * ((t - mu_T) / sigma_T)^2)
       To make it asymmetric, we can define sigma_T depending on sign(t - mu_T).
       sigma_eff = sigma_left if t < mu_T else sigma_right
       However, np.where is not differentiable at the boundary, but BFGS uses 
       finite differences or gradient approximations. It might struggle with 
       the kink.
       
       Better asymmetric form:
       R_T(t) = amp_T * exp(-0.5 * ((t - mu_T) / sigma_T)^2) * (1 + skew_T * (t - mu_T))
       But skew can make it negative.
       
       Let's stick to a robust symmetric Gaussian for T and pH first, but allow 
       the "sharpness" to be fitted via the sigma. The previous MoE used fixed 
       sigmas. Here, sigmas are fitted.
       
       Wait, the inspiration mentioned "Skewed Gaussian terms allow asymmetric 
       temperature and pH response curves".
       
       Let's use a simple skew normal PDF form (simplified):
       f(x) = exp(-0.5 * z^2) / (1 + exp(alpha * z)) ? No, too complex.
       
       Let's use the "Two-sided Gamma" or just parameterize the log of the 
       response as a quadratic, but allow the quadratic to be different on 
       left/right?
       
       Actually, a very effective trick for BFGS is to use:
       R(x) = A * exp(-0.5 * ((x - mu) / sigma)^2)
       And rely on the multiplicative structure to handle interactions.
       
       Let's refine the parameter mapping for stability:
       mu_T = p[0]
       log_sigma_T = p[1]  -> sigma_T = exp(p[1]) (ensures positive)
       log_amp_T = p[2]    -> amp_T = exp(p[2])
       
       mu_pH = p[3]
       log_sigma_pH = p[4] -> sigma_pH = exp(p[4])
       log_amp_pH = p[5]   -> amp_pH = exp(p[5])
       
       For Substrate (s): Monod-like is best.
       R_S(s) = (s / (K_s + s)) * amp_S
       But K_s and amp_S need to be positive.
       log_K_s = p[6] -> K_s = exp(p[6])
       log_amp_S = p[7] -> amp_S = exp(p[7])
       
       For Biomass (b): Usually inhibitory at high density.
       R_B(b) = exp(-k_b * b)
       k_b = p[8] (can be negative? No, inhibition is positive k. But if b is low, effect is small.)
       Let's use R_B(b) = 1 + p[8] * (b - 1) ? No, multiplicative is better.
       R_B(b) = exp(p[8] * (b - 1))
       
       Global Bias: p[9]
       
       Prediction = bias + amp_T * amp_pH * amp_S * R_T_norm * R_pH_norm * R_S_norm * R_B_norm
       
       Wait, the inputs are b, s, temp, pH.
       Ranges:
       b: [1, 4.998]
       s: [1, 4.999]
       temp: [1, 39.98]
       pH: [2, 12]
       
       Normalization is crucial for BFGS.
       Let's normalize inputs to [0, 1] range for the model internals, but 
       parameters like mu_T should be in the normalized space.
       
       b_norm = (b - 1) / 4
       s_norm = (s - 1) / 4
       t_norm = (temp - 1) / 39
       ph_norm = (pH - 2) / 10
       
       Parameters (10):
       p[0]: mu_T_norm
       p[1]: log_sigma_T
       p[2]: log_amp_T
       
       p[3]: mu_pH_norm
       p[4]: log_sigma_pH
       p[5]: log_amp_pH
       
       p[6]: log_K_s_norm
       p[7]: log_amp_S
       
       p[8]: slope_B (linear effect of biomass, additive or multiplicative?)
       Let's use multiplicative: exp(p[8] * b_norm)
       
       p[9]: bias
       
       Model:
       R_T = exp(-0.5 * ((t_norm - p[0]) / exp(p[1]))^2)
       R_pH = exp(-0.5 * ((ph_norm - p[3]) / exp(p[4]))^2)
       
       # Monod for S
       K_s = exp(p[6])
       amp_S = exp(p[7])
       R_S = (s_norm / (K_s + s_norm)) * amp_S
       
       # Biomass
       R_B = exp(p[8] * b_norm)
       
       # Product
       growth = p[9] + exp(p[2]) * exp(p[5]) * R_T * R_pH * R_S * R_B
       
       This uses 10 params.
       It is structurally different from MoE (multiplicative vs additive mixture).
       It is structurally different from Polynomial (mechanistic vs basis expansion).
       It allows for sharp thresholds via small sigma (exp(p[1])).
       
       Potential issue: BFGS might struggle with the product of exponentials if 
       parameters are poorly scaled. The log-params help.
       
       One refinement: The "Skewed" aspect.
       The prompt mentions skewed Gaussians improved score significantly.
       Can we implement a skew term cheaply?
       R_T = exp(-0.5 * z^2) where z = (t - mu) / sigma
       Skew factor: * (1 + skew * z) ?
       If we add a skew param, we run out of params.
       
       Maybe we can use the "difference" in sigmas?
       We have 10 params.
       If we use 4 for T (mu, sig_L, sig_R, amp), 4 for pH (mu, sig_L, sig_R, amp), 
       1 for S (amp), 1 for Bias. That's 10.
       But S needs a K_m too?
       
       Let's try:
       T: mu (p0), sig_left (p1), sig_right (p2), amp (p3)
       pH: mu (p4), sig_left (p5), sig_right (p6), amp (p7)
       S: amp (p8) - Assume K_s is fixed or S is linear?
       Bias: p9
       
       If S is linear: R_S = 1 + p[8]*s_norm? No, S should saturate.
       If we drop K_s for S, we might lose accuracy.
       
       Alternative:
       T: mu (p0), sig (p1), amp (p2)
       pH: mu (p3), sig (p4), amp (p5)
       S: K_s (p6), amp (p7)
       B: slope (p8)
       Bias: p9
       
       This is the symmetric Gaussian version. The "Skew" insight suggests 
       asymmetry is key.
       
       How to get asymmetry with limited params?
       Use a "Skew Normal" approximation:
       f(x) = 2 * phi(x) * Phi(alpha * x)
       Where phi is Gaussian PDF, Phi is CDF.
       This introduces a skew parameter alpha.
       
       If we use Skew Normal for T and pH, we need:
       T: mu, sigma, alpha, amp (4)
       pH: mu, sigma, alpha, amp (4)
       S: K_s, amp (2)
       Bias: 1
       Total: 11. Too many.
       
       We can drop the Bias (assume it's 0 or included in amps).
       Or share the skew?
       
       Let's try:
       T: mu (p0), sigma (p1), skew (p2), amp (p3)
       pH: mu (p4), sigma (p5), skew (p6), amp (p7)
       S: amp (p8) (Assume linear or fixed K)
       B: slope (p9)
       
       If S is just linear: R_S = 1 + p[8]*s_norm?
       Or R_S = exp(p[8]*s_norm)?
       
       Let's implement the Skew Normal PDF form for T and pH.
       SN(x; mu, sigma, alpha) = (2/sigma) * exp(-0.5 * ((x-mu)/sigma)^2) * Phi(alpha * (x-mu)/sigma)
       
       We need Phi (CDF of standard normal).
       From scipy.stats import norm
       norm.cdf is available.
       
       This is computationally heavier but only 7500 rows, so it's fine.
       
       Parameters:
       p[0]: mu_T
       p[1]: log_sigma_T
       p[2]: alpha_T (skew)
       p[3]: log_amp_T
       
       p[4]: mu_pH
       p[5]: log_sigma_pH
       p[6]: alpha_pH
       p[7]: log_amp_pH
       
       p[8]: slope_B (exp(p[8] * b_norm))
       p[9]: bias
       
       What about S?
       If we ignore S in the multiplicative term, we might miss substrate limitation.
       But S is in the inputs.
       Maybe S is best modeled as a simple linear term added to the bias?
       Or maybe S is part of the "B" term?
       
       Let's look at the inputs again: b, s, temp, pH.
       If we use 8 params for T and pH (skewed), we have 2 left.
       One for S, one for B/Bias.
       
       Let's make S a simple linear multiplier: R_S = 1 + p[8] * s_norm
       And B a simple linear multiplier: R_B = 1 + p[9] * b_norm
       But then where is the global bias?
       Maybe the bias is embedded in the amps?
       
       Prediction = (1 + p[8]*s_norm) * (1 + p[9]*b_norm) * Amp_T * R_T_skew * Amp_pH * R_pH_skew
       
       This assumes the base growth rate is determined by T and pH amps, and S and B 
       modulate it linearly.
       
       This uses 10 params.
       
       Let's code this.
    """
    from scipy.stats import norm
    
    # Normalization
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    temp_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    # Extract parameters
    # Temperature (Skew Normal)
    mu_T = params[0]
    sigma_T = np.exp(params[1])
    alpha_T = params[2]
    amp_T = np.exp(params[3])
    
    # pH (Skew Normal)
    mu_pH = params[4]
    sigma_pH = np.exp(params[5])
    alpha_pH = params[6]
    amp_pH = np.exp(params[7])
    
    # Substrate (Linear modulation)
    slope_S = params[8]
    
    # Biomass (Linear modulation)
    slope_B = params[9]
    
    # Calculate Skew Normal PDF components
    # z_T = (temp_norm - mu_T) / sigma_T
    z_T = (temp_norm - mu_T) / sigma_T
    # PDF part: exp(-0.5 * z^2)
    # CDF part: Phi(alpha * z)
    # Note: The standard SN PDF is 2 * phi(z) * Phi(alpha*z)
    # We will use the unnormalized form and let amp_T scale it.
    # Actually, using the full PDF ensures the peak height is related to alpha.
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    # We can drop the constant 1/sqrt(2*pi) and let amp_T absorb it.
    # So R_T = exp(-0.5 * z_T^2) * Phi(alpha_T * z_T)
    
    R_T = np.exp(-0.5 * z_T ** 2) * norm.cdf(alpha_T * z_T)
    
    z_pH = (ph_norm - mu_pH) / sigma_pH
    R_pH = np.exp(-0.5 * z_pH ** 2) * norm.cdf(alpha_pH * z_pH)
    
    # Modulators
    # Ensure positive modulators?
    # If slope is negative and large, 1 + slope*x can be negative.
    # Use exp(slope * x) for guaranteed positivity?
    # If we use exp, the effect is multiplicative and smooth.
    R_S = np.exp(slope_S * s_norm)
    R_B = np.exp(slope_B * b_norm)
    
    # Combine
    prediction = amp_T * amp_pH * R_T * R_pH * R_S * R_B
    
    return prediction
# EVOLVE-BLOCK-END
