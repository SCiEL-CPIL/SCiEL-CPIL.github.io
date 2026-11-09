import numpy as np
import time

def equation(b, s, temp, pH, params):
    """
    Hybrid model: multiplicative Monod-Gaussian-Gaussian core (biological prior)
    plus a smooth additive correction with RBF bases for each variable and
    key interactions. The multiplicative core captures the expected shape;
    the additive RBF correction lets BFGS fit residual structure without
    destabilizing the core.
    
    Params (10 total):
        p[0]: T_center (tanh -> [0,1])
        p[1]: T_width (tanh -> [0.05, 0.45])
        p[2]: pH_center (tanh -> [0,1])
        p[3]: pH_width (tanh -> [0.05, 0.45])
        p[4]: K_s (exp, positive)
        p[5]: mu_max (exp, positive)
        p[6]: w_s (RBF weight for substrate)
        p[7]: w_b (RBF weight for density)
        p[8]: w_TpH (RBF weight for T-pH interaction)
        p[9]: offset
    """
    p = params
    
    # --- Core multiplicative model parameters ---
    T_c = 0.5 * (1.0 + np.tanh(p[0]))
    T_w = 0.05 + 0.40 * (0.5 * (1.0 + np.tanh(p[1])))
    pH_c = 0.5 * (1.0 + np.tanh(p[2]))
    pH_w = 0.05 + 0.40 * (0.5 * (1.0 + np.tanh(p[3])))
    K_s = np.exp(p[4])
    mu_max = np.exp(p[5])
    
    # --- Additive correction weights ---
    w_s = p[6]
    w_b = p[7]
    w_TpH = p[8]
    offset = p[9]
    
    # Normalize inputs to [0, 1] based on known training ranges
    xb = (b - 1.0) / 4.0
    xs = (s - 1.0) / 39.0
    xT = (temp - 1.0) / 39.0
    xPH = (pH - 2.0) / 10.0
    
    # --- Core multiplicative components ---
    # Monod substrate
    sfac = s / (K_s + s + 1e-9)
    
    # Temperature Gaussian bell
    tfac = np.exp(-0.5 * ((xT - T_c) / T_w)**2)
    
    # pH Gaussian bell
    pfac = np.exp(-0.5 * ((xPH - pH_c) / pH_w)**2)
    
    # Core prediction
    base = mu_max * sfac * tfac * pfac
    
    # --- Additive RBF correction ---
    # Fixed-width RBFs for substrate and density (centers at 0.5 for simplicity,
    # giving a smooth bump that can correct for non-Monod substrate effects
    # and density-dependent growth inhibition)
    rbf_width = 0.35
    
    # Substrate RBF (centered at normalized 0.5)
    rbf_s = np.exp(-0.5 * ((xs - 0.5) / rbf_width)**2)
    
    # Density RBF (centered at normalized 0.5)
    rbf_b = np.exp(-0.5 * ((xb - 0.5) / rbf_width)**2)
    
    # T-pH interaction RBF (2D Gaussian centered at fitted T_c, pH_c)
    rbf_TpH = np.exp(-0.5 * (((xT - T_c) / rbf_width)**2 + ((xPH - pH_c) / rbf_width)**2))
    
    # Combine: core + additive corrections
    result = base + w_s * rbf_s + w_b * rbf_b + w_TpH * rbf_TpH + offset
    
    return result
