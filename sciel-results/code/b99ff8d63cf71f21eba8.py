import numpy as np
import time
from scipy.optimize import minimize

def _softplus(x):
    """Numerically stable softplus: log(1+exp(x)) without overflow."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

def _sigmoid_grad(x):
    """Gradient of sigmoid."""
    s = _sigmoid(x)
    return s * (1.0 - s)

def equation(b, s, temp, pH, params):
    p = params
    
    # Parameters with stable softplus
    K_s      = _softplus(p[0]) + 1e-6
    T_opt    = p[1]
    T_width  = _softplus(p[2]) + 1e-3
    pH_opt   = p[3]
    pH_width = _softplus(p[4]) + 1e-3
    mu_max   = _softplus(p[5])
    b_slope  = p[6]
    b_shift  = p[7]
    mu_base  = p[8]
    s_inh    = _softplus(p[9]) + 1e-6  # Substrate inhibition strength (Haldane)
    
    # Normalizations to keep parameters O(1) for BFGS starting at 1
    # b: [1, 5] -> [0, 1]
    bn = (b - 1.0) / 4.0
    
    # Temperature: [1, 40]. Optimal ~37.
    tn = (temp - 20.0) / 10.0
    phn = (pH - 7.0) / 5.0
    
    # Monod with Haldane Inhibition
    sfac = s / (K_s + s + (s**2) / (s_inh + 1e-9))
    
    # Temperature Bell
    tfac = np.exp(-0.5 * ((tn - T_opt) / T_width)**2)
    
    # pH Bell
    pfac = np.exp(-0.5 * ((phn - pH_opt) / pH_width)**2)
    
    # Density Inhibition
    arg = np.clip(b_slope * (bn - b_shift), -50, 50)
    dfac = _sigmoid(-arg)
    
    return mu_max * sfac * tfac * pfac * dfac + mu_base

def _objective_and_grad(params, b, s, temp, pH, y):
    """MSE objective and analytical gradient for fitting."""
    p = params
    n = len(y)
    
    # Parameters with stable softplus
    K_s      = _softplus(p[0]) + 1e-6
    T_opt    = p[1]
    T_width  = _softplus(p[2]) + 1e-3
    pH_opt   = p[3]
    pH_width = _softplus(p[4]) + 1e-3
    mu_max   = _softplus(p[5])
    b_slope  = p[6]
    b_shift  = p[7]
    mu_base  = p[8]
    s_inh    = _softplus(p[9]) + 1e-6
    
    # Normalizations
    bn = (b - 1.0) / 4.0
    tn = (temp - 20.0) / 10.0
    phn = (pH - 7.0) / 5.0
    
    # Monod with Haldane Inhibition
    denom = K_s + s + (s**2) / (s_inh + 1e-9)
    sfac = s / denom
    
    # Temperature Bell
    tdiff = (tn - T_opt) / T_width
    tfac = np.exp(-0.5 * tdiff**2)
    
    # pH Bell
    pdiff = (phn - pH_opt) / pH_width
    pfac = np.exp(-0.5 * pdiff**2)
    
    # Density Inhibition
    arg = np.clip(b_slope * (bn - b_shift), -50, 50)
    dfac = _sigmoid(-arg)
    dfac_grad = _sigmoid_grad(-arg)  # d/d(arg) of sigmoid(-arg) = -sigmoid_grad(-arg)
    # Actually: d/dx sigmoid(-x) = -sigmoid_grad(-x)
    dfac_darg = -dfac_grad
    
    pred = mu_max * sfac * tfac * pfac * dfac + mu_base
    
    if not np.all(np.isfinite(pred)):
        return 1e30, np.zeros(10)
    
    # MSE
    mse = np.mean((pred - y) ** 2)
    
    # Gradient: d(MSE)/d(params) = (2/n) * sum((pred - y) * d(pred)/d(params))
    diff = pred - y
    factor = 2.0 / n
    
    # d(pred)/d(mu_max) = sfac * tfac * pfac * dfac
    d_pred_d_mu_max = sfac * tfac * pfac * dfac
    
    # d(pred)/d(mu_base) = 1
    d_pred_d_mu_base = np.ones_like(pred)
    
    # d(pred)/d(K_s) = mu_max * tfac * pfac * dfac * d(sfac)/d(K_s)
    # d(sfac)/d(K_s) = -s / (denom^2)
    d_sfac_d_K_s = -s / (denom**2)
    d_pred_d_K_s = mu_max * tfac * pfac * dfac * d_sfac_d_K_s
    
    # d(pred)/d(T_opt) = mu_max * sfac * pfac * dfac * d(tfac)/d(T_opt)
    # d(tfac)/d(T_opt) = tfac * (tn - T_opt) / (T_width^2)
    d_tfac_d_T_opt = tfac * (tn - T_opt) / (T_width**2)
    d_pred_d_T_opt = mu_max * sfac * pfac * dfac * d_tfac_d_T_opt
    
    # d(pred)/d(T_width) = mu_max * sfac * pfac * dfac * d(tfac)/d(T_width)
    # d(tfac)/d(T_width) = tfac * (tn - T_opt)^2 / (T_width^3)
    d_tfac_d_T_width = tfac * (tn - T_opt)**2 / (T_width**3)
    d_pred_d_T_width = mu_max * sfac * pfac * dfac * d_tfac_d_T_width
    
    # d(pred)/d(pH_opt) = mu_max * sfac * tfac * dfac * d(pfac)/d(pH_opt)
    d_pfac_d_pH_opt = pfac * (phn - pH_opt) / (pH_width**2)
    d_pred_d_pH_opt = mu_max * sfac * tfac * dfac * d_pfac_d_pH_opt
    
    # d(pred)/d(pH_width) = mu_max * sfac * tfac * dfac * d(pfac)/d(pH_width)
    d_pfac_d_pH_width = pfac * (phn - pH_opt)**2 / (pH_width**3)
    d_pred_d_pH_width = mu_max * sfac * tfac * dfac * d_pfac_d_pH_width
    
    # d(pred)/d(b_slope) = mu_max * sfac * tfac * pfac * d(dfac)/d(b_slope)
    # d(dfac)/d(b_slope) = dfac_darg * d(arg)/d(b_slope) = dfac_darg * (bn - b_shift)
    d_dfac_d_b_slope = dfac_darg * (bn - b_shift)
    d_pred_d_b_slope = mu_max * sfac * tfac * pfac * d_dfac_d_b_slope
    
    # d(pred)/d(b_shift) = mu_max * sfac * tfac * pfac * d(dfac)/d(b_shift)
    # d(dfac)/d(b_shift) = dfac_darg * d(arg)/d(b_shift) = dfac_darg * (-b_slope)
    d_dfac_d_b_shift = dfac_darg * (-b_slope)
    d_pred_d_b_shift = mu_max * sfac * tfac * pfac * d_dfac_d_b_shift
    
    # d(pred)/d(s_inh) = mu_max * tfac * pfac * dfac * d(sfac)/d(s_inh)
    # d(sfac)/d(s_inh) = s * s^2 / (s_inh + 1e-9)^2 / denom^2
    d_sfac_d_s_inh = s * (s**2) / ((s_inh + 1e-9)**2 * denom**2)
    d_pred_d_s_inh = mu_max * tfac * pfac * dfac * d_sfac_d_s_inh
    
    # Now compute gradients for each parameter (accounting for softplus where needed)
    grad = np.zeros(10)
    
    # p[0] = K_s_input, K_s = softplus(p[0]) + 1e-6
    # d(K_s)/d(p[0]) = sigmoid(p[0])
    grad[0] = factor * np.sum(diff * d_pred_d_K_s * _sigmoid(p[0]))
    
    # p[1] = T_opt (direct)
    grad[1] = factor * np.sum(diff * d_pred_d_T_opt)
    
    # p[2] = T_width_input, T_width = softplus(p[2]) + 1e-3
    grad[2] = factor * np.sum(diff * d_pred_d_T_width * _sigmoid(p[2]))
    
    # p[3] = pH_opt (direct)
    grad[3] = factor * np.sum(diff * d_pred_d_pH_opt)
    
    # p[4] = pH_width_input, pH_width = softplus(p[4]) + 1e-3
    grad[4] = factor * np.sum(diff * d_pred_d_pH_width * _sigmoid(p[4]))
    
    # p[5] = mu_max_input, mu_max = softplus(p[5])
    grad[5] = factor * np.sum(diff * d_pred_d_mu_max * _sigmoid(p[5]))
    
    # p[6] = b_slope (direct)
    grad[6] = factor * np.sum(diff * d_pred_d_b_slope)
    
    # p[7] = b_shift (direct)
    grad[7] = factor * np.sum(diff * d_pred_d_b_shift)
    
    # p[8] = mu_base (direct)
    grad[8] = factor * np.sum(diff * d_pred_d_mu_base)
    
    # p[9] = s_inh_input, s_inh = softplus(p[9]) + 1e-6
    grad[9] = factor * np.sum(diff * d_pred_d_s_inh * _sigmoid(p[9]))
    
    # Ensure gradient is finite
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(10)
    
    return mse, grad

def _objective(params, b, s, temp, pH, y):
    """MSE objective for fitting."""
    pred = equation(b, s, temp, pH, params)
    if not np.all(np.isfinite(pred)):
        return 1e30
    return np.mean((pred - y) ** 2)

def fit_params(b, s, temp, pH, y, deadline_margin=5.0):
    """Fit parameters with BFGS using analytical gradient + L-BFGS-B polish."""
    start_time = time.time()
    deadline = 150.0 - deadline_margin
    
    # Data-driven initialization
    b_mean = np.mean(b)
    s_mean = np.mean(s)
    temp_mean = np.mean(temp)
    pH_mean = np.mean(pH)
    y_max = np.max(y)
    y_min = np.min(y)
    
    # Initialize parameters based on data ranges
    # K_s ~ geometric mean of s
    K_s_init = np.log(np.sqrt(s_mean * 10)) if s_mean > 0 else 1.0
    # T_opt ~ temp_mean normalized
    T_opt_init = (temp_mean - 20.0) / 10.0
    # T_width ~ std of temp / 10
    T_width_init = np.log(np.std(temp) / 10.0 + 1e-3) if np.std(temp) > 0 else 1.0
    # pH_opt ~ pH_mean normalized
    pH_opt_init = (pH_mean - 7.0) / 5.0
    # pH_width ~ std of pH / 5
    pH_width_init = np.log(np.std(pH) / 5.0 + 1e-3) if np.std(pH) > 0 else 1.0
    # mu_max ~ y_max
    mu_max_init = np.log(y_max + 1e-6) if y_max > 0 else 1.0
    # b_slope ~ moderate
    b_slope_init = 2.0
    # b_shift ~ b_mean normalized
    b_shift_init = (b_mean - 1.0) / 4.0
    # mu_base ~ y_min
    mu_base_init = y_min
    # s_inh ~ large (weak inhibition)
    s_inh_init = np.log(s_mean * 100 + 1) if s_mean > 0 else 5.0
    
    x0 = np.array([K_s_init, T_opt_init, T_width_init, pH_opt_init, pH_width_init,
                   mu_max_init, b_slope_init, b_shift_init, mu_base_init, s_inh_init])
    
    # BFGS with analytical gradient
    res = minimize(_objective_and_grad, x0, args=(b, s, temp, pH, y), 
                   method='BFGS', jac=True,
                   options={'maxiter': 200, 'gtol': 1e-6})
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return res.x
    
    # Final high-precision polish with L-BFGS-B
    bounds = [
        (-10, 10),   # K_s (softplus input)
        (-2, 4),     # T_opt (normalized)
        (-5, 10),    # T_width (softplus input)
        (-2, 2),     # pH_opt (normalized)
        (-5, 10),    # pH_width (softplus input)
        (-5, 10),    # mu_max (softplus input)
        (-20, 20),   # b_slope
        (-1, 2),     # b_shift (normalized)
        (-1, 2),     # mu_base
        (-5, 15),    # s_inh (softplus input)
    ]
    
    res2 = minimize(_objective, res.x, args=(b, s, temp, pH, y), 
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
    
    # Check deadline again
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return res.x
    
    # Use the better result
    if res2.fun < res.fun:
        return res2.x
    return res.x
