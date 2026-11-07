import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Flexible additive model using softplus-based basis functions.
    This allows the optimizer to fit arbitrary non-linear responses for each input
    without being constrained to specific physical shapes like Monod or Gaussian.
    """
    # Unpack parameters
    # 10 params: 
    # p0: bias
    # p1-p3: weights for b (3 basis centers/widths or similar)
    # Actually, let's use a simpler but flexible structure:
    # y = p0 + p1 * f_b(b) + p2 * f_s(s) + p3 * f_t(temp) + p4 * f_pH(pH) + p5 * f_bs(s*b) + p6 * f_st(temp*s) + p7 * f_bp(b*pH) + p8 * f_ts(temp*pH) + p9 * f_all(b*s*temp*pH)
    # Where f is a softplus-like smooth rectifier or similar.
    
    # Let's try a "Neural" style: sum of weighted sigmoid/softplus activations of linear combinations?
    # No, that's too complex for 10 params.
    
    # Better strategy: 
    # Use a sum of Gaussian-like bumps for each variable, but with 2 bumps per variable? 
    # That uses too many params.
    
    # Alternative: 
    # y = p0 + p1 * b + p2 * b^2 + p3 * s + p4 * s^2 + p5 * temp + p6 * temp^2 + p7 * pH + p8 * pH^2 + p9 * (s * temp)
    # This is a polynomial model. It's very flexible and easy to fit.
    # But the current model is multiplicative. A polynomial might capture the interactions better if the true function is smooth.
    # However, the task says "Symbolic regression". The current score is 0.86.
    
    # Let's look at the inspiration 2: "softplus-based smooth additive model".
    # Let's implement a model where each input passes through a softplus non-linearity with a learned offset and scale, then summed.
    # y = p0 + p1 * softplus(b * p2 + p3) + p4 * softplus(s * p5 + p6) + p7 * softplus(temp * p8 + p9) ... wait, I only have 10 params.
    
    # Let's allocate:
    # p0: bias
    # p1, p2: scale, offset for b
    # p3, p4: scale, offset for s
    # p5, p6: scale, offset for temp
    # p7, p8: scale, offset for pH
    # p9: interaction term weight? Or just sum of 4 softplus terms.
    
    # Model:
    # y = p0 + p1 * softplus(b * p2 + p3) + p4 * softplus(s * p5 + p6) + p7 * softplus(temp * p8 + p9)
    # This leaves pH out! I need to include pH.
    
    # Revised Allocation:
    # p0: bias
    # p1: weight for b term
    # p2: weight for s term
    # p3: weight for temp term
    # p4: weight for pH term
    # p5: scale for b
    # p6: scale for s
    # p7: scale for temp
    # p8: scale for pH
    # p9: global interaction multiplier? Or offset?
    
    # Let's try:
    # y = p0 + p1 * softplus(b * p5) + p2 * softplus(s * p6) + p3 * softplus(temp * p7) + p4 * softplus(pH * p8)
    # This is 9 params. p9 can be a global scale or an interaction.
    # Let's add a cross term: p9 * (softplus(b*p5) * softplus(s*p6))
    
    p0 = params[0]
    w_b = params[1]
    w_s = params[2]
    w_t = params[3]
    w_p = params[4]
    s_b = params[5]
    s_s = params[6]
    s_t = params[7]
    s_p = params[8]
    w_inter = params[9]
    
    # Softplus: log(1 + exp(x))
    # To avoid overflow, use np.logaddexp(0, x)
    
    term_b = np.logaddexp(0.0, b * s_b)
    term_s = np.logaddexp(0.0, s * s_s)
    term_t = np.logaddexp(0.0, temp * s_t)
    term_p = np.logaddexp(0.0, pH * s_p)
    
    # Additive sum
    y = p0 + w_b * term_b + w_s * term_s + w_t * term_t + w_p * term_p
    
    # Interaction: Substrate and Temperature often interact strongly
    y += w_inter * (term_s * term_t)
    
    return y
