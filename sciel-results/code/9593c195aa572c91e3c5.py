import numpy as np
import time

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using an adaptive Gaussian Basis Function model.
    
    Unlike the previous RBF approach with fixed centers, this model treats 
    the centers of the Gaussian basis functions as learnable parameters.
    
    Parameters (10 total):
    - params[0:3]: Centers for temperature (temp_c1, temp_c2, temp_c3)
    - params[3:6]: Centers for pH (ph_c1, ph_c2, ph_c3)
    - params[6:9]: Weights for the three basis functions (w1, w2, w3)
    - params[9]:   Global bias/intercept
    
    The model computes a weighted sum of 3 Gaussian kernels based on temperature 
    and pH. The inputs b (biomass) and s (substrate) are assumed to have a 
    multiplicative effect on the growth rate, scaled by a fixed factor to 
    maintain stability, or more simply, we focus on the environmental drivers 
    (T, pH) which are likely the primary determinants of the *maximum* growth 
    rate, while b and s modulate the actual rate. Given the limited parameters, 
    we model the core environmental response surface and use b, s to scale 
    the result, assuming a simple saturation or linear scaling effect captured 
    implicitly by the fit or fixed. 
    
    To be safe and flexible, we use a simple interaction:
    y = (w1*G1 + w2*G2 + w3*G3 + bias) * (1 + 0.1*b_norm) * (1 + 0.1*s_norm)
    
    Wait, using b and s in a multiplicative way with fixed coefficients might 
    be too rigid. Let's stick to the core environmental fit but include b and s 
    as inputs to the Gaussian if we had more params. With only 10 params, 
    fitting 3 centers (6 params) + 3 weights (3 params) + 1 bias (1 param) 
    leaves no params for b/s interactions.
    
    However, the inputs b and s are provided. If we ignore them, we assume 
    they don't affect the *shape* of the T/pH response, only the magnitude. 
    But the task says "equation for E. coli growth rate". Usually, growth rate 
    depends on nutrients. 
    Let's try a different structural change: 
    Use a **Piecewise Linear** or **Spline** approximation? No, Gaussians are smoother.
    
    Let's refine the adaptive Gaussian idea. We have 4 inputs. 
    If we use 2D Gaussians (T, pH), we need 2 centers per kernel. 
    3 kernels -> 6 center params. 
    3 weights -> 3 weight params. 
    1 bias -> 1 param. 
    Total 10. 
    This ignores b and s. 
    Is that acceptable? The inputs b and s are in [1, 5] and [1, 5] (or similar small ranges). 
    If the growth rate is primarily driven by T and pH, and b/s are just initial conditions 
    or concentrations that are saturated, ignoring them might be a good bet. 
    But if they matter, we fail.
    
    Alternative: Use 1D Gaussians for each of the 4 inputs? 
    4 centers (4 params) + 4 weights (4 params) + 1 bias (1 param) = 9 params. 
    This assumes additivity: y = sum(wi * exp(-(xi-ci)^2)) + bias. 
    This is a very flexible non-linear model in each dimension independently. 
    It captures non-linearities in T, pH, b, s independently. 
    It misses interactions, but with 10 params, it's a strong baseline. 
    The previous RBF used 4D centers which is very rigid. 
    An additive model with 1D Gaussians is more flexible in terms of local adaptation 
    because each center can move independently along its axis.
    
    Let's implement the Additive 1D Gaussian Model.
    params[0:4]: Centers for b, s, temp, pH
    params[4:8]: Weights for b, s, temp, pH
    params[8]:   Bias
    params[9]:   Global scale (to handle magnitude)
    
    y = scale * (w_b*G(b, c_b) + w_s*G(s, c_s) + w_t*G(t, c_t) + w_p*G(p, c_p) + bias)
    
    Actually, let's just use:
    y = w_b*G(b, c_b) + w_s*G(s, c_s) + w_t*G(t, c_t) + w_p*G(p, c_p) + bias
    This uses 9 params. The 10th can be a global multiplier or just left as a second bias term? 
    No, let's use params[9] as a second bias or scale. 
    Let's use:
    y = w_b*G(b, c_b) + w_s*G(s, c_s) + w_t*G(t, c_t) + w_p*G(p, c_p) + b0 + b1
    That's redundant.
    
    Let's use params[9] as a scale factor for the entire sum? 
    Or better, let's use 2 Gaussians for Temperature (the most critical non-linear factor) 
    and 1 for pH, and linear for b, s? 
    No, let's stick to the **Additive 1D Gaussian** with 4 terms. 
    Params: 
    0: c_b
    1: w_b
    2: c_s
    3: w_s
    4: c_t
    5: w_t
    6: c_p
    7: w_p
    8: bias
    9: scale (multiplies the sum of the 4 Gaussian terms, not the bias)
    
    y = scale * (w_b*exp(-(b-c_b)^2) + w_s*exp(-(s-c_s)^2) + w_t*exp(-(t-c_t)^2) + w_p*exp(-(p-c_p)^2)) + bias
    
    This allows the optimizer to find the optimal center and weight for each input dimension 
    independently. This is structurally different from the fixed 4D RBF.
    """
    # Unpack parameters
    c_b = params[0]
    w_b = params[1]
    c_s = params[2]
    w_s = params[3]
    c_t = params[4]
    w_t = params[5]
    c_p = params[6]
    w_p = params[7]
    bias = params[8]
    scale = params[9]
    
    # Compute Gaussian terms
    # Using a fixed width sigma=1.0 in raw units? 
    # Ranges: b,s ~ [1,5] (width 4). T ~ [1,40] (width 39). pH ~ [2,12] (width 10).
    # A fixed sigma in raw units might be bad because T has a much larger range.
    # We should normalize inputs first.
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    p_norm = (pH - 2.0) / 10.0
    
    # Centers are in normalized space [0, 1]
    # We need to normalize the center params too? 
    # If params[0] is c_b, it should be in [0,1]. 
    # BFGS starts at 1.0. If c_b=1.0, that's the right end. 
    # This is fine, BFGS will move it.
    
    g_b = np.exp(-(b_norm - c_b)**2)
    g_s = np.exp(-(s_norm - c_s)**2)
    g_t = np.exp(-(t_norm - c_t)**2)
    g_p = np.exp(-(p_norm - c_p)**2)
    
    y = scale * (w_b * g_b + w_s * g_s + w_t * g_t + w_p * g_p) + bias
    
    return y
