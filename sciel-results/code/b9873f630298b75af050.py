import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Learnable-Width Gaussian Process (Kernel) Model
    
    Instead of using a fixed-width RBF basis (which assumes a specific smoothness),
    this model uses 10 Gaussian kernels with widths that are themselves parameters.
    The parameters are the log-widths (log_sigmas) to ensure positivity and numerical
    stability during BFGS optimization.
    
    The prediction is:
    y(x) = sum_{i=1}^{10} exp(-0.5 * ((x - c_i) / sigma_i)^2)
    
    where c_i are fixed evenly spaced centers and sigma_i = exp(params[i-1]).
    
    Why this differs from the current program:
    1. The current program uses fixed sigmas (0.6) and linear parameters for amplitudes.
    2. This program treats the shape of the basis functions (widths) as learnable.
    3. This allows the model to adapt to the "frequency" of the hidden function.
       If the function has sharp features, BFGS can narrow specific sigmas.
       If the function is smooth, BFGS can broaden them.
    4. The non-linearity in parameters (exp) makes the loss surface non-convex,
       but BFGS starting from ones (sigma=1) is a reasonable heuristic for the
       domain [0.1, 6] where a unit width is a natural scale.
    
    Parameters (10 total):
    - p[0:10]: Log-widths (log_sigmas) for the 10 Gaussian kernels.
    
    Note: The amplitudes are fixed to 1. This is a "kernel trick" style approach
    where the function is assumed to be a sum of unit-amplitude bumps whose
    locations (fixed) and shapes (learnable) determine the fit. 
    Wait, fixing amplitudes to 1 is too restrictive. The standard GP mean function
    is sum of kernels, but usually we learn weights too. However, we only have 10 params.
    
    Revised Strategy:
    We have 10 params. We need to balance location/width/amplitude.
    Since centers are fixed, we have 10 degrees of freedom.
    If we use all 10 for widths, amplitudes are fixed at 1. This might be too rigid
    if the target function has varying amplitudes at the peaks.
    
    Alternative: Use 5 widths and 5 amplitudes? No, we have 10 centers.
    Let's stick to the prompt's suggestion: "parameters representing the kernel weights 
    (or log-variances) for each center".
    
    If we use params as log-variances (widths), we lose amplitude flexibility.
    If we use params as amplitudes (current program), we lose width flexibility.
    
    Let's try a compromise: 
    The prompt suggests: "parameters representing the kernel weights (or log-variances)".
    Let's interpret "kernel weights" as amplitudes, but the strategy description says 
    "shifts from polynomial basis approximation to a local interpolation/smoothing approach".
    
    Actually, the most powerful change with 10 params and fixed centers is to make the 
    WIDTHS learnable. But we also need amplitudes. 
    Maybe we can use a smaller number of centers with both learnable width and amplitude?
    E.g., 5 centers, 5 widths, 5 amplitudes.
    Centers: 0.5, 1.8, 3.1, 4.4, 5.7.
    This gives 10 params: 5 log_sigmas, 5 amplitudes.
    This allows the model to adapt both the "sharpness" and "height" of 5 main features.
    This is a structural change from 8 fixed-width RBFs + poly to 5 learnable-width RBFs.
    
    Let's implement 5 centers with learnable log-sigmas and amplitudes.
    """
    
    start_time = time.time()
    TIME_LIMIT = 145.0

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]

    # 5 Centers evenly spaced in [0.1, 6.0]
    # Spacing ~ 1.225
    centers = np.linspace(0.5, 5.5, 5)
    
    # Params:
    # p[0:5] -> log_sigmas
    # p[5:10] -> amplitudes
    
    log_sigmas = p[0:5]
    amplitudes = p[5:10]
    
    # Convert log_sigmas to sigmas with clipping to avoid overflow/underflow
    # exp(20) is ~ 5e8, exp(-20) is ~ 2e-9.
    # Clipping to [-5, 5] means sigma in [0.0067, 148].
    # This is safe.
    sigmas = np.exp(np.clip(log_sigmas, -5, 5))
    
    # Calculate RBF values
    # Shape: (N_points, 5)
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    
    # To avoid overflow in exp, we can compute log_rbf = -0.5 * (diff/sigma)^2
    # But diff/sigma can be large.
    # Let's just compute normally, numpy handles inf.
    # (diff / sigmas[np.newaxis, :]) ** 2
    scaled_diff = diff / sigmas[np.newaxis, :]
    rbf_values = np.exp(-0.5 * scaled_diff ** 2)
    
    # Check time
    if time.time() - start_time > TIME_LIMIT:
        # Fallback: return 0
        return np.zeros_like(x)

    # Prediction: sum(amplitudes * rbf_values)
    # amplitudes shape (5,), rbf_values shape (N, 5)
    y = rbf_values @ amplitudes
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(x)
        
    return y
