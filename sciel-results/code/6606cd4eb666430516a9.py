import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a Multi-Resolution RBF (Radial Basis Function) strategy.
    
    Strategy:
    - Instead of a fixed global basis (polynomials/trig), we use 10 Gaussian "bumps".
    - The first 10 parameters (p[0:10]) define the centers of these Gaussians in x.
    - The next 10 parameters (p[10:20]) define the amplitudes (weights) of these Gaussians.
    - The next 10 parameters (p[20:30]) define the inverse widths (1/sigma) to allow 
      varying resolution (sharp vs smooth).
    - The last 10 parameters (p[30:40]) are a small linear polynomial fallback to ensure 
      global trend capture and numerical stability.
    
    This allows the optimizer to "splat" high-capacity regions where the function is complex 
    and keep flat regions simple, which is often superior to global basis functions for 
    arbitrary smooth targets.
    """
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # Safety margin for the 150s limit
    
    # Ensure x is 1D for safe operations, though evaluator likely passes 1D
    x = np.asarray(x).flatten()
    
    # Extract parameter groups
    # We need to handle the case where params might be shorter or longer, 
    # but the contract says length 10? 
    # Wait, the contract says: "The evaluator fits params, an array of length 10".
    # My current program uses params[0..9]. 
    # If I want to use more parameters, I must pack them into the 10 available slots.
    
    # RE-STRATEGY FOR 10 PARAMS ONLY:
    # Since I only have 10 parameters, I cannot have 10 centers, 10 weights, 10 widths.
    # I must design a compact 10-param RBF or hybrid.
    
    # Let's use 5 RBFs.
    # Params 0-2: Centers for 3 RBFs? No, let's do 2 RBFs + Polynomial.
    # Or, better: Use the 10 params to define a small set of adaptive features.
    
    # Alternative Strategy: Adaptive Frequency Fourier + Polynomial
    # But the directive says "MATERIALLY DIFFERENT". RBF is different.
    
    # Let's pack 2 RBFs and a Cubic Polynomial into 10 params.
    # RBF 1: Center c1, Width w1, Amp a1 (3 params)
    # RBF 2: Center c2, Width w2, Amp a2 (3 params)
    # Poly: a0 + a1*x + a2*x^2 + a3*x^3 (4 params)
    # Total: 10 params.
    
    c1 = params[0]
    w1 = params[1]
    a1 = params[2]
    
    c2 = params[3]
    w2 = params[4]
    a2 = params[5]
    
    p0 = params[6]
    p1 = params[7]
    p2 = params[8]
    p3 = params[9]
    
    # Clip centers to valid range [-2, 2] to avoid overflow in exp if width is small
    # Clip widths to ensure positive and non-zero
    # If width is too small, exp becomes 0 or Inf.
    # We use softplus or clipping to ensure w > 0.
    # Actually, BFGS will optimize raw params. We must map them to valid ranges.
    
    # Map c1, c2 to [-2, 2] using tanh
    c1_mapped = 2.0 * np.tanh(c1)
    c2_mapped = 2.0 * np.tanh(c2)
    
    # Map w1, w2 to [0.1, 10.0] using softplus + offset
    # w = exp(softplus(p)) + 0.1
    # Or simpler: w = abs(p) + 0.1
    # Using abs makes it non-smooth at 0, which can hurt BFGS.
    # Use softplus: log(1 + exp(x))
    def softplus(x):
        return np.logaddexp(0, x)
    
    w1_mapped = softplus(w1) + 0.1
    w2_mapped = softplus(w2) + 0.1
    
    # Amplitudes a1, a2 can be any real number, no mapping needed.
    
    # Compute RBFs
    # Gaussian: exp(-((x - c)^2) / (2 * w^2))
    # To avoid overflow/underflow issues, we keep w reasonable.
    rbf1 = np.exp(-((x - c1_mapped) ** 2) / (2.0 * w1_mapped ** 2))
    rbf2 = np.exp(-((x - c2_mapped) ** 2) / (2.0 * w2_mapped ** 2))
    
    # Compute Polynomial
    # Use normalized x for polynomial to improve conditioning
    xn = x / 2.0
    poly = p0 + p1 * xn + p2 * (xn ** 2) + p3 * (xn ** 3)
    
    # Combine
    result = a1 * rbf1 + a2 * rbf2 + poly
    
    # Check for non-finite values
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
