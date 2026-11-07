import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid strategy:
    1. The model is a sum of 2 decaying sinusoids:
       y = A1 * exp(-a1*x) * sin(w1*x + p1) + A2 * exp(-a2*x) * sin(w2*x + p2)
       + c1 + c2*x + c3*x^2 + c4*x^3
    2. The evaluator's BFGS is a single-start local optimizer. To handle non-convexity
       (frequencies/phases), we perform a deterministic multi-start local optimization
       inside this function.
    3. We use a fixed time budget (100s) to run several local minimizations from
       different starting points and return the parameters that yield the lowest
       training MSE.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p_init = np.asarray(params, dtype=np.float64)
    if p_init.ndim != 1:
        p_init = p_init.flatten()
        
    # Ensure we have 10 params
    if len(p_init) < 10:
        p_init = np.pad(p_init, (0, 10 - len(p_init)))
    else:
        p_init = p_init[:10]

    # Target: 10 params
    # Indices:
    # 0: A1, 1: a1, 2: w1, 3: p1 (phase)
    # 4: A2, 5: a2, 6: w2, 7: p2 (phase)
    # 8: c0 (offset), 9: c1 (linear slope)
    # Note: I dropped the quadratic/cubic terms to keep the search space manageable
    # and because decaying sinusoids + linear offset is a very strong basis for
    # many smooth functions in [0, 6]. If the target is a polynomial, the linear
    # term helps, but the sinusoids can approximate curvature if needed (though
    # poorly). Let's stick to 2 sinusoids + 2 linear coeffs.
    
    # Actually, let's make it slightly more general:
    # y = A1 * exp(-a1*x) * cos(w1*x + p1) + A2 * exp(-a2*x) * cos(w2*x + p2) + c0 + c1*x
    # Using cos is equivalent to sin with a phase shift.
    
    def model(x, p):
        A1, a1, w1, ph1 = p[0], p[1], p[2], p[3]
        A2, a2, w2, ph2 = p[4], p[5], p[6], p[7]
        c0, c1 = p[8], p[9]
        
        # Ensure stability: if a1 or a2 are very negative, exp blows up.
        # But BFGS/optimization should keep them reasonable.
        # We clip a1, a2 to [-10, 10] to prevent overflow during evaluation.
        a1_c = np.clip(a1, -10, 10)
        a2_c = np.clip(a2, -10, 10)
        
        term1 = A1 * np.exp(-a1_c * x) * np.cos(w1 * x + ph1)
        term2 = A2 * np.exp(-a2_c * x) * np.cos(w2 * x + ph2)
        
        return term1 + term2 + c0 + c1 * x

    # We need y_train to calculate MSE. But we don't have y_train!
    # Wait, the evaluator calls equation(x, params). It does NOT pass y.
    # Therefore, I CANNOT compute the training MSE inside equation() to guide
    # a multi-start search. I only have x.
    
    # RE-EVALUATION:
    # The task says: "The evaluator fits params... minimizing the training MSE."
    # This implies the evaluator has the data. My function `equation` is just the
    # forward pass. I cannot access y.
    # Therefore, I CANNOT perform internal optimization based on error.
    # I must rely on the evaluator's BFGS.
    
    # So, the "multi-start" idea is invalid because I don't have y.
    # I must return a function that is:
    # 1. Easy for BFGS to optimize (smooth, not too many local minima).
    # 2. Capable of representing the target.
    
    # The current RBF approach is smooth but has fixed widths.
    # The "Inspiration" mentions that non-linear parametric models (exp decay, sine)
    # allowed exact representation of wider classes.
    
    # Strategy: Use a "Rich" Parametric Model that is still smooth for BFGS.
    # Let's try a sum of 3 decaying sinusoids + linear.
    # To help BFGS, we should use parameters that are well-scaled.
    # Frequencies can be tricky. Let's constrain them implicitly or use a basis
    # that is less prone to local minima.
    
    # Alternative: A Chebyshev polynomial of high degree?
    # No, polynomials oscillate (Runge's phenomenon) at edges if degree is high,
    # but for degree 9, it might be okay on [0.1, 6].
    # However, the current RBF is score 0.5. Polynomials often struggle with
    # non-polynomial shapes.
    
    # Let's look at the "Inspiration 2" score: 8087. That was bad.
    # "Inspiration 3" score: 2774. Also bad.
    # Current RBF: 0.5. This is good.
    
    # The key is that the target might be a specific smooth function.
    # If the target is e.g. sin(x) or exp(-x), the RBF approximates it well.
    # But if the target is exactly representable by a parametric form,
    # a parametric form with enough flexibility will fit it perfectly (MSE ~ 0).
    
    # Let's try a model that can represent:
    # 1. Polynomials (via low frequency sinusoids approximating lines/curves? No.)
    # 2. Exponentials
    # 3. Sinusoids
    
    # A sum of exponentials: sum( A_i * exp(-a_i * x) )
    # This is great for decaying functions, bad for oscillatory ones.
    
    # A sum of sinusoids: sum( A_i * sin(w_i * x + p_i) )
    # This is great for oscillatory, bad for non-periodic.
    
    # Hybrid: sum( A_i * exp(-a_i * x) * sin(w_i * x + p_i) )
    # This covers a huge class.
    
    # The problem with BFGS on this is the local minima of frequency/phase.
    # However, if we initialize parameters well, it might work.
    # But I can't control the initialization (it's all ones).
    
    # Is there a way to make the function "convex" or "easier" for BFGS?
    # What if I use a fixed set of basis functions that are orthogonal or well-conditioned?
    # The RBF does this.
    
    # Let's try a different basis: **Fourier Series** with fixed frequencies.
    # y = sum_{k=1}^{5} ( a_k * cos(k*x) + b_k * sin(k*x) ) + c0
    # This requires 11 params. We have 10.
    # Let's use 4 harmonics:
    # y = a1*cos(x) + b1*sin(x) + a2*cos(2x) + b2*sin(2x) + a3*cos(3x) + b3*sin(3x) + c0 + c1*x + c2*x^2
    # This is a linear model in parameters!
    # BFGS on a linear model is just linear regression. It will converge perfectly
    # (if the matrix is well-conditioned).
    # The risk is if the target function is not well-approximated by this basis.
    # But a Fourier-like basis + polynomial tail is a very strong universal approximator
    # for smooth functions on a bounded interval.
    
    # Why did the current RBF get 0.5?
    # RBF is also linear in parameters (amplitudes).
    # So both are linear-in-parameters models.
    # The difference is the basis functions.
    # RBF: Localized.
    # Fourier: Global oscillatory.
    
    # If the target is smooth and non-oscillatory, RBF might be better.
    # If the target is oscillatory, Fourier is better.
    
    # Let's try a **Hybrid Basis**:
    # 3 RBFs (localized) + 3 Fourier terms (global) + 2 Polynomial terms.
    # Total 8 basis functions. We have 10 params.
    # We can add 2 more RBFs or Fourier terms.
    
    # Let's construct:
    # 4 RBFs with fixed centers: 1.0, 2.5, 4.0, 5.5
    # 2 Cosine terms: cos(2*pi*x/6), cos(4*pi*x/6)
    # 2 Sine terms: sin(2*pi*x/6), sin(4*pi*x/6)
    # 1 Constant
    # 1 Linear
    # Total: 4 + 2 + 2 + 1 + 1 = 10 parameters.
    
    # This model is linear in params. BFGS will solve it exactly (up to numerical precision).
    # The quality depends entirely on the basis.
    
    def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64).flatten()
        if len(p) < 10:
            p = np.pad(p, (0, 10 - len(p)))
        else:
            p = p[:10]
            
        # Basis 1: 4 RBFs
        centers = np.array([0.5, 2.0, 3.5, 5.0])
        sigma = 1.0
        diff = x[:, np.newaxis] - centers[np.newaxis, :]
        rbf = np.exp(-0.5 * (diff / sigma) ** 2)
        
        # Basis 2: 2 Cos, 2 Sin
        # Fundamental frequency for range ~6: 2*pi/6 = pi/3
        w1 = np.pi / 3.0
        w2 = 2 * np.pi / 3.0
        
        cos1 = np.cos(w1 * x)
        sin1 = np.sin(w1 * x)
        cos2 = np.cos(w2 * x)
        sin2 = np.sin(w2 * x)
        
        # Basis 3: Poly
        const = np.ones_like(x)
        lin = x
        
        # Combine
        # p[0:4] -> RBF amplitudes
        # p[4] -> cos1 amp
        # p[5] -> sin1 amp
        # p[6] -> cos2 amp
        # p[7] -> sin2 amp
        # p[8] -> const
        # p[9] -> lin
        
        y = (rbf @ p[0:4] + 
             p[4] * cos1 + 
             p[5] * sin1 + 
             p[6] * cos2 + 
             p[7] * sin2 + 
             p[8] * const + 
             p[9] * lin)
             
        return y

    # Wait, I defined equation inside equation? No, I should just define the logic directly.
    # I will re-write the clean version.
    
    pass # Placeholder, I will output the clean code below.
    
# Actual implementation

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        x = x.flatten()
        
    p = np.asarray(params, dtype=np.float64).flatten()
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
    else:
        p = p[:10]
        
    # Hybrid Basis:
    # 4 RBFs (fixed centers, fixed width)
    # 2 Cos, 2 Sin (fixed frequencies)
    # 1 Constant
    # 1 Linear
    
    # RBF Centers: Spread across [0.1, 6.0]
    centers = np.array([0.5, 2.0, 3.5, 5.0])
    sigma = 1.0
    diff = x[:, np.newaxis] - centers[np.newaxis, :]
    rbf = np.exp(-0.5 * (diff / sigma) ** 2)
    
    # Fourier terms
    # Domain ~ 6.0. Period ~ 6.0 -> w = 2*pi/6 = pi/3
    w1 = np.pi / 3.0
    w2 = 2 * np.pi / 3.0
    
    cos1 = np.cos(w1 * x)
    sin1 = np.sin(w1 * x)
    cos2 = np.cos(w2 * x)
    sin2 = np.sin(w2 * x)
    
    const = np.ones_like(x)
    lin = x
    
    y = (rbf @ p[0:4] + 
         p[4] * cos1 + 
         p[5] * sin1 + 
         p[6] * cos2 + 
         p[7] * sin2 + 
         p[8] * const + 
         p[9] * lin)
         
    return y
