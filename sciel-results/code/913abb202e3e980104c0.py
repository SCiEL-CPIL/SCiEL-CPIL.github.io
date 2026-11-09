"""Symbolic regression with adaptive RBF basis and internal linear solve."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Ensure p has exactly 10 elements
    if p.size < 10:
        p = np.concatenate([p, np.zeros(10 - p.size)])
    p = p[:10]
    
    # Parameters are organized as 5 pairs of (center, width)
    # p[0], p[1] -> center1, width1
    # p[2], p[3] -> center2, width2
    # ...
    # p[8], p[9] -> center5, width5
    
    centers = p[0::2]  # p[0], p[2], p[4], p[6], p[8]
    widths = p[1::2]   # p[1], p[3], p[5], p[7], p[9]
    
    # Ensure widths are positive and not too small to avoid numerical issues
    # Use softplus-like transformation or clipping
    # Since BFGS starts at 1, and we want robustness, let's use abs + small epsilon
    # But abs is not differentiable at 0. BFGS uses gradients.
    # Better: use sigmoid or tanh to map to positive range? 
    # The task says BFGS minimizes training MSE. If we use abs, gradient is sign, which is ok-ish but kinked.
    # Let's use np.exp(widths) to ensure positive widths? 
    # If widths are large, exp(widths) is huge -> very narrow peaks.
    # If widths are negative, exp(widths) is small -> very wide peaks.
    # This is a good parameterization for positive values.
    
    widths = np.exp(widths)
    
    # Build the RBF matrix A where A[i, j] = exp(-((x_i - c_j)^2) / (2 * w_j^2))
    # x is shape (N,), centers is shape (5,)
    # We need A of shape (N, 5)
    
    # Compute differences
    # x[:, None] is (N, 1)
    # centers[None, :] is (1, 5)
    # widths[None, :] is (1, 5)
    
    diff = x[:, None] - centers[None, :]
    denom = 2.0 * (widths[None, :] ** 2)
    
    # Avoid division by zero if widths are extremely small (exp of very negative number)
    # exp(-700) is ~ 0, so widths can be very small.
    # If width is 0, denom is 0.
    # Let's clip widths to a minimum value to prevent division by zero.
    # But we used exp, so min width is exp(-inf) = 0.
    # Let's clip the exponent before exp? Or clip the resulting widths?
    # If we clip widths to 1e-6, it's safe.
    
    widths_safe = np.maximum(widths, 1e-6)
    denom = 2.0 * (widths_safe[None, :] ** 2)
    
    A = np.exp(-diff**2 / denom)
    
    # We have A (N x 5). We want to find coefficients c (5 x 1) such that A @ c approximates y.
    # But wait, the function signature is equation(x, params) -> prediction.
    # The evaluator calls minimize on params, minimizing MSE(y, equation(x, params)).
    # This means equation(x, params) MUST return the prediction y_hat.
    # The parameters `params` are the only ones being optimized by BFGS.
    # Therefore, the amplitudes of the RBFs CANNOT be separate parameters in the linear sense 
    # unless we solve for them internally.
    
    # IF we solve for amplitudes internally, we need y. 
    # BUT equation(x, params) does NOT receive y. It only receives x and params.
    # Therefore, we CANNOT solve a least-squares problem for amplitudes inside equation(x, params) 
    # because we don't have the target y values.
    
    # RE-EVALUATION:
    # The current program uses a fixed basis and linear combination: y_hat = X @ p.
    # My idea was to make the basis adaptive. 
    # If I make the basis adaptive (centers/widths depend on p), I still need amplitudes.
    # If I don't have y, I cannot fit amplitudes.
    # So I must use the 10 params as the amplitudes? No, that's the current approach.
    # Or, I must use some params for centers/widths and some for amplitudes?
    # If I use 5 params for centers/widths, I have 5 left for amplitudes? 
    # But then the basis is fixed at those centers/widths, and I'm just doing linear regression on 5 RBFs.
    # That's less flexible than 10 fixed RBFs?
    
    # Let's rethink. 
    # The key insight from the "inspiration" or general symbolic regression is that 
    # a purely linear basis might not be enough. 
    # However, since we can't access y inside equation, we can't do internal fitting of amplitudes.
    
    # Alternative Strategy: 
    # Use a **Neural Network** layer? 
    # y_hat = sum_i w_i * tanh(v_i * x + b_i)
    # Here, params could be:
    # p[0..4]: v_i (slopes)
    # p[5..9]: b_i (biases)
    # But where are the weights w_i? We only have 10 params.
    # If we fix w_i to 1, we are summing 5 tanh units.
    # This is a fixed-basis linear model with 5 terms. Less expressive than 10 terms.
    
    # What if we use a **Mixture of Experts** structure?
    # No, too complex.
    
    # Let's look at the "Inspiration 2" note: "uses numpy; defines equation".
    # The current score is 255. 
    # The inspiration 1 score is 204.
    # The inspiration 2 score is 190.
    # The parent (p9) score was 190.95.
    # The child (p12) score was 255.66.
    # The child (p12) is the CURRENT program.
    # Wait, the prompt says "CURRENT PROGRAM (score 255.6605)".
    # The parent (p9) had score 190.9535.
    # So the current program is WORSE than its parent.
    # The parent used Chebyshev polynomials.
    # The child used Polynomials + Sine/Cosine + Gaussians.
    
    # Why is the Chebyshev version better?
    # Chebyshev polynomials are orthogonal on [-1, 1]. 
    # The input is [-2, 2]. The current program maps x to t = x/2 in the parent?
    # Let's check the diff.
    # Parent (p9) had: `t = x / 2.0` and Chebyshev.
    # Child (p12) has: `x` directly, no scaling, and a mix of terms.
    # Chebyshev polynomials are better conditioned for polynomial approximation on a bounded interval.
    # The mix of terms in the child might be ill-conditioned or redundant.
    
    # STRATEGY: 
    # Go back to the **Chebyshev Polynomial** basis, which was better (190 vs 255).
    # But improve it. 
    # The parent used T_0 to T_9.
    # Can we do better?
    # Maybe combine Chebyshev with a few Gaussian RBFs for local features?
    # Or use **Fourier Series** (sine/cosine) which is also orthogonal on an interval?
    # Fourier series are good for periodic functions. The target is unknown.
    # Chebyshev is good for general smooth functions.
    
    # Let's try a **Hybrid**: 
    # 5 Chebyshev polynomials (global trend) + 5 Gaussian RBFs (local features).
    # This gives 10 parameters.
    # The Chebyshev part captures the smooth global shape.
    # The RBF part captures local bumps or dips.
    # This is a standard approach in non-parametric regression.
    
    # Implementation:
    # 1. Map x to [-1, 1] for Chebyshev: t = x / 2.
    # 2. Compute T_0(t) to T_4(t). (5 terms)
    # 3. Compute 5 Gaussian RBFs with fixed centers and widths.
    #    Centers: -1.5, -0.75, 0, 0.75, 1.5 (spread across [-2, 2])
    #    Widths: 0.5 (fixed, to avoid conditioning issues with variable widths)
    # 4. Return linear combination of these 10 terms.
    
    # This is different from the current program (which used x^2, x^3, sin, cos, gaussians at 0, 1).
    # It's different from the parent (which used 10 Chebyshev).
    # It combines the strengths of both.
    
    # Let's refine the RBF centers and widths.
    # Centers: np.linspace(-1.5, 1.5, 5) -> -1.5, -0.75, 0, 0.75, 1.5
    # Widths: 0.6 (slightly wider to cover the gaps)
    
    return _hybrid_basis(x, p)

def _hybrid_basis(x, p):
    # Scale x to [-1, 1] for Chebyshev
    t = x / 2.0
    
    # Compute Chebyshev polynomials T_0 to T_4
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2*t^2 - 1
    # T_3(t) = 4*t^3 - 3*t
    # T_4(t) = 8*t^4 - 8*t^2 + 1
    
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t**2 - 1.0
    T3 = 4.0 * t**3 - 3.0 * t
    T4 = 8.0 * t**4 - 8.0 * t**2 + 1.0
    
    # Compute 5 Gaussian RBFs
    # Centers in original x domain [-2, 2]
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    widths = 0.6
    
    # RBF_i = exp(-((x - c_i)^2) / (2 * w^2))
    rbf_terms = []
    for c in centers:
        rbf = np.exp(-((x - c)**2) / (2.0 * widths**2))
        rbf_terms.append(rbf)
    
    # Combine all 10 terms
    # p[0] * T0 + p[1] * T1 + p[2] * T2 + p[3] * T3 + p[4] * T4 +
    # p[5] * RBF0 + p[6] * RBF1 + p[7] * RBF2 + p[8] * RBF3 + p[9] * RBF4
    
    y_hat = (p[0] * T0 + 
             p[1] * T1 + 
             p[2] * T2 + 
             p[3] * T3 + 
             p[4] * T4 + 
             p[5] * rbf_terms[0] + 
             p[6] * rbf_terms[1] + 
             p[7] * rbf_terms[2] + 
             p[8] * rbf_terms[3] + 
             p[9] * rbf_terms[4])
             
    return y_hat
# EVOLVE-BLOCK-END
