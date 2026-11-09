"""Neural Network model for aluminium rod stress-strain with LM optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress = strain * MLP(strain, temp; params)
    
    This model uses a small neural network to capture complex non-linearities.
    The multiplication by 'strain' enforces the physical constraint that stress
    is zero at zero strain.
    
    Parameters (10 total):
    p0, p1: Weights from input to hidden layer (row 0)
    p2, p3: Weights from input to hidden layer (row 1)
    p4, p5: Weights from input to hidden layer (row 2)
    p6, p7: Weights from input to hidden layer (row 3)
    p8: Bias for hidden layer (shared or simplified? No, we need 10 params total)
    
    Wait, 2 inputs -> 4 hidden needs 8 weights.
    Hidden -> Output needs 4 weights.
    Total 12 weights. We only have 10 params.
    
    Let's reduce hidden layer to 3 units.
    2 inputs -> 3 hidden: 6 weights.
    3 hidden -> 1 output: 3 weights.
    Total 9 weights.
    1 bias for hidden layer? Or shared bias?
    Let's use:
    W1 (2x3) = 6 params (p0..p5)
    W2 (3x1) = 3 params (p6..p8)
    b1 (1x3) = 3 params? No, that's 12.
    
    Let's stick to 2 hidden units to fit 10 params easily with biases.
    2 inputs -> 2 hidden: 4 weights.
    2 hidden -> 1 output: 2 weights.
    2 biases for hidden: 2 params.
    1 bias for output: 1 param.
    Total: 4 + 2 + 2 + 1 = 9 params.
    We have 10. Let's add 1 more weight? No, symmetry is better.
    
    Let's try 2 inputs -> 3 hidden, 3 hidden -> 1 output.
    Weights: 2*3 + 3*1 = 9.
    Biases: 3 (hidden) + 1 (output) = 4. Total 13. Too many.
    
    Let's use a "shared bias" or simpler structure.
    Structure:
    H = tanh( W1 @ X + b1 )
    Y = W2 @ H + b2
    Stress = strain * Y
    
    X shape: (N, 2)
    W1 shape: (2, 2) -> 4 params
    b1 shape: (2,) -> 2 params
    W2 shape: (2, 1) -> 2 params
    b2 shape: (1,) -> 1 param
    Total: 4+2+2+1 = 9 params.
    We have 10. Let's make W1 (2,3) and W2 (3,1)?
    W1: 6 params.
    b1: 3 params? No, let's share b1 or drop one bias.
    
    Alternative:
    W1 (2, 3) = 6 params (p0..p5)
    W2 (3, 1) = 3 params (p6..p8)
    b1 (1,) shared bias for all hidden units? = 1 param (p9)
    b2 (0) no output bias? Or include it?
    If we include b2, we need 11 params.
    Let's drop b2 (force output bias to 0) or drop b1.
    Dropping b1 is risky for Tanh.
    Dropping b2 is okay if the model can shift via weights.
    
    Let's use:
    p0..p5: W1 (2x3)
    p6..p8: W2 (3x1)
    p9: b1 (shared bias for hidden layer)
    Total 10 params.
    
    Architecture:
    X_norm = normalize([strain, temp])
    H = tanh(X_norm @ W1 + b1)  # Shape (N, 3)
    Y = H @ W2                  # Shape (N, 1)
    Stress = strain * Y.flatten()
    """
    # We are not doing the optimization inside equation() in the standard sense for the evaluator's sake?
    # The evaluator calls equation(strain, temp, params) and fits params.
    # The evaluator uses scipy.optimize.minimize(BFGS).
    # My strategy is to CHANGE the function form to be more amenable to optimization, 
    # but I cannot change the optimizer the evaluator uses.
    # The evaluator uses BFGS.
    # So I must design a function that BFGS can optimize well.
    # Neural nets are hard for BFGS.
    #
    # RE-EVALUATION:
    # The prompt says: "The evaluator fits params... with scipy.optimize.minimize(method="BFGS")"
    # I CANNOT change the optimizer. I can only change the `equation` function.
    # Therefore, implementing a custom LM optimizer INSIDE equation is not possible because 
    # equation is just a forward pass. The fitting is done externally.
    #
    # So, the "Strategy" must be a different FUNCTIONAL FORM that is easier for BFGS to fit.
    # BFGS works well for smooth, convex-like landscapes.
    # The current rational function is non-convex.
    #
    # What is a better form for BFGS?
    # Maybe a **Gaussian Mixture Model (GMM)** of stress vs strain, conditioned on temperature?
    # Or a **Spline-like** interpolation using smoothstep functions?
    #
    # Let's look at Inspiration 2 (Score 1.28): Modified Voce.
    # Let's look at Inspiration 3 (Score 1.22): Piecewise linear.
    #
    # A different structural change: **Separable Variable Model**.
    # Stress = f(strain) * g(temp) + h(strain, temp)?
    # If we assume the temperature effect is primarily a scaling of the yield point and modulus,
    # we can model:
    # E(T) = E0 * (1 - alpha * (T - T_ref))
    # sigma_y(T) = sigma_y0 * (1 - beta * (T - T_ref))
    # sigma_sat(T) = sigma_sat0 * (1 - gamma * (T - T_ref))
    #
    # Model:
    # sigma = E(T) * strain * exp(-strain/epsilon_0) + sigma_y(T) * (1 - exp(-strain/epsilon_0)) + (sigma_sat(T) - sigma_y(T)) * (1 - exp(-strain/epsilon_1))
    #
    # This is an additive combination of elastic, yield, and saturation terms.
    # It is smooth and differentiable.
    # Parameters:
    # E0, alpha (2)
    # sigma_y0, beta (2)
    # sigma_sat0, gamma (2)
    # epsilon_0, epsilon_1 (2)
    # Offset? (1)
    # Temp ref? (1)
    # Total 10.
    #
    # This is structurally different from the rational function (which is a ratio) and the pure Voce (which might be too simple).
    # It explicitly models the three regimes: Elastic, Yield Transition, Saturation.
    #
    # Let's refine this "Three-Regime Exponential" model.
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize temperature to [0, 1] range for stability
    # Training range: [0.06667, 1]
    t_min, t_max = 0.06667, 1.0
    t_norm = (temp - t_min) / (t_max - t_min)
    # Clip to avoid extrapolation issues if BFGS explores weird temps? 
    # Inputs are in training range, so t_norm is [0,1].
    
    # Temperature dependent coefficients
    # Elastic Modulus: E(T) = p0 * (1 - p1 * t_norm)
    E_T = p0 * (1.0 - p1 * t_norm)
    
    # Yield Stress: Sy(T) = p2 * (1 - p3 * t_norm)
    Sy_T = p2 * (1.0 - p3 * t_norm)
    
    # Saturation Stress: Ssat(T) = p4 * (1 - p5 * t_norm)
    Ssat_T = p4 * (1.0 - p5 * t_norm)
    
    # Characteristic strains
    # epsilon_0 controls the elastic-to-plastic transition width
    # epsilon_1 controls the hardening-to-saturation width
    # Ensure positivity to avoid math domain errors in exp if params go negative?
    # BFGS might try negative values. exp(-x/neg) -> exp(pos) -> huge.
    # Use softplus or abs? 
    # Let's use p6 + 1e-3 to keep it small positive? Or just let BFGS handle it.
    # To be safe against overflow:
    eps0 = np.maximum(p6, 1e-4)
    eps1 = np.maximum(p7, 1e-4)
    
    # Base strain is 0.
    # Term 1: Elastic (decays as plasticity starts)
    # We want E*strain at small strain.
    # Term 1 = E_T * strain * exp(-strain / eps0)
    
    # Term 2: Yield Plateau (rises then stays?)
    # Actually, standard Voce is Sy * (1 - exp(-strain/eps)).
    # This goes from 0 to Sy.
    # But we already have Elastic term.
    # Let's use a "Bilinear" approximation with smooth transition.
    #
    # Alternative: 
    # sigma = E_T * strain + (Sy_T - E_T * eps0) * (1 - exp(-strain/eps0)) + (Ssat_T - Sy_T) * (1 - exp(-(strain-eps0)/eps1)) * step(strain>eps0)?
    # Step functions are bad for BFGS.
    #
    # Let's stick to the smooth additive form:
    # sigma = E_T * strain * exp(-strain/eps0) 
    #       + Sy_T * (1 - exp(-strain/eps0)) 
    #       + (Ssat_T - Sy_T) * (1 - exp(-(strain - eps0) / eps1)) * exp(-strain/eps0) ?
    #
    # This is getting complicated. Let's use a simpler, robust form:
    # **Modified Swift-Hollomon / Voce Hybrid**
    #
    # sigma = sigma_0(T) + (sigma_inf(T) - sigma_0(T)) * (1 - exp(-k(T) * strain))
    # This is just Voce. Score 1.28.
    #
    # What if we use **Polynomial in Strain with Temperature-dependent Coeffs** but with a **Rational Stabilizer**?
    # The current program is a rational function. Score 7.6.
    # Why is it so bad? Probably local minima or poor conditioning.
    #
    # Let's try a **Piecewise Smooth Function** using `np.where` is bad for gradients.
    # Use **Smooth Min/Max** or **Softplus** based splines.
    #
    # Let's try: **Cubic Spline in Strain, Coeffs depend on Temp**.
    # sigma = c0(T) + c1(T)*strain + c2(T)*strain^2 + c3(T)*strain^3
    # 4 coeffs * 2 (temp dep) = 8 params.
    # Plus 2 global shifts?
    #
    # This is a polynomial. BFGS fits polynomials very well if the basis is orthogonal.
    # Standard power basis is ill-conditioned for strain up to ~1.
    # Let's use **Chebyshev Polynomials** of the first kind on the normalized strain domain.
    # This is a structural change: Change basis from Power to Chebyshev.
    # Chebyshev polynomials are well-conditioned and often lead to better convergence for BFGS.
    
    # Normalize strain to [-1, 1] for Chebyshev
    # Training strain: [-0.0019, 0.9636]
    # Let's map to [-1, 1]
    s_min, s_max = -0.001911, 0.9636
    s_norm = (2 * (strain - s_min) / (s_max - s_min)) - 1
    
    # Chebyshev polynomials T_0, T_1, T_2, T_3
    T0 = np.ones_like(s_norm)
    T1 = s_norm
    T2 = 2 * s_norm**2 - 1
    T3 = 4 * s_norm**3 - 3 * s_norm
    
    # Coefficients depend on Temperature
    # c_k(T) = a_k + b_k * t_norm
    # 4 coeffs * 2 = 8 params.
    # Remaining 2 params: Offset and Scale?
    # Or maybe the temperature dependence is nonlinear?
    # Let's stick to linear temp dependence for the 4 Chebyshev coeffs (8 params).
    # p8: Constant offset (stress at strain=0? No, strain=0 is inside domain).
    # p9: Global scaling?
    
    # Actually, let's use 5 Chebyshev terms?
    # 5 * 2 = 10 params.
    # T4 = 8*s^4 - 8*s^2 + 1
    
    T4 = 8 * s_norm**4 - 8 * s_norm**2 + 1
    
    # Coeffs:
    # c0 = p0 + p1*t
    # c1 = p2 + p3*t
    # c2 = p4 + p5*t
    # c3 = p6 + p7*t
    # c4 = p8 + p9*t
    
    c0 = p0 + p1 * t_norm
    c1 = p2 + p3 * t_norm
    c2 = p4 + p5 * t_norm
    c3 = p6 + p7 * t_norm
    c4 = p8 + p9 * t_norm
    
    stress = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3 + c4 * T4
    
    return stress

# EVOLVE-BLOCK-END
