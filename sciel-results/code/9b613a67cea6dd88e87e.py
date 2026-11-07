"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock. Keep a safety margin.
_DEADLINE_SECONDS = 145.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local Basis Function Model (RBF/Gaussian Kernels).
    
    Instead of a global functional form (polynomial/trig), we model the target
    as a weighted sum of 5 Gaussian kernels centered at 5 points in the input
    space (t, x, v).
    
    Parameters (10 total):
    - 5 centers: (t_i, x_i, v_i) for i=0..4
    - 5 widths:  sigma_i for i=0..4
    
    The evaluator's BFGS optimizer will fit these parameters.
    To ensure the model can represent the data, we implicitly assume the weights
    are 1 (or scaled by the width to maintain energy), but this is a rigid 
    assumption. 
    
    Wait, if weights are fixed to 1, the model is very constrained. 
    A better structural change: Use the 10 params to define a 
    "Spline-like" or "Piecewise" structure? No, BFGS needs differentiability.
    
    Let's use a **Mixture of Experts** style or just a **High-Order Neural Net** 
    with fixed weights? No, weights must be fitted.
    
    Let's stick to the RBF idea but make the "weights" part of the parameters?
    We only have 10 params. 
    If we use 5 RBFs, we need 5 centers * 3 dims = 15 params. Too many.
    
    Alternative Structural Change: **Fourier Series with Harmonic Progression**.
    The current program uses 2 sin terms. A true oscillator response often 
    contains harmonics. 
    Let's use 10 parameters to define a Fourier series with 5 harmonics 
    (sine and cosine for each).
    Params:
    p0: Frequency base f0
    p1-p9: Amplitudes for sin(f0*t), cos(f0*t), sin(2*f0*t), cos(2*f0*t), ...
    
    This is still a global model.
    
    Let's try a **Polynomial in x and v with time-modulated coefficients**.
    Model: y = A(t) * x + B(t) * x^2 + C(t) * v + D(t) * v^2 ...
    Where A(t) = p0 * sin(p1*t + p2) ...
    This is getting complex.
    
    Let's look at the "Inspiration 3" (score 51.5) which used a "Non-linear 
    parametric model for a driven Duffing/Van der Pol oscillator". 
    The current score is 1.28. The seed is likely much lower. 
    The current program (1.28) uses 2 sin + 2 power terms.
    
    A MATERIALLY DIFFERENT approach: **Neural Network with Fixed Architecture, Fitted Weights**.
    We can implement a small, deep neural network (e.g., 3 layers) where the 
    10 parameters are the weights of the final layer (or a subset of weights) 
    while the rest are fixed? No, BFGS fits ALL 10 params.
    
    If we define a function with 10 free parameters that form a universal 
    approximator locally:
    
    **Strategy: Scaled Gaussian Radial Basis Functions (RBF) with Shared Width**
    
    We have 10 params.
    Let's define 2 RBFs in the (x, v) plane, modulated by time?
    
    Actually, the most robust "structural" change for symbolic regression 
    tasks where the true form is unknown is often a **Taylor Series expansion 
    around multiple points** or a **Lagrange Interpolation** style basis, 
    but those are hard to fit with BFGS from scratch.
    
    Let's try **Chebyshev Polynomials**. 
    Chebyshev polynomials are orthogonal and provide a stable basis for 
    approximating smooth functions on a bounded interval.
    Inputs t, x, v are in known ranges:
    t: [30, 50]
    x: [-0.2056, 0.1749]
    v: [-0.3391, 0.3624]
    
    We can map these to [-1, 1] and use a multivariate Chebyshev basis.
    With 10 parameters, we can afford a moderate number of terms.
    For example:
    1. Constant
    2. T1(t)
    3. T1(x)
    4. T1(v)
    5. T2(t)
    6. T2(x)
    7. T2(v)
    8. T1(t)*T1(x)
    9. T1(t)*T1(v)
    10. T1(x)*T1(v)
    
    This is a linear model in the parameters, but the basis functions are 
    Chebyshev polynomials of the mapped inputs. This is structurally different 
    from the current "2 sin + 2 power" model because:
    1. It handles the bounded nature of the inputs explicitly via mapping.
    2. Chebyshev polynomials minimize the maximum error (minimax property) 
       for polynomial approximations, which can be more efficient than 
       power series or simple trigonometric terms for bounded domains.
    3. It includes interaction terms (cross-terms) which the current model 
       lacks (current model is sum of univariate terms).
    
    Let's implement this.
    """
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    # Map inputs to [-1, 1] for Chebyshev stability
    # t range: [30, 50]
    t_min, t_max = 30.0, 50.0
    # x range: [-0.2056, 0.1749]
    x_min, x_max = -0.2056, 0.1749
    # v range: [-0.3391, 0.3624]
    v_min, v_max = -0.3391, 0.3624

    # Clip to range to avoid extrapolation issues (though training is in-range)
    t_c = np.clip(t, t_min, t_max)
    x_c = np.clip(x, x_min, x_max)
    v_c = np.clip(v, v_min, v_max)

    # Affine map to [-1, 1]
    # z = 2*(val - min)/(max - min) - 1
    t_z = 2.0 * (t_c - t_min) / (t_max - t_min) - 1.0
    x_z = 2.0 * (x_c - x_min) / (x_max - x_min) - 1.0
    v_z = 2.0 * (v_c - v_min) / (v_max - v_min) - 1.0

    # Clip z to [-1, 1] to ensure Chebyshev definition validity (T_n(cos(theta)))
    t_z = np.clip(t_z, -1.0, 1.0)
    x_z = np.clip(x_z, -1.0, 1.0)
    v_z = np.clip(v_z, -1.0, 1.0)

    # Compute Chebyshev Polynomials of the first kind
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2*z^2 - 1
    # T_3(z) = 4*z^3 - 3*z
    
    T0 = np.ones_like(t_z)
    
    T1_t = t_z
    T1_x = x_z
    T1_v = v_z
    
    T2_t = 2.0 * t_z**2 - 1.0
    T2_x = 2.0 * x_z**2 - 1.0
    T2_v = 2.0 * v_z**2 - 1.0
    
    # Interaction terms (cross-terms)
    T1_tx = t_z * x_z
    T1_tv = t_z * v_z
    T1_xv = x_z * v_z

    # Linear combination with 10 parameters
    # p0: T0 (bias)
    # p1: T1(t)
    # p2: T1(x)
    # p3: T1(v)
    # p4: T2(t)
    # p5: T2(x)
    # p6: T2(v)
    # p7: T1(t)*T1(x)
    # p8: T1(t)*T1(v)
    # p9: T1(x)*T1(v)
    
    out = (
        p[0] * T0 +
        p[1] * T1_t +
        p[2] * T1_x +
        p[3] * T1_v +
        p[4] * T2_t +
        p[5] * T2_x +
        p[6] * T2_v +
        p[7] * T1_tx +
        p[8] * T1_tv +
        p[9] * T1_xv
    )

    # Guarantee finiteness and correct shape
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)
# EVOLVE-BLOCK-END
