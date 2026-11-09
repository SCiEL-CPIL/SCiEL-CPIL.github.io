"""Bootstrap portfolio: model-and-optimize for stress-strain of aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Record the start time to implement a deadline guard.
_START_TIME = time.time()
_TIMEOUT_LIMIT = 140.0

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Stress of an aluminium rod from strain and temperature.
    
    Uses a hybrid model combining a smooth elastic-plastic transition 
    (rational function), a piecewise-linear "zigzag" basis for yield points 
    and work-hardening segments, and a linear temperature-dependent offset.
    
    This structure leverages the finding that piecewise-linear basis functions 
    with parameterized breakpoints (via abs-based rectification) capture 
    discrete yield points and work-hardening segments in aluminium 
    stress-strain curves more effectively than smooth rational/tanh families.
    
    Model:
    stress = E(T)*strain / (1 + H*strain) 
           + A1 * max(0, strain - b1) 
           + A2 * max(0, strain - b2) 
           + A3 * max(0, strain - b3)
           + C(T)
    
    Where:
    E(T) = p0 + p1*temp + p2*temp**2
    H    = p3
    b1   = p4  (first yield point)
    b2   = p5  (second yield point)
    b3   = p6  (third yield point)
    A1   = p7  (slope change at first yield)
    A2   = p8  (slope change at second yield)
    A3   = p9  (slope change at third yield)
    C(T) = 0   (absorbed into E and A terms; or we could add a constant)
    
    Actually, we have 10 parameters. Let's reassign:
    p0, p1, p2: E(T) = p0 + p1*temp + p2*temp**2
    p3: H (hardening modulus for rational term)
    p4, p5, p6: breakpoints b1, b2, b3
    p7, p8, p9: slopes A1, A2, A3
    
    Wait, that's 10 parameters. But we also need an offset. 
    The rational term E*strain/(1+H*strain) is ~ E*strain for small strain.
    If we want an offset, we might need to adjust.
    However, the zigzag terms max(0, strain-b) are zero for strain < b.
    So for strain < min(b1,b2,b3), stress = E(T)*strain/(1+H*strain).
    This passes through the origin if strain=0.
    If the data has an offset, we might need it.
    But we only have 10 params.
    
    Let's check the seed program again.
    Seed: E(T)*strain/(1+H*strain) + S(T)*tanh(K(T)*strain) + B(T)
    Params: p0..p9.
    E(T) = p0+p1T+p2T^2 (3)
    H = p3 (1)
    S(T) = p4+p5T (2)
    K(T) = p6+p7T (2)
    B(T) = p8+p9T (2)
    Total 10.
    
    My proposed model:
    E(T) = p0+p1T+p2T^2 (3)
    H = p3 (1)
    b1 = p4 (1)
    b2 = p5 (1)
    b3 = p6 (1)
    A1 = p7 (1)
    A2 = p8 (1)
    A3 = p9 (1)
    Total 10.
    
    This model lacks an explicit offset B(T).
    However, if the yield points are close to 0, the zigzag terms can act like an offset for large strains.
    But for small strains, it's purely elastic.
    If the data has a non-zero intercept, this model might struggle.
    
    Alternative: Use fewer breakpoints.
    E(T) = p0+p1T+p2T^2 (3)
    H = p3 (1)
    b1 = p4 (1)
    A1 = p5 (1)
    B(T) = p6+p7T (2)
    C = p8 (1) -- constant offset?
    D = p9 (1) -- maybe another breakpoint?
    
    Let's stick to the directive: "piecewise-linear 'sawtooth' or 'zigzag' basis... parameterized breakpoints".
    The knowledge says: "Piecewise-linear basis with parameterized breakpoints via abs-based rectification... Measured: 1 -> 1.228".
    This suggests the zigzag approach is effective.
    
    Let's try to fit the offset into the rational term or assume the offset is small/zero.
    Or, we can make one of the "slopes" actually an offset if the breakpoint is 0?
    max(0, strain - 0) = max(0, strain). This is 0 for negative strain, linear for positive.
    If we set b1=0, then A1*max(0, strain) adds a linear term for positive strain.
    It doesn't add a constant offset.
    
    To get a constant offset, we need a term that is non-zero at strain=0.
    max(0, strain - b) is 0 if strain < b. If b < 0, then for strain=0, max(0, -b) = -b.
    So if we allow negative breakpoints, we can get an offset.
    However, strain range is [-0.001911, 0.9636].
    If b is negative, say -0.01, then at strain=0, term is A*(-0.01) = -0.01A.
    This effectively adds a constant.
    
    So, allowing negative breakpoints allows the model to fit an offset.
    
    Model:
    stress = E(T)*strain / (1 + H*strain) 
           + A1 * max(0, strain - b1) 
           + A2 * max(0, strain - b2) 
           + A3 * max(0, strain - b3)
    
    Parameters:
    p0, p1, p2: E(T)
    p3: H
    p4, p5, p6: b1, b2, b3
    p7, p8, p9: A1, A2, A3
    
    This uses all 10 params.
    
    Implementation details:
    - max(0, x) can be implemented as 0.5 * (x + abs(x)).
    - This is differentiable everywhere (subgradient at 0).
    - BFGS uses numerical gradients, so this is fine.
    
    Deadline guard:
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)
    
    Inputs:
    strain: [-0.001911, 0.9636]
    temp: [0.06667, 1]
    
    Let's write the code.
    """
    if time.time() - _START_TIME > _TIMEOUT_LIMIT:
        return np.zeros_like(strain, dtype=np.float64)

    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Elastic modulus with quadratic temperature dependence
    E_T = p0 + p1 * temp + p2 * temp**2
    
    # Hardening parameter for rational term
    H = p3
    
    # Breakpoints for piecewise-linear terms
    b1 = p4
    b2 = p5
    b3 = p6
    
    # Slopes for piecewise-linear terms
    A1 = p7
    A2 = p8
    A3 = p9
    
    # Rational term: E(T)*strain / (1 + H*strain)
    den = 1.0 + H * strain
    eps = 1e-6
    den_safe = np.where(np.abs(den) < eps, np.sign(den + 1e-300) * eps, den)
    term_rational = (E_T * strain) / den_safe
    
    # Piecewise-linear terms: A * max(0, strain - b)
    # max(0, x) = 0.5 * (x + abs(x))
    term1 = A1 * 0.5 * ((strain - b1) + np.abs(strain - b1))
    term2 = A2 * 0.5 * ((strain - b2) + np.abs(strain - b2))
    term3 = A3 * 0.5 * ((strain - b3) + np.abs(strain - b3))
    
    stress = term_rational + term1 + term2 + term3
    
    # Ensure output is finite
    stress = np.where(np.isfinite(stress), stress, 0.0)
    
    return stress
# EVOLVE-BLOCK-END
