import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Models acceleration using a rational function approximation (Padé-like form).
    
    y = N(x, v) / D(x, v)
    
    where N is a quadratic polynomial in normalized x and v (6 terms)
    and D is a quadratic polynomial in normalized x and v with constant 1 (4 terms).
    
    Total parameters: 10.
    
    Normalization is applied to improve conditioning for BFGS.
    x_norm = x / 0.6
    v_norm = v / 0.3
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Normalize inputs to roughly [-1, 1] based on training ranges
    # x range: [-0.5728, 0.5275] -> scale ~ 0.6
    # v range: [-0.1754, 0.2821] -> scale ~ 0.3
    x_n = x / 0.6
    v_n = v / 0.3
    
    # Precompute powers
    x2 = x_n * x_n
    v2 = v_n * v_n
    xv = x_n * v_n
    
    # Numerator: quadratic in x_n, v_n
    # p0 + p1*x + p2*v + p3*x^2 + p4*xv + p5*v^2
    numerator = p0 + p1 * x_n + p2 * v_n + p3 * x2 + p4 * xv + p5 * v2
    
    # Denominator: 1 + p6*x + p7*v + p8*x^2 + p9*v^2
    # The constant 1 is fixed to ensure scale invariance and prevent trivial solutions
    # where both num and den scale to 0 or inf.
    denominator = 1.0 + p6 * x_n + p7 * v_n + p8 * x2 + p9 * v2
    
    # Safety check: if denominator is close to zero, add a small epsilon
    # to avoid division by zero during optimization.
    # The training data range is small, so denominator should stay away from 0
    # if params are reasonable.
    # Using a hard floor/clip on denominator magnitude is risky for gradients.
    # Instead, we rely on the optimizer staying in a safe region.
    # However, to prevent catastrophic failure if optimizer explores bad regions,
    # we can add a tiny epsilon to the denominator if it gets too small.
    # A more robust approach for BFGS is to just let it fail if it goes bad,
    # but the task says "non-finite error costs 1e30".
    
    # To be safe against division by zero during the search:
    # If |denom| < 1e-6, we set it to 1e-6 * sign(denom) or just 1e-6.
    # But this creates a discontinuity.
    # Given the small input range, the denominator is likely well-behaved.
    # Let's just compute it directly.
    
    return numerator / denominator
