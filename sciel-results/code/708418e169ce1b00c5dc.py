import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod.
    Uses a smooth piecewise model with elastic and plastic regions,
    temperature-dependent parameters, and polynomial corrections.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    strain = np.asarray(strain, dtype=np.float64)
    temp = np.asarray(temp, dtype=np.float64)
    
    # Temperature-dependent elastic modulus
    E = p0 + p1 * temp
    
    # Temperature-dependent yield strain
    yield_strain = p2 + p3 * temp
    
    # Temperature-dependent yield stress
    sigma_y = p4 + p5 * temp
    
    # Temperature-dependent hardening modulus
    K = p6 + p7 * temp
    
    # Smooth transition sharpness
    k = np.abs(p8) + 1e-10
    
    # Smooth hinge function: approximates max(x, 0) but is differentiable
    # Using: x * (1 - exp(-k*x)) for x > 0, which is smooth and grows linearly for large x
    # For x < 0, this gives a small negative value, which we clip to 0
    x = strain - yield_strain
    # Use a smooth approximation to max(x, 0)
    # softplus-like: log(1 + exp(k*x)) / k, but this can overflow
    # Better: use x * sigmoid(k*x) which is smooth and approximates max(x,0)
    # Actually, let's use: for x >= 0, x; for x < 0, x * exp(k*x) (smooth at 0)
    # This is: x * (1 - 0.5*(1 - tanh(k*x))) = x * 0.5*(1 + tanh(k*x))
    # But this gives 0 at x=0, not x at x=0. We want continuity.
    
    # Let's use: plastic_strain_smooth = x * sigmoid(k*x)
    # At x=0: 0. For large positive x: x. For large negative x: ~0.
    # This is smooth and differentiable everywhere.
    sigmoid_kx = 1.0 / (1.0 + np.exp(-k * x))
    plastic_strain_smooth = x * sigmoid_kx
    
    # Ensure non-negative for physical meaning, but keep smoothness
    # Actually, x * sigmoid(k*x) is already >= 0 for all x when k > 0
    # Let's verify: if x < 0, sigmoid(k*x) < 0.5, so x * sigmoid(k*x) > 0 (negative * negative)
    # Wait, that's not right. If x < 0, sigmoid(k*x) is between 0 and 0.5, so x * sigmoid(k*x) is negative.
    # We want max(x, 0) behavior. Let's use:
    # smooth_max(x, 0) = 0.5 * (x + sqrt(x^2 + eps)) where eps = 1/k^2
    # This is smooth and approximates max(x, 0)
    eps = 1.0 / (k * k)
    smooth_max_x_0 = 0.5 * (x + np.sqrt(x * x + eps))
    
    # Plastic stress contribution
    plastic_stress = K * smooth_max_x_0
    
    # Total stress: elastic + plastic
    # In elastic region (x < 0): stress = E * strain
    # In plastic region (x > 0): stress = E * strain + K * (strain - yield_strain)
    # But we want a smooth transition. The elastic part E*strain is always present.
    # The plastic part is added smoothly.
    stress = E * strain + plastic_stress
    
    # Add polynomial corrections for better fit
    # These can capture any systematic deviations from the simple model
    stress += p8 * strain * temp + p9 * strain**2
    
    return stress
