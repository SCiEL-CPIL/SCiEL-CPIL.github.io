"""
Strategy: Rational Function (Padé-like) Ansatz with Yield Shift and Quadratic Temperature Scaling.

This approach replaces the sum-of-sigmoids with a single rational function 
$\frac{N(\epsilon)}{D(\epsilon)}$ which is known to approximate stress-strain 
curves (linear elastic to saturation) very efficiently with fewer parameters. 
The yield point is handled by shifting the strain variable. 
Temperature effects are modeled as quadratic scaling and offset, allowing 
for more complex thermal softening/hardening behaviors than linear terms.

Parameters:
p[0]: Elastic modulus (slope at origin)
p[1]: Numerator curvature
p[2]: Denominator linear coeff
p[3]: Denominator quadratic coeff
p[4]: Yield strain shift
p[5]: Temp scale linear
p[6]: Temp scale quadratic
p[7]: Temp offset linear
p[8]: Temp offset quadratic
p[9]: Temp offset cubic
"""
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    p = params
    
    # Extract parameters
    E = p[0]
    B_num = p[1]
    C_den = p[2]
    D_den = p[3]
    eps_y = p[4]
    
    T_s1 = p[5]
    T_s2 = p[6]
    T_o1 = p[7]
    T_o2 = p[8]
    T_o3 = p[9]
    
    # Shift strain to center on yield
    eps_s = eps - eps_y
    
    # Rational function numerator and denominator
    # N = E * eps_s + B_num * eps_s^2
    # D = 1 + C_den * eps_s + D_den * eps_s^2
    num = E * eps_s + B_num * eps_s**2
    den = 1.0 + C_den * eps_s + D_den * eps_s**2
    
    # Safety guard for denominator
    # If denominator is very small or non-finite, replace with small epsilon
    # We use a sign-preserving clip to avoid discontinuities in sign if possible,
    # but given the range, simple absolute value floor is safer for stability.
    # However, if den is negative, the stress might flip. 
    # We assume the optimizer will keep den positive in the domain.
    # To prevent 1e30 error from inf, we clip den.
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8 * np.sign(den), den)
    # If sign is 0, use 1e-8
    den_safe = np.where(den == 0, 1e-8, den_safe)
    
    rational_part = num / den_safe
    
    # Temperature scaling and offset
    # Scale: 1 + T_s1 * T + T_s2 * T^2
    temp_scale = 1.0 + T_s1 * T + T_s2 * T**2
    # Offset: T_o1 * T + T_o2 * T^2 + T_o3 * T^3
    temp_offset = T_o1 * T + T_o2 * T**2 + T_o3 * T**3
    
    sigma = temp_scale * rational_part + temp_offset
    
    # Final safety net for any remaining non-finite values
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    return sigma
