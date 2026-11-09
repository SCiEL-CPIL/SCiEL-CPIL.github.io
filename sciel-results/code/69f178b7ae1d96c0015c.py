import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Rational basis:
    # Numerator: p0 + p1*x + p2*y + p3*z + p4*x*y + p5*x*z + p6*y*z
    # Denominator: 1 + p7*x + p8*y + p9*z
    # 
    # This allows modeling hyperbolic, exponential-like, and other non-polynomial behaviors.
    # The "1" in the denominator ensures that at initialization (all p=1), 
    # denom = 1 + x + y + z, which is in [4, 7], so no division by zero.
    
    num = p[0] + p[1]*x + p[2]*y + p[3]*z + p[4]*x*y + p[5]*x*z + p[6]*y*z
    den = 1.0 + p[7]*x + p[8]*y + p[9]*z
    
    # Safety: if den is close to 0, clip it to avoid huge values
    # The inputs are in [0.5, 2]. 
    # If p[7..9] are large, den can be small.
    # We want to avoid non-finite results.
    # If |den| < 1e-6, we set den to 1e-6 * sign(den) or just 1e-6.
    # Actually, if den is 0, the prediction is inf. 
    # Let's clip den to be away from 0.
    # However, if the true function has a pole, we might want to allow it? 
    # But the task says inputs are in [0.5, 2] and targets are noiseless finite values.
    # So the target doesn't have poles in this domain.
    # Thus, the denominator should not be zero for the optimal fit.
    # We just need to avoid numerical instability during optimization.
    
    # Use a safe division
    # np.divide with where and out
    result = np.zeros_like(x)
    mask = np.abs(den) > 1e-8
    np.divide(num, den, out=result, where=mask)
    
    # For points where den is too small, set result to 0 or some large penalty?
    # Better to set to 0 to keep it finite, but this creates a discontinuity.
    # Alternatively, add a small constant to den: den_safe = den + 1e-8 * np.sign(den)
    # But if den is exactly 0, sign is 0.
    # Let's just use a small epsilon in the denominator:
    # den_safe = np.where(np.abs(den) < 1e-8, 1e-8, den)
    
    den_safe = np.where(np.abs(den) < 1e-8, 1e-8, den)
    result = num / den_safe
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result
