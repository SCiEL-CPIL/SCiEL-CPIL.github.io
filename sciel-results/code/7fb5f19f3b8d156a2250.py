"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline management
_start_time = None
_best_params = None
_best_score = np.inf

def _get_basis(x, y, z, params):
    """
    Construct a non-linear, parameterized basis.
    params: [a, b, c, d, e, f, g, h, i, j]
    
    Strategy: Use a combination of terms that BFGS can optimize non-linearly.
    We use a "soft" product/sum structure to allow for multiplicative interactions
    that are not just simple monomials.
    
    Let's define the prediction as:
    p = w0 + w1*x^a + w2*y^b + w3*z^c + w4*(x*y)^d + w5*(y*z)^e + w6*(z*x)^f + w7*exp(-g*(x+y+z)) + w8*sin(h*(x+y)) + w9*cos(i*z)
    
    This is non-linear in params (a,b,c,d,e,f,g,h,i are exponents/arguments).
    BFGS can handle this.
    """
    # Unpack params
    # We need to be careful with exponents to keep values finite.
    # Clip exponents to reasonable range to avoid overflow/underflow
    a = np.clip(params[0], -2.0, 2.0)
    b = np.clip(params[1], -2.0, 2.0)
    c = np.clip(params[2], -2.0, 2.0)
    d = np.clip(params[3], -2.0, 2.0)
    e = np.clip(params[4], -2.0, 2.0)
    f = np.clip(params[5], -2.0, 2.0)
    g = np.clip(params[6], 0.0, 5.0)
    h = np.clip(params[7], 0.0, 5.0)
    i = np.clip(params[8], 0.0, 5.0)
    j = params[9] # Weight for the last term or constant offset? Let's make it a constant.
    
    # We need 10 params. The above uses 10.
    # Let's refine the structure to ensure it's smooth and differentiable.
    # 
    # Term 1: Constant
    t0 = np.ones_like(x)
    
    # Term 2: x^a
    t1 = np.power(x, a)
    
    # Term 3: y^b
    t2 = np.power(y, b)
    
    # Term 4: z^c
    t3 = np.power(z, c)
    
    # Term 5: (xy)^d
    t4 = np.power(x * y, d)
    
    # Term 6: (yz)^e
    t5 = np.power(y * z, e)
    
    # Term 7: (zx)^f
    t6 = np.power(z * x, f)
    
    # Term 8: exp(-g*(x+y+z))
    t7 = np.exp(-g * (x + y + z))
    
    # Term 9: sin(h*(x+y))
    t8 = np.sin(h * (x + y))
    
    # Term 10: cos(i*z)
    t9 = np.cos(i * z)
    
    # Combine with fixed weights? No, the params ARE the coefficients/exponents.
    # Wait, if params are exponents, where are the weights?
    # The evaluator optimizes `params`. If `params` are exponents, the weights are fixed to 1?
    # That's too restrictive. 
    # 
    # Alternative: Use a small neural network with 1 hidden layer.
    # Input: x, y, z (3 nodes)
    # Hidden: 10 nodes? No, params are length 10.
    # We can't have a full NN with 10 params.
    #
    # Let's go back to the linear basis but make it ADAPTIVE?
    # No, the evaluator fits params. The structure is fixed.
    #
    # The key insight from "Chebyshev basis" (score 7.39) is that better basis = better score.
    # The current program uses a mix of monomials and non-linear terms (sin, cos, exp).
    # Score is 9.63.
    #
    # Let's try a **Rational Function** basis.
    # f(x,y,z) = (p0 + p1*x + p2*y + p3*z + p4*x*y + p5*y*z + p6*z*x) / (1 + p7*x + p8*y + p9*z)
    # This is non-linear in params (denominator).
    # BFGS can fit this. It can represent hyperbolic-like behaviors.
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    num = p0 + p1*x + p2*y + p3*z + p4*x*y + p5*y*z + p6*z*x
    den = 1.0 + p7*x + p8*y + p9*z
    
    # Avoid division by zero
    den = np.where(np.abs(den) < 1e-6, 1e-6 * np.sign(den), den)
    
    pred = num / den
    
    return pred

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Rational Function Model.
    
    The evaluator fits 10 parameters using BFGS.
    We model the function as a ratio of two linear functions (in x,y,z).
    This allows for non-linear behavior (hyperbolic decay, saturation) 
    that a simple linear combination of monomials might miss or fit poorly.
    
    Deadline guard: If we exceed 140s, return a simple linear prediction.
    """
    global _start_time, _best_params, _best_score
    
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    if elapsed > 140.0:
        # Fallback: Simple linear model using first 4 params as weights, rest ignored or zeroed
        # This is a safe, fast operation.
        # But params might be garbage. Let's just return a constant or linear.
        # To be safe, return 0 or mean of inputs?
        # Actually, if we are timing out, the fit is likely stuck.
        # Return a finite value to avoid 1e30 penalty if possible, but score will be bad.
        # Let's return a simple linear prediction using the first 4 params.
        return params[0] + params[1]*x + params[2]*y + params[3]*z
    
    try:
        pred = _get_basis(x, y, z, params)
        
        # Check for non-finite values
        if not np.all(np.isfinite(pred)):
            # Replace non-finite with 0 to allow optimization to continue (though score will be bad)
            pred = np.where(np.isfinite(pred), pred, 0.0)
            
        return pred
    except Exception:
        # In case of any error, return a safe finite value
        return np.ones_like(x)

# EVOLVE-BLOCK-END
