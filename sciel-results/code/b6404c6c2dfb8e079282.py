import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    p = params
    
    # Ensure params are finite
    if not np.all(np.isfinite(p)):
        return np.zeros_like(x)

    # Map inputs from [0.5, 2.0] to [-1, 1]
    # center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75

    # Construct Chebyshev polynomials of the first kind T_n(t)
    # T_0(t) = 1
    # T_1(t) = t
    # T_2(t) = 2t^2 - 1
    # T_3(t) = 4t^3 - 3t
    
    # We have 10 parameters. We can distribute them across the variables.
    # A reasonable symmetric allocation for a 3-variable problem:
    # p[0]: Constant term (T0 for all)
    # p[1]: u term (T1(u))
    # p[2]: v term (T1(v))
    # p[3]: w term (T1(w))
    # p[4]: u^2 term (T2(u))
    # p[5]: v^2 term (T2(v))
    # p[6]: w^2 term (T2(w))
    # p[7]: u^3 term (T3(u))
    # p[8]: v^3 term (T3(v))
    # p[9]: w^3 term (T3(w))
    
    # This ignores cross-terms (like uv), but with only 10 params, including
    # cross-terms would reduce the degree for single variables or make the model
    # very sparse. Given the inputs are independent uniform, a sum of univariate
    # functions is a strong prior. If the hidden function is separable, this is optimal.
    # If it has interactions, this will underfit, but the current Gaussian model
    # also fails to capture global trends well.
    
    # Let's try a mix that allows some interaction if we can, but 10 params is tight.
    # Option A: Full quadratic (10 params exactly for 3 vars: 1, x, y, z, x^2, y^2, z^2, xy, xz, yz)
    # This is actually a very standard and strong candidate.
    # The current program uses Gaussians. The "inspiration" that scored 7.39 used a quadratic.
    # The current score is 1.29. The quadratic inspiration was WORSE (7.39 > 1.29 is worse? No, lower is better. 7.39 is worse than 1.29).
    # Wait, the prompt says "lower is better".
    # Current score: 1.2994.
    # Inspiration 1 (quadratic): 7.3933. This is much worse.
    # Inspiration 2: 3.1172.
    # Inspiration 3: 1.9120.
    
    # So the current Gaussian model (1.29) is actually BETTER than the full quadratic (7.39) and other polynomials.
    # Why? Maybe the function is NOT a simple polynomial. Maybe it's localized?
    # But the Gaussian model is a sum of 3 independent Gaussians (plus bias).
    # This implies the function is likely separable: f(x,y,z) = g(x) + h(y) + k(z) + c.
    # OR, the BFGS optimizer got stuck in a local minimum for the polynomial due to ill-conditioning or the specific landscape.
    
    # Let's rethink. If the function is separable, a basis of functions that can represent arbitrary 1D shapes is good.
    # Gaussians are local. Polynomials are global.
    # What if we use a Fourier-like basis or a more flexible 1D basis?
    # Or what if we keep the separability assumption but use a better 1D basis?
    # Chebyshev polynomials are orthogonal and well-conditioned.
    # Let's try a separable model with Chebyshev polynomials up to degree 3 for each variable.
    # 1 (const) + 3 (linear) + 3 (quad) + 3 (cubic) = 10 params.
    # This is structurally similar to the quadratic but with higher degree freedom for each variable,
    # and better conditioning than raw powers.
    
    T0_u = np.ones_like(u)
    T1_u = u
    T2_u = 2*u**2 - 1
    T3_u = 4*u**3 - 3*u
    
    T0_v = np.ones_like(v)
    T1_v = v
    T2_v = 2*v**2 - 1
    T3_v = 4*v**3 - 3*v
    
    T0_w = np.ones_like(w)
    T1_w = w
    T2_w = 2*w**2 - 1
    T3_w = 4*w**3 - 3*w
    
    # Combine
    result = (p[0] * T0_u + 
              p[1] * T1_u + 
              p[2] * T1_v + 
              p[3] * T1_w + 
              p[4] * T2_u + 
              p[5] * T2_v + 
              p[6] * T2_w + 
              p[7] * T3_u + 
              p[8] * T3_v + 
              p[9] * T3_w)
    
    # Ensure result is finite
    if not np.all(np.isfinite(result)):
        return np.zeros_like(x)
        
    return result
