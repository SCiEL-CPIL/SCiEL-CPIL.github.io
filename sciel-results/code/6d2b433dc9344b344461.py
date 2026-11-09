import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Multiplicative stress model: elastic response modulated by hardening factor.
    
    sigma = E(T) * eps * (1 + a(T)*tanh(b(T)*eps) + c(T)*eps + d(T)*eps^2)
    
    The multiplicative structure ensures sigma(0)=0 and couples plastic
    hardening to the elastic baseline, better capturing yield/hardening
    coupling in metals.
    
    Params:
    p[0]: E0 (Young's modulus base)
    p[1]: c_E (temp coeff for E)
    p[2]: a0 (hardening amplitude base)
    p[3]: c_a (temp coeff for a)
    p[4]: b0 (hardening scale base, controls tanh transition)
    p[5]: c_b (temp coeff for b)
    p[6]: c0 (linear strain correction base)
    p[7]: c_c (temp coeff for c)
    p[8]: d0 (quadratic strain correction base)
    p[9]: c_d (temp coeff for d)
    """
    eps = strain
    T = temp
    
    # Parameter unpacking
    E0 = params[0]
    c_E = params[1]
    a0 = params[2]
    c_a = params[3]
    b0 = params[4]
    c_b = params[5]
    c0 = params[6]
    c_c = params[7]
    d0 = params[8]
    c_d = params[9]
    
    # Temperature-dependent coefficients
    E_T = E0 * (1.0 + c_E * T)
    a_T = a0 * (1.0 + c_a * T)
    b_T = b0 * (1.0 + c_b * T)
    c_T = c0 * (1.0 + c_c * T)
    d_T = d0 * (1.0 + c_d * T)
    
    # Elastic baseline
    elastic = E_T * eps
    
    # Hardening factor: 1 + a*tanh(b*eps) + c*eps + d*eps^2
    # tanh provides smooth saturation of the primary hardening contribution
    tanh_arg = b_T * eps
    hardening = 1.0 + a_T * np.tanh(tanh_arg) + c_T * eps + d_T * (eps * eps)
    
    stress = elastic * hardening
    
    return stress
