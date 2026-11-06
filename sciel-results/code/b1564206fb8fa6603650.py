import numpy as np
from scipy.optimize import minimize

def construct():
    # Start with Gauss-Legendre
    n = 8
    nodes0, weights0 = np.polynomial.legendre.leggauss(n)
    nodes0 = (nodes0 + 1) / 2
    weights0 = weights0 * 0.5
    
    # Define test cosmologies (CPL)
    # E(z) = (1+z)^(3/2) * (1+z)^(3*w1/2) / (1+w1*z)^(3*w1/2) ? 
    # Standard CPL: w(z) = w0 + w1*z/(1+z)
    # E(z) = (1+z)^(3/2) * exp(1.5 * w1 * (1 - 1/(1+z)) / w0? No.
    # Integral of 3 w(u) / (1+u) du.
    # For CPL, E(z) = (1+z)^(3/2) * (1+w1*z)^(3*w1/2) / (1+w1*z)^(...)
    # Let's use a simpler approximation for the optimizer:
    # E(z) = (1+z)^(3/2) for w=-1
    # E(z) = (1+z)^(3/2) * (1+z)^(1.5*w1) for w0=-1, w1!=0?
    
    # To be safe, I will use a generic power law E(z) = (1+z)^gamma
    # gamma in [1.5, 2.0, 2.5]
    
    def error_function(params):
        # params: 16 values
        nodes = params[:8]
        weights = params[8:]
        
        # Enforce constraints softly or check validity
        if np.any(np.diff(nodes) <= 0) or np.any(nodes < 0) or np.any(nodes > 1):
            return 1e10
        if np.any(weights < 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-6:
            return 1e10
            
        max_err = 0.0
        # Test a few z values and models
        for z in [0.1, 0.5, 1.0, 2.0]:
            for gamma in [1.5, 2.0, 2.5]:
                # Exact integral: int_0^z (1+z')^(-gamma) dz'
                # = [ (1+z')^(1-gamma) / (1-gamma) ]_0^z
                # = ( (1+z)^(1-gamma) - 1 ) / (1-gamma)
                # Note: if gamma=1, it's ln(1+z). Our gammas are != 1.
                exact = ((1+z)**(1-gamma) - 1) / (1-gamma)
                
                # Quadrature
                # dL/(c/H0) = (1+z) * z * sum w_i / E(z*u_i)
                # E(z*u) = (1+z*u)^gamma
                a_vals = 1.0 / (1.0 + z * nodes)
                # Wait, prompt says a_i = 1/(1+z*nodes[i])
                # And integrand is 1/E(a_i)?
                # E(a) is H(a)/H0. If E(z) = (1+z)^gamma, then E(a) is not (1+a)^gamma.
                # The prompt says E(a) = H(a)/H0.
                # If the model is defined in terms of z, E(z) = H(z)/H0.
                # Then E(a) where a=1/(1+z') is H(1/(1+z'))/H0.
                # This is confusing.
                # Usually E(z) is defined as H(z)/H0.
                # The prompt says: sum weights[i] / E(a_i) where a_i = 1/(1+z*nodes[i]).
                # This implies the integrand is 1/E(a) = H0/H(a).
                # And the variable of integration is a?
                # dL = c/H0 * (1+z) * int_0^z dz'/E(z').
                # Change variable a = 1/(1+z'). da = -1/(1+z')^2 dz' = -a^2 dz'.
                # dz' = -da/a^2.
                # Limits: z'=0 -> a=1. z'=z -> a=1/(1+z).
                # int_0^z dz'/E(z') = int_1^{1/(1+z)} (-da/a^2) / (H(a)/H0)
                # = H0 * int_{1/(1+z)}^1 da / (a^2 H(a)).
                # This doesn't match the prompt's formula directly.
                # Prompt: (1+z)*z*sum w_i / E(a_i).
                # This looks like an approximation of int_0^1 du / E(z*u).
                # I will assume the prompt's formula is the one to match.
                
                # I will assume E(z) = (1+z)^gamma for the test.
                # Then E(a_i) in the prompt's context likely means the Hubble factor at the redshift corresponding to a_i?
                # No, a_i is scale factor.
                # If E(a) = H(a)/H0, and H(a) = H0 a^{-3/2} for matter.
                # Then 1/E(a) = a^{3/2}.
                # The prompt says sum w_i / E(a_i).
                # So integrand is 1/E(a).
                # If E(a) = a^{-3/2}, then 1/E(a) = a^{3/2}.
                # Let's test with E(a) = a^{-3/2} (matter).
                # Then term is a_i^{3/2}.
                
                # I will just use the power law in z for E(z) and assume the prompt's E(a_i) 
                # is a notation for the Hubble factor at the redshift z_i = nodes[i]*z.
                # i.e., 1/E(z_i).
                
                z_i = z * nodes
                E_vals = (1 + z_i) ** gamma
                approx = z * np.dot(weights, 1.0 / E_vals)
                
                rel_err = abs(approx - exact) / abs(exact)
                if rel_err > max_err:
                    max_err = rel_err
                    
        return max_err

    # Optimization
    x0 = np.concatenate([nodes0, weights0])
    res = minimize(error_function, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if res.success:
        nodes_opt = res.x[:8]
        weights_opt = res.x[8:]
        # Normalize weights
        weights_opt = weights_opt / np.sum(weights_opt)
        return {"nodes": nodes_opt, "weights": weights_opt}
    else:
        return {"nodes": nodes0, "weights": weights0}
