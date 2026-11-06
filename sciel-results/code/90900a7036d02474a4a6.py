"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct an 8-node positive quadrature rule for dimensionless luminosity distance.
    
    Strategy:
    1. Start with Gauss-Legendre nodes/weights on [0,1] (spectrally accurate for smooth integrands).
    2. The integrand for luminosity distance in CPL models is 1/E(z') where E(a)=H(a)/H0.
       For flat spatial geometry with CPL dark energy, E(a) = (a^(-3(1+w0+wa))) * (1 - a^(-3(1+w0+wa)))^(-1/2) * (1 - a^(-3(1+w0+wa)))... 
       Actually, let's be more precise. The Hubble parameter is:
       H(a)^2 = H0^2 [Ωm a^-3 + Ωk a^-2 + ΩΛ a^-3(1+w0+wa) exp(3wa(1/a-1))]
       For flat universe, Ωk=0, so H(a)^2 = H0^2 [Ωm a^-3 + ΩΛ a^-3(1+w0+wa) exp(3wa(1/a-1))]
       E(a) = H(a)/H0 = sqrt(Ωm a^-3 + ΩΛ a^-3(1+w0+wa) exp(3wa(1/a-1)))
       
       The luminosity distance is:
       dL/(c/H0) = (1+z) * ∫_0^z dz' / E(z')
       where z' is the redshift variable, and a' = 1/(1+z')
       
       Changing variables from z' to a': when z'=0, a'=1; when z'=z, a'=1/(1+z)
       dz' = -da'/a'^2
       So ∫_0^z dz'/E(z') = ∫_{1/(1+z)}^1 da'/(a'^2 * E(a'))
       
       The task states: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])
       
       Wait, this is a different formulation. Let me re-read the task.
       
       "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
       
       So the quadrature is applied to the integral ∫_0^z dz'/E(z'), and the nodes are in [0,1]
       representing the fraction of the redshift range. So z' = z * x where x is in [0,1].
       Then a' = 1/(1+z') = 1/(1+z*x)
       
       The integral becomes ∫_0^1 z * dx / E(1/(1+z*x))
       So dL/(c/H0) = (1+z) * z * ∫_0^1 dx / E(1/(1+z*x))
       
       The quadrature approximates ∫_0^1 dx / E(1/(1+z*x)) ≈ sum_i w_i / E(1/(1+z*x_i))
       
       So dL/(c/H0) ≈ (1+z) * z * sum_i w_i / E(a_i) where a_i = 1/(1+z*x_i)
       
       The nodes x_i are in [0,1], strictly increasing. The weights w_i are positive and sum to 1.
       
       For the quadrature to be accurate, we need to approximate ∫_0^1 f(x) dx where f(x) = 1/E(1/(1+z*x)).
       
       The function f(x) depends on z and the cosmological parameters. For different z and different
       cosmologies, f(x) has different shapes. We need a universal quadrature that works well
       for all reasonable z and all CPL cosmologies.
       
       Key insight: For small z, f(x) is nearly constant (E is nearly 1), so any quadrature works.
       For large z, f(x) can vary significantly. The worst case is likely at high z with strong
       dark energy evolution.
       
       Gauss-Legendre quadrature is optimal for smooth functions. Since E(a) is smooth for all
       physical CPL parameters, the integrand f(x) = 1/E(1/(1+z*x)) should be smooth in x.
       
       Let's start with Gauss-Legendre nodes and weights on [0,1].
    """
    # Get Gauss-Legendre nodes and weights on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    
    # Transform from [-1, 1] to [0, 1]
    # x_new = (x_old + 1) / 2, w_new = w_old / 2
    nodes = (x_gl + 1.0) / 2.0
    weights = w_gl / 2.0
    
    # Ensure nodes are strictly increasing and in [0,1]
    # Gauss-Legendre nodes are already in (-1,1), so transformed nodes are in (0,1)
    # and strictly increasing.
    
    # Ensure weights are positive and sum to 1
    # Gauss-Legendre weights are positive and sum to 2 on [-1,1], so after scaling they sum to 1.
    
    # Due to floating point, we might need to normalize
    weights = weights / np.sum(weights)
    
    # Ensure nodes are strictly in [0,1] with a small margin
    eps = 1e-10
    nodes = np.clip(nodes, eps, 1.0 - eps)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
