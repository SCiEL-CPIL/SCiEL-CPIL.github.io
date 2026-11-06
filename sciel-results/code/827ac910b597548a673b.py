"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Returns a universal 8-point positive quadrature rule for the luminosity distance
    integral in flat FLRW cosmologies.
    
    The integral is:
    dL/(c/H0) = (1+z) * z * \int_0^z dz' / E(z')
    
    We approximate \int_0^z dz' / E(z') using nodes and weights in the z-space.
    However, the formula given in the task is:
    dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])
    
    This implies the quadrature is applied to the integral:
    I(z) = \int_0^1 du / E(u)  ? No.
    
    Let's look at the formula again:
    dL/(c/H0) = (1+z) * z * \sum w_i / E(a_i)
    with a_i = 1 / (1 + z * n_i)
    
    Standard luminosity distance:
    d_L = (1+z) * c/H0 * \int_0^z dz' / E(z')
    
    Let u = z'/z, so z' = z*u, dz' = z*du.
    \int_0^z dz'/E(z') = \int_0^1 z*du / E(z*u) = z * \int_0^1 du / E(z*u)
    
    So d_L/(c/H0) = (1+z) * z * \int_0^1 du / E(z*u)
    
    The task says: sum_i weights[i] / E(a_i) where a_i = 1/(1+z*nodes[i]).
    This matches if we interpret the integral variable as u, and the argument to E is a = 1/(1+z*u).
    Wait, E(a) = H(a)/H0.
    E(z*u) = H(z*u)/H0.
    Usually E(z) is defined as H(z)/H0.
    So the term is 1/E(z*u).
    
    The task formula: sum w_i / E(a_i) with a_i = 1/(1+z*n_i).
    This is equivalent to sum w_i * H0/H(z*n_i).
    
    So we need to approximate \int_0^1 du / E(z*u) with \sum w_i / E(z*n_i).
    Wait, the task says E(a_i) where a_i = 1/(1+z*n_i).
    If E is defined as E(a) = H(a)/H0, then E(a_i) = H(a_i)/H0.
    But a_i is a scale factor.
    z*n_i is a redshift.
    So a = 1/(1+z).
    So a_i = 1/(1+z*n_i) corresponds to redshift z_i = z*n_i.
    So E(a_i) = E(z_i) = H(z_i)/H0.
    
    So the quadrature approximates \int_0^1 du / E(z*u) \approx \sum w_i / E(z*n_i).
    
    The integrand is f(u; z) = 1 / E(z*u).
    We want to find nodes n_i and weights w_i such that for all z and all cosmologies,
    \sum w_i f(n_i; z) \approx \int_0^1 f(u; z) du.
    
    Since f(u; z) depends on z, a single set of nodes/weights must work for all z.
    This is a hard problem.
    
    However, for small z, E(z) \approx 1. f(u) \approx 1.
    For large z, E(z) varies.
    
    The "midpoint rule" uses n_i = (i+0.5)/8 and w_i = 1/8.
    This is the 8-point midpoint rule for \int_0^1.
    
    Can we do better?
    Gauss-Legendre nodes and weights are optimal for polynomials.
    If 1/E(z*u) is smooth in u, Gauss-Legendre might be better than midpoint.
    
    Let's try Gauss-Legendre nodes and weights on [0,1].
    """
    n = 8
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n)
    # Map to [0, 1]
    nodes = 0.5 * (x + 1)
    weights = 0.5 * w
    
    # Ensure nodes are strictly increasing (leggauss returns sorted)
    # Ensure weights are positive (they are for Legendre)
    # Ensure sum of weights is 1 (it is)
    
    # Check constraints:
    # Nodes in [0,1], strictly increasing.
    # Weights positive, sum to 1.
    
    # Let's verify strictly increasing.
    # leggauss nodes are distinct and symmetric.
    # Mapping preserves order.
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
