"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_jacobi

def construct():
    # Use Gauss-Jacobi quadrature with alpha=0, beta=1 on [0,1].
    # This biases sampling towards t=1 (small a, high redshift), where the
    # integrand 1/E(a) tends to be larger and more variable in dark-energy models.
    # The weight function is (1-t)^0 * t^1 = t, giving more weight to the
    # high-z end of the integration interval.
    
    # Get nodes and weights for the interval [0, 1] with alpha=0, beta=1
    # roots_jacobi(n, alpha, beta) returns nodes on [-1, 1]
    # We need to transform to [0, 1]
    x, w = roots_jacobi(8, 0.0, 1.0)
    
    # Transform from [-1, 1] to [0, 1]
    # t = (x + 1) / 2
    # dt = dx / 2
    # The weights for the transformed integral need adjustment
    # For Gauss-Jacobi on [-1,1] with weight (1-x)^alpha (1+x)^beta:
    # sum w_i f(x_i) approximates integral_{-1}^{1} (1-x)^alpha (1+x)^beta f(x) dx
    # We want integral_0^1 f(t) dt
    # With t = (x+1)/2, x = 2t-1, dx = 2dt
    # Weight (1-x)^alpha (1+x)^beta = (1-(2t-1))^alpha (1+(2t-1))^beta = (2-2t)^alpha (2t)^beta = 2^(alpha+beta) (1-t)^alpha t^beta
    # So integral_{-1}^{1} 2^(alpha+beta)(1-t)^alpha t^beta f((2t-1)) 2 dt = 2^(alpha+beta+1) integral_0^1 (1-t)^alpha t^beta f(t) dt
    # For alpha=0, beta=1: 2^(0+1+1) = 4
    # So sum w_i f(x_i) approximates 4 * integral_0^1 t f(t) dt
    # We want integral_0^1 f(t) dt, so we need to divide by the weight function effect
    # Actually, the quadrature formula gives: sum w_i f(x_i) ≈ integral_{-1}^{1} (1-x)^alpha (1+x)^beta f(x) dx
    # After transformation: ≈ 2^(alpha+beta+1) integral_0^1 (1-t)^alpha t^beta f(t) dt
    # To get integral_0^1 f(t) dt, we'd need to use a different approach or adjust
    # 
    # Actually, let's use the fact that we can directly construct quadrature for 
    # integral_0^1 f(t) dt by using Gauss-Legendre as the base, but that's what we had.
    # 
    # Alternative: Use Gauss-Jacobi quadrature but interpret it correctly.
    # The roots_jacobi gives a rule that integrates (1-x)^alpha (1+x)^beta f(x) exactly.
    # For our purpose, we want a rule for integral_0^1 f(t) dt.
    # 
    # Let's instead use a different approach: find optimal nodes and weights
    # that minimize the worst-case error over a family of plausible integrands.
    # 
    # For now, let's use Gauss-Lobatto or a modified approach.
    # Actually, let's just use Gauss-Legendre but with a slight modification:
    # use a weighted least-squares approach to find nodes/weights that minimize
    # the maximum error over a set of test integrands.
    
    # For simplicity and reliability, let's use Gauss-Legendre as it's well-tested
    # and has good properties. But let's try a few different quadrature rules.
    
    # Actually, the issue is that roots_jacobi gives weights for the weighted integral.
    # To use it for unweighted integral, we'd need f(t) to absorb the weight.
    # This doesn't directly help.
    
    # Let's go back to Gauss-Legendre but consider that the integrand 
    # g(t) = 1/E(1/(1+zt)) may have specific properties.
    # For flat LambdaCDM, E(a) = (Omega_m a^{-3} + Omega_Lambda)^0.5
    # 1/E(a) = (Omega_m a^{-3} + Omega_Lambda)^(-0.5)
    # With a = 1/(1+zt), this is (Omega_m (1+zt)^3 + Omega_Lambda)^(-0.5)
    # This is a smooth, monotone decreasing function of t.
    
    # Gauss-Legendre should work well for this. Let's stick with it.
    x, w = np.polynomial.legendre.leggauss(8)
    nodes = (x + 1.0) / 2.0
    weights = w / 2.0
    
    # Ensure numerical precision for the sum
    # The sum of weights should be 1 within machine epsilon
    total = np.sum(weights)
    if abs(total - 1.0) > 1e-14:
        weights = weights / total
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
