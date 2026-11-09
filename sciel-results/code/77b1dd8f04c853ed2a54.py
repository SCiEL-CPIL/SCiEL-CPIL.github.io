"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _E_cpl(a, w0, wa, om_m, om_r, om_k=0.0):
    """E(a) = H(a)/H0 for CPL model."""
    a_safe = np.maximum(a, 1e-10)
    h2 = om_m * a_safe**(-3) + om_r * a_safe**(-4) + om_k * a_safe**0 + \
         (1 - om_m - om_r - om_k) * a_safe**(-3*(1+w0+wa))
    return np.sqrt(np.maximum(h2, 1e-30))

def _integrand_val(x, z, w0, wa, om_m, om_r, om_k):
    """The integrand 1/E(a) where a = 1/(1+z*x)."""
    a = 1.0 / (1.0 + z * x)
    return 1.0 / _E_cpl(a, w0, wa, om_m, om_r, om_k)

def _true_dL(z, w0, wa, om_m, om_r, om_k, n=5000):
    """Compute the true integral using high-resolution Simpson's rule."""
    x = np.linspace(0, 1, n)
    if n % 2 == 0:
        n += 1
        x = np.linspace(0, 1, n)
    
    y = _integrand_val(x, z, w0, wa, om_m, om_r, om_k)
    
    h = 1.0 / (n - 1)
    integral = h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]))
    return integral

def _quadrature_error(params, z, w0, wa, om_m, om_r, om_k):
    """Compute relative error of quadrature vs true value for one model."""
    nodes = params[:8]
    weights = params[8:]
    
    if np.any(nodes < 0) or np.any(nodes > 1):
        return 1e10
    if np.any(np.diff(nodes) <= 0):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
        
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
    quad_val = np.sum(weights / E_vals)
    
    true_val = _true_dL(z, w0, wa, om_m, om_r, om_k)
    
    if true_val < 1e-10:
        return abs(quad_val - true_val)
        
    return abs(quad_val - true_val) / true_val

def _loss(params):
    """Max relative error across training models."""
    models = [
        (0.0, 0.0, 0.3, 0.05, 0.65),
        (-0.5, 0.5, 0.3, 0.05, 0.65),
        (-1.0, 0.0, 0.25, 0.05, 0.70),
        (0.5, -0.5, 0.4, 0.05, 0.55),
        (-0.9, 0.3, 0.3, 0.0, 0.70),
        (-1.1, 0.2, 0.2, 0.05, 0.75),
        (-1.5, 0.5, 0.3, 0.05, 0.65),
        (1.0, -1.0, 0.4, 0.05, 0.55),
    ]
    
    z_vals = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    max_err = 0.0
    for m in models:
        w0, wa, om_m, om_r, om_k = m
        for z in z_vals:
            err = _quadrature_error(params, z, w0, wa, om_m, om_r, om_k)
            if err > max_err:
                max_err = err
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Define Nodes: Chebyshev roots mapped to [0, 1]
    # Chebyshev nodes cluster near the boundaries, which is beneficial for
    # integrands that vary rapidly near x=1 (high z) or x=0 (low z).
    n_nodes = 8
    # Roots of T_n(x) in [-1, 1]
    theta = (2 * np.arange(n_nodes) + 1) * np.pi / (2 * n_nodes)
    x_cheb = np.cos(theta)
    # Map [-1, 1] to [0, 1]
    nodes = (x_cheb + 1.0) / 2.0
    nodes = np.sort(nodes)  # Ensure increasing order
    
    # 2. Define Weight Calculation via Moment Matching
    # We want the quadrature to exactly integrate the "worst-case" curvature.
    # The dominant non-polynomial term in CPL is a^(-3(1+w0+wa)).
    # Let alpha = 3(1+w0+wa).
    # We construct a linear system A w = b where A[i,j] = f_j(nodes[i])
    # and b is the true integral of f_j.
    # We choose a set of representative exponents alpha_k to match.
    
    # Representative alpha values covering the range of w0, wa
    # w0 in [-1.5, 1], wa in [-1, 1] -> 1+w0+wa in [-1.5, 3]
    # alpha in [-4.5, 9]
    alphas = np.linspace(-3.0, 7.0, 8)
    
    # Define the basis functions f_j(a) = a^(-alpha_j)
    # Note: The integrand is 1/E(a). E(a) = sqrt(sum om_i a^(-3i) + om_de a^(-alpha)).
    # Matching moments of 1/E is hard. Instead, we match moments of the DE term itself?
    # No, the task asks to minimize error of the full integral.
    # Let's use a weighted least squares approach to find weights w such that
    # sum w_i f_j(nodes_i) approx true_int(f_j) for a set of test functions f_j.
    
    # Test functions: a^p for p in a set of powers that capture the curvature.
    # Since a = 1/(1+zx), a^p is a smooth function of x.
    powers = np.linspace(-2.0, 2.0, 8) # p values for a^p
    
    # Build matrix A where A[i, j] = (a_i)^powers[j]
    # a_i = 1/(1 + z*x_i). Wait, the integral is over x.
    # The quadrature approximates int_0^1 g(x) dx.
    # g(x) = 1/E(a(x)).
    # We want sum w_i g(x_i) ~= int g(x) dx.
    # If we choose g(x) from a basis, we can solve for w.
    # Let's use a basis of polynomials in x? No, that's Gauss-Legendre.
    # Let's use a basis of functions that resemble 1/E(a).
    # 1/E(a) ~ a^(3/2) for matter dominated.
    # Let's use basis functions h_k(x) = (1/(1+z*x))^k for k in some set.
    # This is model-dependent on z.
    
    # Alternative Structural Strategy:
    # Use the fact that for flat universe, 1/E(a) is well-approximated by a polynomial in a.
    # But we need universal weights.
    
    # Let's try a different moment matching:
    # Minimize the error for a set of "extreme" CPL models by solving a linear system
    # for the weights, given the fixed Chebyshev nodes.
    # We pick 8 specific (z, model) combinations that represent the worst cases.
    # For each, we compute the "true" integral I_k and the vector of integrand values v_k.
    # We want w such that w . v_k ~= I_k for all k.
    # This is an overdetermined system (if we use more than 8 cases) or square (8 cases).
    # Let's use 8 diverse cases to solve for 8 weights exactly (or least squares if more).
    
    cases = []
    # Select 8 diverse (z, w0, wa, om_m, om_r, om_k)
    # We want to cover the parameter space corners.
    test_models = [
        (0.5, 0.0, 0.0, 0.3, 0.05, 0.65),   # Low z, Lambda
        (1.0, -1.5, 0.0, 0.3, 0.05, 0.65),  # Med z, Phantom
        (2.0, 1.0, 0.0, 0.4, 0.05, 0.55),   # High z, Quintessence
        (5.0, -1.0, 0.5, 0.3, 0.05, 0.65),  # Very High z, Wa positive
        (0.5, -0.5, -0.5, 0.2, 0.05, 0.75), # Low z, Wa negative
        (10.0, 0.0, 0.0, 0.3, 0.05, 0.65),  # Extreme z, Lambda
        (2.0, -1.5, 1.0, 0.3, 0.0, 0.70),   # Med z, Extreme Wa
        (1.0, 1.0, -1.0, 0.4, 0.05, 0.55),  # Med z, Extreme Wa
    ]
    
    A = np.zeros((8, 8))
    b = np.zeros(8)
    
    for i, (z, w0, wa, om_m, om_r, om_k) in enumerate(test_models):
        # Compute integrand values at our nodes
        a_vals = 1.0 / (1.0 + z * nodes)
        E_vals = _E_cpl(a_vals, w0, wa, om_m, om_r, om_k)
        g_vals = 1.0 / E_vals
        A[i, :] = g_vals
        
        # Compute true integral
        b[i] = _true_dL(z, w0, wa, om_m, om_r, om_k)
        
    # Solve A w = b
    try:
        weights, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    except:
        # Fallback to Gauss-Legendre weights if singular
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        x0_gl = (x_gl + 1.0) / 2.0
        w0_gl = w_gl / 2.0
        nodes = x0_gl
        weights = w0_gl
        return {"nodes": nodes, "weights": weights}
        
    # Check if weights are positive. If not, we need to adjust.
    # Moment matching often produces negative weights.
    # The task requires positive weights.
    # If any weight is negative, we must project onto the positive simplex or restart with different cases.
    
    if np.any(weights <= 0):
        # Project weights to positive simplex
        # Simple projection: clip and renormalize? No, that changes the sum.
        # Better: Solve with constraints? Or pick different test cases.
        # Let's try to add a small regularization to encourage positivity or just use GL if it fails.
        # Actually, let's try to find a set of cases that yields positive weights.
        # Or, we can use a weighted least squares with positivity constraint via scipy.optimize if available, 
        # but let's stick to standard library/numpy for speed and reliability.
        
        # Heuristic: If weights are negative, fall back to Gauss-Legendre 
        # but with Chebyshev nodes? No, GL weights are for GL nodes.
        
        # Let's try a different set of test cases that are "smoother" to encourage positive weights.
        # Or, we can simply use the Gauss-Legendre weights mapped to Chebyshev nodes? No.
        
        # Let's try to solve for weights using a set of polynomial moments instead, which guarantees positive weights for orthogonal polynomials.
        # But Chebyshev nodes with Chebyshev weights are optimal for polynomials.
        # Let's just use the standard Chebyshev quadrature weights.
        # Chebyshev quadrature weights:
        # For nodes x_i = cos((2i-1)pi/(2N)), weights w_i = pi/N.
        # Mapped to [0,1], the weights are halved?
        # Integral from -1 to 1 is pi * sum w_i f(x_i) where w_i = 1/N.
        # Integral from 0 to 1: change var u = (x+1)/2. dx = du.
        # This is getting complicated.
        
        # Let's just check if the lstsq solution is "close" to positive.
        # If not, we use a simple heuristic: 
        # The Gauss-Legendre rule is very robust. 
        # The Chebyshev rule is also robust.
        # Let's return the Chebyshev nodes with uniform weights? 
        # Uniform weights on Chebyshev nodes is a valid quadrature (trapezoidal in theta space).
        
        # Let's try uniform weights on Chebyshev nodes first as a safe fallback if moment matching fails positivity.
        weights = np.ones(8) / 8.0
        
        # Compare error of Moment-Matched (if positive) vs Uniform vs GL
        # We already know Moment-Matched might be negative.
        # Let's calculate error for Uniform Chebyshev.
        
    # We need to ensure weights sum to 1.
    weights = weights / np.sum(weights)
    
    # If weights were negative and we clipped/projected, the sum might not be 1 or they might be non-positive.
    # The code above: if any <= 0, it sets to uniform.
    # Let's refine: if moment matching yields negative weights, we should probably stick to a known positive rule.
    # Let's re-evaluate the "Uniform Chebyshev" strategy.
    
    # Actually, let's try to find a better set of moments that yields positive weights.
    # Positive weights are guaranteed if the nodes are Gauss nodes of some orthogonal polynomial.
    # Chebyshev nodes are Gauss nodes for the weight function 1/sqrt(1-x^2).
    # So if we use the exact Chebyshev weights, they are all positive.
    
    # Let's compute the exact Chebyshev weights for [0,1].
    # Standard Chebyshev quadrature on [-1, 1]:
    # w_i = pi / N
    # On [0, 1], let x = 2t - 1. dx = 2dt.
    # Integral_0^1 f(t) dt = Integral_{-1}^1 f((x+1)/2) * 1/2 dx
    # = 1/2 * sum_i (pi/N) f((x_i+1)/2)
    # So the weights for the mapped nodes are w_i_mapped = (pi/N) * 1/2 = pi/(2N).
    # Sum of weights = N * pi/(2N) = pi/2.
    # We need sum to be 1. So normalize by pi/2.
    # w_i_final = (pi/(2N)) / (pi/2) = 1/N.
    # So uniform weights are the exact Chebyshev quadrature weights!
    
    # So, the "Uniform Weights on Chebyshev Nodes" IS the Gauss-Chebyshev quadrature rule.
    # This is a structurally sound, positive-weight rule.
    
    # Why did we do moment matching? To adapt to 1/E(a).
    # But Gauss-Chebyshev is optimal for functions with endpoint singularities (like 1/sqrt(1-x^2)).
    # 1/E(a) is smooth on [0,1].
    # Gauss-Legendre is optimal for smooth functions.
    # The current program (score 81) uses optimized GL-like nodes.
    # My proposal: Use Gauss-Chebyshev nodes with Uniform Weights.
    # This is a different design family (Chebyshev vs Legendre).
    
    # Let's stick to the Gauss-Chebyshev rule (Uniform weights on Chebyshev nodes) as the primary candidate.
    # It is robust, positive, and exploits the node distribution.
    
    # Let's verify the error of this rule quickly to ensure it's not terrible.
    # If it's worse than GL, we might need to optimize the weights for these fixed nodes.
    
    # Let's do a quick optimization of weights for these fixed Chebyshev nodes.
    # This is a convex problem if we minimize sum of squared errors?
    # Minimize max error is not convex.
    # But we can minimize the L2 error of the moments.
    
    # Let's use the moment matching weights if they are positive.
    # If not, use Uniform.
    
    # Let's re-run the moment matching check properly.
    weights_mm, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    if np.all(weights_mm > 0):
        weights = weights_mm / np.sum(weights_mm)
    else:
        # Fallback to Uniform (Gauss-Chebyshev)
        weights = np.ones(8) / 8.0
        
    # Final check
    if not np.all(np.diff(nodes) > 0):
        # Fallback to GL
        x_gl, w_gl = np.polynomial.legendre.leggauss(8)
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
        weights = weights / np.sum(weights)
        
    return {"nodes": nodes, "weights": weights}

# EVOLVE-BLOCK-END
