"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma


def construct():
    """
    Construct an 8-node positive quadrature rule for the dimensionless
    luminosity distance integral in flat Lambda-CDM-like models.
    
    The integral to approximate is:
        I = ∫_0^1 z / E(a(z)) dz
    where a = 1/(1+z), so z = 1/a - 1.
    
    However, the task specifies:
        dL/(c/H0) = (1+z) * z * sum_i weights[i] / E(a_i)
    with a_i = 1/(1 + z * nodes[i]).
    
    Wait, let me re-read carefully:
    "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i), where a_i=1/(1+z*nodes[i])"
    
    This is confusing. Let me parse it again.
    
    Actually, I think the formula means: for a given redshift z, the luminosity distance is
    dL/(c/H0) = (1+z) * ∫_0^z dz' / E(a(z'))
    
    But the quadrature is applied to the integral part. Let me re-read:
    
    "dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)"
    
    Hmm, that doesn't look right dimensionally. Let me think about this differently.
    
    Actually, I think the task is defining a specific quadrature rule where:
    - We have 8 nodes t_i in [0,1]
    - Weights w_i summing to 1
    - For a given redshift z, we evaluate the integrand at points a_i = 1/(1+z*t_i)
    - The approximation is: dL/(c/H0) = (1+z) * z * sum_i w_i / E(a_i)
    
    Wait, but that would mean the quadrature is approximating ∫_0^1 dt / E(1/(1+zt))
    and the full formula has an extra factor of z.
    
    Let me verify: if E(a) = 1 (constant), then dL/(c/H0) should be (1+z)*z.
    With our formula: (1+z)*z*sum_i w_i / 1 = (1+z)*z*1 = (1+z)*z. ✓
    
    So the quadrature is approximating ∫_0^1 dt / E(1/(1+zt)) using 8 nodes t_i and weights w_i.
    
    The key insight: the integrand depends on z through the argument 1/(1+zt).
    For different z values, the function we're integrating changes shape.
    We need a universal rule that works well across different cosmologies.
    
    Strategy: We want nodes and weights that minimize the worst-case relative error
    across a set of test cosmologies.
    
    Let's define the problem more carefully:
    - For a given cosmology (defined by E(a)), and a target redshift z,
      the exact integral is J(z) = ∫_0^1 dt / E(1/(1+zt))
    - Our approximation is J_approx(z) = sum_i w_i / E(1/(1+z*t_i))
    - The relative error is |J_approx - J| / J
    
    We want to minimize the max relative error over:
    - All test cosmologies
    - All test redshifts z
    
    This is a min-max optimization problem.
    
    Approach: 
    1. Define a set of representative cosmologies (wCDM models with different w0, wa)
    2. Define a set of test redshifts
    3. Optimize nodes and weights to minimize worst-case relative error
    
    For the optimization, we can use:
    - Parameterize nodes as strictly increasing values in (0,1)
    - Use a barrier or transformation to enforce positivity of weights and sum to 1
    
    Let me implement this:
    """
    
    # Define a set of test cosmologies: wCDM with w0, wa pairs
    # E(a)^2 = (1-a^4) * a^(-3(1+w0+wa)) * (1+a*wa)^(-3)
    # For flat wCDM: E(a)^2 = a^(-3(1+w0+wa)) * (1+a*wa)^(-3) + (1-a^4)
    # Actually, for flat wCDM: E(a)^2 = Ω_m a^(-3) + Ω_Λ a^(-3(1+w0+wa))(1+a*wa)^(-3)
    # But let's use a simpler form. The standard CPL parametrization:
    # w(a) = w0 + wa*(1-a)
    # E(a)^2 = Ω_m a^(-3) + (1-Ω_m) * exp(3∫_a^1 (1+w(a'))/a' da')
    # This is complex. Let's use a simpler model for testing.
    
    # For simplicity, let's use E(a) = a^(-3(1+w)/2) for constant w, plus matter-only and Lambda-CDM
    # Actually, let's just use a few representative E(a) functions.
    
    def make_E(w0, wa, Om=0.3):
        """Create E(a) function for CPL dark energy."""
        def E(a):
            # For flat CPL: E(a)^2 = Om*a^(-3) + (1-Om)*a^(-3*(1+w0+wa))*(1+a*wa)^(-3)
            # This is an approximation. Let's use the exact integral form.
            # Actually, for simplicity, use: E(a)^2 = Om*a^(-3) + (1-Om)*exp(3*(w0+wa)*np.log(1/(1+a*wa)) + 3*wa*(1-a))
            # This is getting complex. Let's use a simpler parametrization.
            
            # Use: E(a)^2 = Om*a^(-3) + (1-Om)*a^(-3*(1+w0+wa))*(1+a*wa)^(-3)
            # This is not exact but representative.
            term_matter = Om * a**(-3)
            term_de = (1 - Om) * a**(-3 * (1 + w0 + wa)) * (1 + a * wa)**(-3)
            return np.sqrt(term_matter + term_de)
        return E
    
    # Test cosmologies
    test_cosmos = [
        make_E(-1.0, 0.0),   # Lambda-CDM
        make_E(-0.9, -0.3),  # wCDM-like
        make_E(-1.1, 0.5),   # Quintessence-like
        make_E(-0.8, -0.5),  # Phantom-like
        make_E(-1.0, 0.2),   # Mildly evolving
        make_E(-1.2, 0.8),   # Strongly evolving
    ]
    
    # Test redshifts
    test_zs = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    n = 8
    
    # Initial guess: use Gauss-Legendre nodes and weights
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    # Map from [-1,1] to [0,1]
    nodes_init = 0.5 * (x_gl + 1)
    weights_init = 0.5 * w_gl
    
    def objective(params):
        """Minimize worst-case relative error."""
        # Extract nodes and weights from params
        # Use softplus to ensure positive weights
        nodes = params[:n]
        log_weights = params[n:]
        
        # Enforce strictly increasing nodes in (0,1)
        # Use sigmoid-like transformation
        nodes = 1.0 / (1.0 + np.exp(-10 * (nodes - 0.5)))
        # Ensure strict increase by adding small offsets
        for i in range(1, n):
            nodes[i] = max(nodes[i], nodes[i-1] + 1e-6)
        # Clamp to [1e-6, 1-1e-6]
        nodes = np.clip(nodes, 1e-6, 1 - 1e-6)
        
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        max_rel_err = 0.0
        
        for E_func in test_cosmos:
            for z in test_zs:
                # Compute exact integral using adaptive quadrature
                def integrand(t):
                    a = 1.0 / (1.0 + z * t)
                    return 1.0 / E_func(a)
                
                from scipy.integrate import quad
                exact, _ = quad(integrand, 0, 1, limit=100)
                
                # Compute quadrature approximation
                approx = 0.0
                for i in range(n):
                    a_i = 1.0 / (1.0 + z * nodes[i])
                    approx += weights[i] / E_func(a_i)
                
                rel_err = abs(approx - exact) / exact if exact > 0 else 0.0
                max_rel_err = max(max_rel_err, rel_err)
        
        return max_rel_err
    
    # This optimization is expensive. Let's use a simpler approach.
    # Instead of full min-max optimization, let's use a known good quadrature rule.
    # 
    # Alternative approach: Use Gaussian quadrature with a weight function that
    # captures the typical behavior of 1/E(a).
    #
    # For Lambda-CDM, 1/E(a) ~ a^(3/2) for small a (matter dominated) and ~1 for large a.
    # So the integrand 1/E(1/(1+zt)) has a power-law behavior.
    #
    # Let's try a Gauss-Jacobi quadrature rule, which is optimal for integrands
    # of the form (1-t)^alpha * t^beta.
    
    # For a typical cosmology, the integrand 1/E(1/(1+zt)) as a function of t
    # is smooth but has varying curvature. A Gauss-Legendre rule should be good.
    
    # Let's just return the Gauss-Legendre rule, which is known to be optimal
    # for smooth functions.
    
    return {"nodes": nodes_init, "weights": weights_init}
# EVOLVE-BLOCK-END
