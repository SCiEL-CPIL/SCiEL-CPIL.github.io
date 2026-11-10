import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import roots_cheby2, roots_legendre

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    # Define a comprehensive set of test cosmologies and redshifts
    # The rule must be universal, so we optimize for a representative grid
    models = [
        (0.3, 0.7, -1.0, 0.0),   # LCDM
        (0.3, 0.7, -1.0, 0.3),   # CPL
        (0.3, 0.7, -1.0, -0.3),  # CPL
        (0.3, 0.7, -0.8, -0.5),  # CPL
        (0.3, 0.7, -1.2, 0.2),   # CPL
        (0.3, 0.7, -0.5, 0.0),   # wCDM
        (0.25, 0.75, -1.0, 0.0), # Different Om
        (0.35, 0.65, -1.0, 0.0), # Different Om
    ]
    
    # Redshifts: logarithmic spacing to cover low and high z
    z_vals = np.unique(np.concatenate([
        np.linspace(0.01, 1.0, 10),
        np.logspace(np.log10(1.0), np.log10(10.0), 15)
    ]))

    def E_of_z(z, Om, OL, w0, wa):
        """Compute E(z) = H(z)/H0 for CPL model."""
        # Handle LCDM specifically for numerical stability and speed
        if abs(wa) < 1e-12 and abs(w0 + 1.0) < 1e-12:
            return np.sqrt(Om * (1.0 + z)**3 + OL)
        else:
            # General CPL
            term1 = Om * (1.0 + z)**3
            # E^2 = Om(1+z)^3 + OL(1+z)^(3(1+w0)) exp(3wa z/(1+z))
            exponent = 3.0 * wa * z / (1.0 + z)
            term2 = OL * (1.0 + z)**(3.0 * (1.0 + w0)) * np.exp(exponent)
            return np.sqrt(term1 + term2)

    def true_distance(z, Om, OL, w0, wa):
        """High-accuracy reference distance using adaptive quadrature."""
        if z <= 0:
            return 0.0
        integrand = lambda zp: 1.0 / E_of_z(zp, Om, OL, w0, wa)
        # Use high precision quad
        val, _ = quad(integrand, 0, z, limit=200, epsabs=1e-14, epsrel=1e-14)
        return (1.0 + z) * val

    def quadrature_distance(z, nodes, weights, Om, OL, w0, wa):
        """Compute distance using the quadrature rule."""
        if z <= 0:
            return 0.0
        z_nodes = z * nodes
        E_vals = E_of_z(z_nodes, Om, OL, w0, wa)
        integral_approx = np.sum(weights / E_vals)
        return (1.0 + z) * z * integral_approx

    # Pre-compute reference distances to speed up objective function
    # ref_dists[i][j] is true distance for model i, redshift j
    ref_dists = np.zeros((len(models), len(z_vals)))
    for i, m in enumerate(models):
        for j, z in enumerate(z_vals):
            ref_dists[i][j] = true_distance(z, *m)

    def params_to_nodes_weights(params):
        """
        Convert parameters to nodes and weights.
        params: [u1, d1, d2, ..., d7, y1, y2, ..., y8]
        u1 > 0, di > 0 ensure strictly increasing nodes in (0,1)
        y_i are log-weights
        """
        u1 = params[0]
        ds = params[1:8]
        ys = params[8:16]
        
        nodes = np.zeros(8)
        nodes[0] = u1
        for i in range(1, 8):
            nodes[i] = nodes[i-1] + ds[i-1]
        
        # Check bounds
        if nodes[0] <= 0 or nodes[-1] >= 1.0:
            return None, None
        if not np.all(np.diff(nodes) > 1e-9):
            return None, None
            
        # Normalize weights from log-space
        y_max = np.max(ys)
        w_exp = np.exp(ys - y_max)
        w_sum = np.sum(w_exp)
        if w_sum < 1e-15:
            return None, None
        weights = w_exp / w_sum
        
        return nodes, weights

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        nodes, weights = params_to_nodes_weights(params)
        if nodes is None:
            return 1e10
            
        max_err = 0.0
        # Vectorized calculation where possible, but loop over models/z for clarity
        # z_nodes is 8x1, weights is 8x1
        # We need to compute sum(weights / E(z*nodes)) for each model/z
        
        for i, m in enumerate(models):
            Om, OL, w0, wa = m
            for j, z in enumerate(z_vals):
                d_true = ref_dists[i][j]
                if d_true < 1e-10:
                    continue
                
                z_nodes = z * nodes
                E_vals = E_of_z(z_nodes, Om, OL, w0, wa)
                integral_approx = np.sum(weights / E_vals)
                d_quad = (1.0 + z) * z * integral_approx
                
                rel_err = abs(d_quad - d_true) / d_true
                if rel_err > max_err:
                    max_err = rel_err
                    if max_err > 0.5: # Early exit if error is large
                        return max_err
                        
        return max_err

    # --- Initialization Strategies ---
    
    def get_params_from_nodes_weights(nodes, weights):
        u1 = nodes[0]
        ds = np.diff(nodes)
        ys = np.log(weights + 1e-15)
        return np.concatenate(([u1], ds, ys))

    # Strategy 1: Chebyshev Nodes (Clenshaw-Curtis style)
    # Chebyshev nodes of the second kind mapped to [0,1]
    # x_i = 0.5 * (1 + cos(pi * i / 7)) for i=0..7? 
    # Standard Clenshaw-Curtis uses roots of Chebyshev polynomials? 
    # Actually, Clenshaw-Curtis uses extrema of Chebyshev polynomials: cos(k*pi/N).
    # Let's use extrema for better endpoint behavior if needed, but task says nodes in [0,1] strictly increasing.
    # Extrema include 0 and 1. The task says nodes in [0,1]. "Strictly increasing in [0,1]".
    # Usually quadrature nodes are interior. Let's check "strictly increasing in [0,1]".
    # If nodes are 0 or 1, it might be okay, but E(0) is well defined.
    # However, standard Gauss-Legendre is interior.
    # Let's use interior Chebyshev-Lobatto nodes (excluding endpoints) or just map roots.
    # Roots of Chebyshev T_8: cos((2k-1)pi/16) for k=1..8.
    # These are in (-1, 1). Map to (0, 1).
    
    k = np.arange(1, 9)
    c_nodes = np.cos((2*k - 1) * np.pi / 16.0)
    c_nodes = (c_nodes + 1.0) / 2.0
    c_nodes = np.sort(c_nodes) # Ensure increasing
    
    # Weights for Chebyshev-Gauss quadrature
    # w_i = pi / (N * sin^2(theta_i)) ? No, that's for integral of f(cos(theta)).
    # For integral over [0,1], we map x = 0.5(1+t).
    # dx = 0.5 dt.
    # Integral_0^1 f(x) dx = 0.5 Integral_{-1}^1 f(0.5(1+t)) dt.
    # Chebyshev-Gauss quadrature approximates Integral_{-1}^1 g(t) dt = sum w_i g(t_i).
    # Weights w_i = pi / (N * sin^2(theta_i)) where t_i = cos(theta_i).
    # theta_i = (2k-1)pi/(2N).
    thetas = (2*k - 1) * np.pi / 16.0
    c_weights_t = np.pi / (8.0 * np.sin(thetas)**2)
    # Map to [0,1] weights: w_x = 0.5 * w_t
    c_weights = 0.5 * c_weights_t
    c_weights = c_weights / np.sum(c_weights) # Normalize just in case
    
    init_params_cheb = get_params_from_nodes_weights(c_nodes, c_weights)

    # Strategy 2: Gauss-Legendre
    gl_nodes, gl_weights = roots_legendre(8)
    gl_nodes = (gl_nodes + 1.0) / 2.0
    gl_weights = gl_weights / 2.0
    init_params_gl = get_params_from_nodes_weights(gl_nodes, gl_weights)
    
    # Strategy 3: Midpoint
    mid_nodes = (np.arange(8) + 0.5) / 8.0
    mid_weights = np.full(8, 1.0/8.0)
    init_params_mid = get_params_from_nodes_weights(mid_nodes, mid_weights)

    bounds = []
    bounds.append((1e-6, 0.999999)) # u1
    for i in range(7):
        bounds.append((1e-6, 0.999999)) # ds
    for i in range(8):
        bounds.append((-20.0, 20.0)) # log-weights

    best_params = init_params_cheb.copy()
    best_score = objective(init_params_cheb)
    
    # Optimization Loop
    starts = [init_params_cheb, init_params_gl, init_params_mid]
    
    for start_p in starts:
        if time.time() > deadline - 5:
            break
            
        try:
            res = minimize(objective, start_p, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
            
        # Local refinement with Powell if time permits
        if time.time() < deadline - 2 and best_score < 0.1:
             try:
                 res_powell = minimize(objective, best_params, method='Powell', bounds=bounds,
                                       options={'maxiter': 50, 'maxfun': 500, 'xtol': 1e-10, 'ftol': 1e-10})
                 if res_powell.fun < best_score:
                     best_score = res_powell.fun
                     best_params = res_powell.x.copy()
             except Exception:
                 pass

    # Extract final nodes and weights
    nodes, weights = params_to_nodes_weights(best_params)
    if nodes is None:
        # Fallback
        nodes, weights = c_nodes, c_weights

    # Final sanity checks
    if not np.all(np.diff(nodes) > 0) or nodes[0] <= 0 or nodes[-1] >= 1.0:
        nodes, weights = gl_nodes, gl_weights
        
    if not np.all(weights > 0):
        weights = gl_weights

    weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
