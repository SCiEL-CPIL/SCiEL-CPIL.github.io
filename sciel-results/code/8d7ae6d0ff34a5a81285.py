import numpy as np
import scipy.optimize
import time

# EVOLVE-BLOCK-START

def _E_cpl(z, w0, w1, Om):
    """
    Compute E(z) = H(z)/H0 for flat CPL model.
    z: redshift
    w0, w1: CPL parameters
    Om: Omega_m
    """
    # z >= 0
    z_safe = np.maximum(z, 1e-12)
    one_plus_z = 1.0 + z_safe
    
    # E(z)^2 = Om(1+z)^3 + (1-Om)(1+z)^(3(1+w0)) * exp(3*w1*(1 - 1/(1+z)))
    # Note: 1 - 1/(1+z) = z/(1+z)
    # Let's use the log form for stability:
    # 3(1+w0) ln(1+z) + 3*w1 * (z/(1+z))
    
    # Alternative stable form:
    # E(z)^2 = Om * (1+z)^3 + (1-Om) * exp( 3*(1+w0)*ln(1+z) + 3*w1*(z/(1+z)) )
    
    # Actually, the standard integral result is:
    # E(a)^2 = Om a^-3 + ODE * exp(3 * Int_{1}^{a} (1+w(a'))/a' da')
    # where w(a') = w0 + w1(1-a')
    # Int = (1+w0) ln(a) + w1 (a - ln(a))  ... wait, limits?
    # Int from 1 to a:
    # ∫ (1+w0+w1-w1 a')/a' da' = (1+w0+w1) ln(a) - w1 (a-1)
    # So E(a)^2 = Om a^-3 + ODE * exp( 3 * [ (1+w0+w1) ln(a) - w1 (a-1) ] )
    # In terms of z, a = 1/(1+z)
    # ln(a) = -ln(1+z)
    # a-1 = -z/(1+z)
    # Exponent = 3 * [ -(1+w0+w1) ln(1+z) + w1 z/(1+z) ]
    # = -3(1+w0+w1) ln(1+z) + 3 w1 z/(1+z)
    
    # Let's stick to z formulation directly to avoid confusion.
    # E(z)^2 = Om (1+z)^3 + (1-Om) (1+z)^{3(1+w0)} exp(3 w1 (1 - 1/(1+z)))
    # Wait, the exponent term:
    # ∫_0^z (1+w(z'))/(1+z') dz'
    # w(z') = w0 + w1 (1 - 1/(1+z')) = w0 + w1 z'/(1+z')
    # 1+w(z') = 1+w0 + w1 z'/(1+z')
    # (1+w(z'))/(1+z') = (1+w0)/(1+z') + w1/(1+z')^2
    # Integral = (1+w0) ln(1+z) - w1 [ 1/(1+z') ]_0^z = (1+w0) ln(1+z) - w1 (1/(1+z) - 1)
    # = (1+w0) ln(1+z) + w1 (1 - 1/(1+z))
    # = (1+w0) ln(1+z) + w1 (z/(1+z))
    
    # So E(z)^2 = Om (1+z)^3 + (1-Om) exp( 3 * [ (1+w0) ln(1+z) + w1 (z/(1+z)) ] )
    # = Om (1+z)^3 + (1-Om) (1+z)^{3(1+w0)} exp( 3 w1 z/(1+z) )
    
    term_m = Om * one_plus_z**3
    term_de = (1.0 - Om) * one_plus_z**(3.0 * (1.0 + w0)) * np.exp(3.0 * w1 * z_safe / one_plus_z)
    
    return np.sqrt(term_m + term_de)

def _exact_integral(z, w0, w1, Om):
    """
    Compute integral I = int_0^z dz'/E(z')
    """
    if z == 0:
        return 0.0
    f = lambda zp: 1.0 / _E_cpl(zp, w0, w1, Om)
    # Use quad for high accuracy
    result, _ = scipy.integrate.quad(f, 0, z, limit=100)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n_nodes = 8

    # Training Set: (w0, w1, Om)
    # We want to cover a reasonable range of dark energy models.
    # Validation will likely use similar or slightly different parameters.
    # The task says "disjoint public validation cosmologies", implying we shouldn't overfit to specific points but capture the manifold.
    models = [
        (-1.0, 0.0, 0.3),  # LambdaCDM
        (-1.2, 0.0, 0.3),  # Phantom
        (-0.8, 0.0, 0.3),  # Quintessence
        (-1.0, -0.5, 0.3), # CPL negative wa
        (-1.0, 0.5, 0.3),  # CPL positive wa
        (-0.8, -1.0, 0.3), # Extreme CPL
        (-1.2, 1.0, 0.3),  # Extreme CPL
        (-1.0, 0.0, 0.2),  # Low Om
        (-1.0, 0.0, 0.35), # High Om
        (-0.9, 0.5, 0.25), # Mixed
        (-1.1, -0.3, 0.28),# Mixed
        (-0.9, -0.8, 0.32),# Mixed
    ]
    
    # Redshifts for training
    # We need to cover the full range z in [0, 10] or so.
    # The integrand 1/E(z) changes character.
    # At low z, matter dominated. At high z, matter dominated.
    # In between, dark energy matters.
    z_train = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0])

    n_models = len(models)
    n_z = len(z_train)

    # Precompute exact integrals
    exact_vals = np.zeros((n_models, n_z))
    for i, (w0, w1, Om) in enumerate(models):
        for j, z in enumerate(z_train):
            try:
                exact_vals[i, j] = _exact_integral(z, w0, w1, Om)
            except:
                exact_vals[i, j] = np.nan

    # Filter out NaNs
    valid_mask = ~np.isnan(exact_vals)
    
    if not np.any(valid_mask):
        # Fallback
        nodes = (np.arange(n_nodes) + 0.5) / n_nodes
        weights = np.ones(n_nodes) / n_nodes
        return {"nodes": nodes, "weights": weights}

    # We will optimize nodes and weights simultaneously using a robust method.
    # Strategy:
    # 1. Use log-parameterization for nodes to ensure strict increase and bounds.
    #    Let nodes be defined by cumulative sums of positive differences.
    #    x_0, d_1, ..., d_7 where x_0 > 0, d_i > 0, sum(x_0 + d_i) < 1.
    #    Actually, simpler: define 8 parameters u_i in R.
    #    x_i = sigmoid(u_i). Sort them.
    #    Weights: define 8 parameters v_i in R.
    #    w_i = softmax(v_i).
    
    # However, sorting sigmoid outputs breaks differentiability at ties, but ties are measure zero.
    # A better parameterization for strictly increasing nodes in [0,1]:
    # Let t_i be 8 parameters.
    # nodes = cumulative sum of softmax(t) * (1 - 2*margin) + margin
    # This ensures they are in [margin, 1-margin] and strictly increasing (if softmax outputs are positive).
    # But softmax outputs are positive, so cumulative sum is strictly increasing.
    # Let's use this.
    
    def _params_to_nodes_weights(params):
        # params: 16 floats
        # First 8: t for node spacing
        # Next 8: v for weights
        t = params[:n_nodes]
        v = params[n_nodes:]
        
        # Nodes
        # Spacing proportions
        probs = np.exp(t - np.max(t))
        probs /= np.sum(probs)
        
        # Total span
        margin = 1e-6
        span = 1.0 - 2.0 * margin
        # Cumulative sum
        cum_probs = np.cumsum(probs)
        # Normalize cum_probs to [0, 1]
        cum_probs = cum_probs / cum_probs[-1]
        nodes = margin + span * cum_probs
        
        # Weights
        w = np.exp(v - np.max(v))
        w /= np.sum(w)
        
        return nodes, w

    def _objective(params):
        # Check time
        if time.time() > deadline:
            return 1e10
        
        nodes, weights = _params_to_nodes_weights(params)
        
        # Compute approximate integrals
        max_err = 0.0
        for i in range(n_models):
            if not valid_mask[i, :].all():
                continue
            w0, w1, Om = models[i]
            for j in range(n_z):
                if not valid_mask[i, j]:
                    continue
                z = z_train[j]
                # a_i = 1 / (1 + z * nodes[i])
                # But we are integrating in z-space: int_0^z dz'/E(z')
                # Approx: z * sum_k weights[k] / E(z * nodes[k])
                # Note: nodes are in [0,1]. z*nodes are in [0, z].
                # This matches the change of variable z' = z * x.
                
                z_vals = z * nodes
                E_vals = _E_cpl(z_vals, w0, w1, Om)
                inv_E = 1.0 / E_vals
                approx_val = z * np.dot(weights, inv_E)
                
                exact_val = exact_vals[i, j]
                if exact_val == 0:
                    err = 0.0
                else:
                    err = abs(approx_val - exact_val) / exact_val
                
                if err > max_err:
                    max_err = err
        
        return max_err

    # Initial guess
    # Gauss-Legendre nodes mapped to [0,1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_nodes)
    nodes_gl = (x_gl + 1.0) / 2.0
    nodes_gl = np.sort(nodes_gl)
    
    # Invert node parameterization
    # nodes = margin + span * cum_probs
    # cum_probs = (nodes - margin) / span
    # probs = diff(cum_probs) with first element being cum_probs[0]
    margin = 1e-6
    span = 1.0 - 2.0 * margin
    cum_probs = (nodes_gl - margin) / span
    # Ensure cum_probs is strictly increasing and in [0,1]
    cum_probs = np.clip(cum_probs, 1e-10, 1.0 - 1e-10)
    probs = np.diff(cum_probs, prepend=0.0)
    # Invert softmax: t = log(probs)
    t_init = np.log(probs)
    
    # Invert weight parameterization
    # weights = softmax(v)
    # v = log(weights)
    w_init = w_gl / np.sum(w_gl) # Normalize just in case
    v_init = np.log(w_init)
    
    params_init = np.concatenate([t_init, v_init])
    
    # Optimize
    best_params = params_init.copy()
    best_score = _objective(params_init)
    
    # Use Nelder-Mead first as it is robust to non-differentiability (sorting etc, though we avoided sorting in param space)
    # Actually, our param space is smooth.
    # Let's try L-BFGS-B as well.
    
    # Time check
    if time.time() < deadline - 20:
        try:
            res = scipy.optimize.minimize(
                _objective, 
                params_init, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if time.time() < deadline - 10:
        try:
            res = scipy.optimize.minimize(
                _objective, 
                best_params, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final refinement with Nelder-Mead again
    if time.time() < deadline - 5:
        try:
            res = scipy.optimize.minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    nodes_final, weights_final = _params_to_nodes_weights(best_params)
    
    # Ensure strict increasing and positive weights
    # Nodes from cumulative sum of positive probs are strictly increasing.
    # Weights from softmax are positive.
    # Sum of weights is 1.
    
    # Clip nodes to [0,1] just in case of floating point issues
    nodes_final = np.clip(nodes_final, 1e-12, 1.0 - 1e-12)
    
    return {"nodes": nodes_final, "weights": weights_final}

# EVOLVE-BLOCK-END
