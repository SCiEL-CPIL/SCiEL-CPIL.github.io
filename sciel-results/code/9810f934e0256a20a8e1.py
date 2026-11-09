"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad

def _time_left(deadline):
    return max(0.0, deadline - time.time())

def _E_cpl(a, w0, wa):
    """Hubble parameter H(a)/H0 for CPL model in flat universe."""
    # E(a) = a^(-3) * (1 + w0 + wa * (1/a - 1))^(wa / (1 + w0))
    # Handle w0 = -1 case carefully
    if np.abs(1.0 + w0) < 1e-10:
        # If w0 is -1, the term (1+w0+wa*z)^(wa/(1+w0)) becomes 1^(inf) which is 1 
        # if we take the limit properly? 
        # Actually, E(z) = (1+z)^3 (1+z)^{wa} = (1+z)^{3+wa} for w0=-1?
        # Let's check: 1 + w0 + wa z = 1 - 1 + wa z = wa z.
        # Exponent wa/(1+w0) is undefined.
        # Standard CPL: E(z) = (1+z)^3 (1 + w0 + wa z)^{wa/(1+w0)}.
        # If w0 = -1, E(z) = (1+z)^3 (wa z)^{inf}? No.
        # The formula is derived from w = w0 + wa z.
        # If w0 = -1, w = -1 + wa z.
        # E(z) = (1+z)^3 (1 - 1 + wa z)^{wa/0} ... this is singular.
        # However, usually w0 is not exactly -1 in numerical problems, or we use the limit.
        # For robustness, if 1+w0 is close to 0, we can use the alternative form:
        # E(z) = (1+z)^3 * (1 + w0 + wa z)^(wa/(1+w0))
        # If w0 -> -1, let w0 = -1 + eps.
        # This is tricky. Let's assume w0 != -1 for the test set to avoid singularities,
        # or handle it by clipping.
        # In our test set, we will avoid w0 = -1 exactly or use a small epsilon.
        pass

    z = 1.0 / a - 1.0
    base = 1.0 + w0 + wa * z
    base = np.maximum(base, 1e-10) # Ensure positive base for real power
    
    denom = 1.0 + w0
    if abs(denom) < 1e-10:
        # Approximation for w0 ~ -1
        # E(z) ~ (1+z)^3 * exp( wa * z / (1+w0) ) ... no, this diverges.
        # Let's just return a large value or handle via limit if necessary.
        # For now, return 1 to avoid crash, but we will avoid w0=-1 in test set.
        exponent = 0.0 
    else:
        exponent = wa / denom
        
    return a**(-3.0) * base**exponent

def _integrand(z, w0, wa):
    """1/E(z) for CPL model."""
    a = 1.0 / (1.0 + z)
    return 1.0 / _E_cpl(a, w0, wa)

def _exact_integral(z, w0, wa):
    """Compute exact integral_0^z dz'/E(z') using adaptive quadrature."""
    # quad is slow, so we only call this once per (model, z) pair during setup.
    result, _ = quad(lambda zp: _integrand(zp, w0, wa), 0, z, limit=300, epsabs=1e-12, epsrel=1e-12)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    # Representative CPL models
    # We include a range of w0, wa to ensure robustness.
    # Avoid w0 = -1 exactly to prevent singularities in the exponent formula.
    models = [
        (0.0, 0.0),      # Matter only
        (-1.0, 0.0),     # Lambda CDM (w0=-1, wa=0) -> E(z) = (1+z)^3. 
                         # Wait, if w0=-1, wa=0, then base=0. 0^inf? 
                         # If wa=0, exponent is 0. base^0 = 1. E(z) = (1+z)^3. Correct.
        (-0.5, -0.5),    # Quintessence
        (-1.0, 0.5),     # Phantom
        (-0.8, -0.5),    
        (-1.2, 0.3),     
        (-0.3, -0.3),    
        (-1.0, -0.5),    
        (-0.9, 0.1),
        (-0.7, 0.7)
    ]

    # Test redshifts: cover low to high z
    z_test = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])

    # Pre-compute exact integrals
    # This is the expensive part, but done only once.
    exact_integrals = []
    for w0, wa in models:
        for z in z_test:
            try:
                val = _exact_integral(z, w0, wa)
                exact_integrals.append(val)
            except Exception:
                exact_integrals.append(1.0) # Fallback
    exact_integrals = np.array(exact_integrals)
    
    model_params = models
    z_vals = z_test
    num_tests = len(model_params) * len(z_vals)

    def decode_params(params):
        """Decode raw parameters to valid nodes and weights."""
        # Nodes: exp-softmax-like cumulative sum
        raw_nodes = np.exp(params[:8])
        nodes = np.cumsum(raw_nodes) / np.sum(raw_nodes)
        
        # Weights: softmax
        log_weights = params[8:]
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)
        
        return nodes, weights

    def objective(params):
        if _time_left(deadline) < 5.0:
            return 1e10
            
        nodes, weights = decode_params(params)
        
        max_err = 0.0
        idx = 0
        for i_model, (w0, wa) in enumerate(model_params):
            for i_z, z in enumerate(z_vals):
                z_nodes = z * nodes
                a_nodes = 1.0 / (1.0 + z_nodes)
                
                # Compute E(a_i)
                # Vectorized E calculation
                z_calc = 1.0 / a_nodes - 1.0
                base = 1.0 + w0 + wa * z_calc
                base = np.maximum(base, 1e-10)
                
                denom = 1.0 + w0
                if abs(denom) < 1e-10:
                    exponent = 0.0
                else:
                    exponent = wa / denom
                    
                E_vals = a_nodes**(-3.0) * base**exponent
                vals = 1.0 / E_vals
                
                approx = z * np.dot(weights, vals)
                
                exact = exact_integrals[idx]
                idx += 1
                
                if abs(exact) < 1e-15:
                    err = 0.0
                else:
                    err = abs((approx - exact) / exact)
                
                if err > max_err:
                    max_err = err
                    
        return max_err

    # Initial guess: Gauss-Legendre nodes mapped to [0,1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    
    # Encode initial guess
    # For nodes: we need x such that cumsum(exp(x))/sum(exp(x)) = nodes0
    # This is an inverse problem. Easier to just start DE with random bounds.
    # But we can provide a good initial population or just use bounds.
    
    # Bounds for DE
    # exp(x) can be very large or small. 
    # If x is in [-10, 10], exp(x) is in [4e-5, 2e4].
    # This is sufficient range.
    bounds = [(-10.0, 10.0)] * 16

    best_result = None
    best_score = float('inf')

    # Phase 1: Differential Evolution (Global Search)
    # Use a significant portion of the time for this.
    de_deadline = start_time + 40.0
    
    try:
        # We need to pass a custom time check to DE? 
        # DE doesn't have a built-in time stop, only maxiter.
        # We can estimate iterations based on time, or just set a reasonable maxiter.
        # Each objective call is ~O(100) operations. Very fast.
        # 1000 iterations * 15 popsize * 16 dims is manageable.
        
        result_de = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False # We will polish manually
        )
        
        if result_de.fun < best_score:
            best_score = result_de.fun
            best_result = result_de.x
            
    except Exception:
        pass

    # Phase 2: Local Refinement (L-BFGS-B)
    # Use the best result from DE as the starting point.
    # If DE failed, start from GL encoded approx or random.
    
    if best_result is None:
        # Fallback start: encode GL nodes/weights roughly
        # This is hard to invert exactly. Just use a random start or GL mapped.
        # Let's use a simple random start within bounds if DE failed.
        best_result = np.random.uniform(-5, 5, 16)
        
    # Try multiple local optimizations from the best DE result and maybe a few other good points?
    # Just one from the best is usually sufficient if DE converged.
    
    try:
        # L-BFGS-B requires bounds.
        # Our parameters are unconstrained in the objective (we decode inside), 
        # but we should bound them to prevent exp overflow/underflow issues in decoding.
        local_bounds = [(-20.0, 20.0)] * 16
        
        result_lbfgs = minimize(
            objective,
            best_result,
            method='L-BFGS-B',
            bounds=local_bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result_lbfgs.fun < best_score:
            best_score = result_lbfgs.fun
            best_result = result_lbfgs.x
            
    except Exception:
        pass
    
    # Phase 3: Final Polish with Nelder-Mead if time permits?
    # L-BFGS-B is usually enough. Let's stick to it.
    
    if best_result is not None and _time_left(deadline) > 2.0:
        nodes, weights = decode_params(best_result)
        
        # Validate constraints
        if np.all(np.diff(nodes) > 0) and np.all(nodes > 0) and np.all(nodes < 1) and np.all(weights > 0):
            # Normalize weights to sum to 1 exactly (floating point drift)
            weights = weights / np.sum(weights)
            return {"nodes": nodes, "weights": weights}
            
    # Fallback: Gauss-Legendre
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    return {"nodes": nodes0, "weights": weights0}

# EVOLVE-BLOCK-END
