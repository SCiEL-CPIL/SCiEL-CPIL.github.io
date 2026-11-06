<<<<<<< SEARCH
def construct():
    # Diverse cosmology sample for design (not validation)
    design_cosmologies = [
        (-0.5, 0.0),    # constant w
        (-0.7, 0.0),
        (-0.9, 0.0),
        (-1.0, 0.0),    # cosmological constant
        (-0.5, 0.5),
        (-0.7, 0.3),
        (-0.9, 0.1),
        (-0.3, 0.7),
        (-1.1, -0.1),
        (-0.6, 0.4),
        (-0.8, 0.2),
        (-0.4, 0.6),
    ]
    
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin
    
    # Strategy: Basis-function projection + node optimization
    # We'll try several node families and find optimal weights for each
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # Define node families to try
    node_families = []
    
    # 1. Beta-distribution nodes (various alpha, beta)
    for alpha in [1.5, 2.0, 2.5, 3.0]:
        for beta in [1.5, 2.0, 2.5, 3.0]:
            if alpha + beta > 1:
                from scipy.stats import beta as beta_dist
                # Generate 8 nodes from beta distribution, scaled to [0,1]
                # Use quantiles for better spread
                probs = np.linspace(0.1, 0.9, 8)
                nodes = beta_dist.ppf(probs, alpha, beta)
                node_families.append(nodes)
    
    # 2. Power-law nodes
    for p in [0.7, 1.0, 1.3, 1.5, 2.0]:
        nodes = np.linspace(0, 1, 8) ** p
        nodes[0] = 0.05  # avoid zero
        node_families.append(nodes)
    
    # 3. Log-spaced nodes
    for base in [1.2, 1.3, 1.5]:
        nodes = 1.0 / (base ** np.linspace(0, 7, 8))
        nodes = nodes / nodes.max()  # normalize to [0,1]
        node_families.append(nodes)
    
    # 4. Uniform nodes
    node_families.append((np.arange(8) + 0.5) / 8.0)
    
    # 5. Gauss-Legendre nodes mapped to [0,1]
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1) / 2
    node_families.append(gl_nodes)
    
    # 6. Clenshaw-Curtis nodes
    from numpy.polynomial import chebyshev as C
    cc_nodes = 0.5 * (1 + np.cos(np.pi * np.arange(8) / 7))
    cc_nodes = cc_nodes[::-1]  # reverse to get increasing
    node_families.append(cc_nodes)
    
    # For each node family, find optimal positive weights
    for nodes in node_families:
        if time.time() > deadline:
            break
        
        # Ensure nodes are strictly increasing in (0,1)
        nodes = np.sort(nodes)
        if nodes[0] <= 0 or nodes[-1] >= 1:
            nodes = np.clip(nodes, 0.01, 0.99)
        if np.any(np.diff(nodes) <= 0):
            continue
        
        # Find optimal weights using constrained optimization
        # Weights must be positive and sum to 1
        
        def objective(weights):
            # Soft constraint: penalty for non-positive weights or sum != 1
            w = np.exp(weights)
            w = w / w.sum()  # normalize
            return worst_error(nodes, w, design_cosmologies, (0.2, 2.5), 10)
        
        # Initial guess: uniform
        x0 = np.zeros(8)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            w = np.exp(result.x)
            w = w / w.sum()
            
            err = worst_error(nodes, w, design_cosmologies, (0.2, 2.5), 10)
            
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = w.copy()
                
        except Exception:
            pass
    
    # If we found a good solution, return it
    if best_nodes is not None and best_err < 5.0:
        # Final refinement with more redshift points
        if time.time() < deadline - 10:
            # Refine with more z points
            def refine_objective(params):
                nodes_r = params[:8]
                weights_r = np.exp(params[8:])
                weights_r = weights_r / weights_r.sum()
                return worst_error(nodes_r, weights_r, design_cosmologies, (0.1, 3.0), 15)
            
            x0 = np.concatenate([best_nodes, np.log(best_weights)])
            
            try:
                result = minimize(refine_objective, x0, method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11})
                
                nodes_r = result.x[:8]
                weights_r = np.exp(result.x[8:])
                weights_r = weights_r / weights_r.sum()
                
                # Ensure nodes are valid
                if np.all(np.diff(nodes_r) > 0) and nodes_r[0] > 0 and nodes_r[-1] < 1:
                    err_r = worst_error(nodes_r, weights_r, design_cosmologies, (0.1, 3.0), 15)
                    if err_r < best_err:
                        best_nodes = nodes_r
                        best_weights = weights_r
                        best_err = err_r
                        
            except Exception:
                pass
        
        return {"nodes": best_nodes, "weights": best_weights}
    
    # Fallback: composite midpoint
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
=======
def construct():
    # Diverse cosmology sample for design (not validation)
    design_cosmologies = [
        (-0.5, 0.0),
        (-0.7, 0.0),
        (-0.9, 0.0),
        (-1.0, 0.0),
        (-0.5, 0.5),
        (-0.7, 0.3),
        (-0.9, 0.1),
        (-0.3, 0.7),
        (-1.1, -0.1),
        (-0.6, 0.4),
        (-0.8, 0.2),
        (-0.4, 0.6),
    ]
    
    start_time = time.time()
    deadline = start_time + 75.0  # 75s deadline, 15s safety margin
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # Define node families to try
    node_families = []
    
    # 1. Gauss-Legendre nodes mapped to [0,1]
    gl_nodes, _ = np.polynomial.legendre.leggauss(8)
    gl_nodes = (gl_nodes + 1) / 2
    node_families.append(gl_nodes)
    
    # 2. Clenshaw-Curtis nodes
    cc_nodes = 0.5 * (1 + np.cos(np.pi * np.arange(8) / 7))
    cc_nodes = cc_nodes[::-1]
    node_families.append(cc_nodes)
    
    # 3. Uniform nodes
    node_families.append((np.arange(8) + 0.5) / 8.0)
    
    # 4. Beta-distribution nodes
    from scipy.stats import beta as beta_dist
    for alpha, beta in [(1.5, 1.5), (2.0, 1.5), (1.5, 2.0), (2.0, 2.0), (2.5, 1.5), (1.5, 2.5)]:
        probs = np.linspace(0.05, 0.95, 8)
        nodes = beta_dist.ppf(probs, alpha, beta)
        node_families.append(nodes)
    
    # 5. Power-law nodes
    for p in [0.8, 1.0, 1.2, 1.5]:
        nodes = np.linspace(0.05, 1.0, 8) ** p
        nodes = np.sort(nodes)
        if nodes[0] > 0 and nodes[-1] < 1:
            node_families.append(nodes)
    
    # For each node family, find optimal positive weights using smooth surrogate
    for nodes in node_families:
        if time.time() > deadline - 20:
            break
        
        nodes = np.sort(nodes)
        if nodes[0] <= 0 or nodes[-1] >= 1:
            nodes = np.clip(nodes, 0.01, 0.99)
        if np.any(np.diff(nodes) <= 0):
            continue
        
        # Use smooth surrogate for optimization
        def soft_objective(log_weights):
            w = np.exp(log_weights)
            w = w / w.sum()
            return soft_worst_error(nodes, w, design_cosmologies, (0.2, 2.5), 10, beta=50.0)
        
        x0 = np.zeros(8)
        
        try:
            result = minimize(soft_objective, x0, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            w = np.exp(result.x)
            w = w / w.sum()
            
            err = worst_error(nodes, w, design_cosmologies, (0.2, 2.5), 10)
            
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = w.copy()
                
        except Exception:
            pass
    
    # Joint refinement of nodes and weights if time permits
    if best_nodes is not None and time.time() < deadline - 15:
        def joint_soft_objective(params):
            nodes_r = params[:8]
            # Enforce strict ordering with soft barrier
            diffs = np.diff(nodes_r)
            if np.any(diffs <= 0.001):
                return 1e6
            if nodes_r[0] <= 0.001 or nodes_r[-1] >= 0.999:
                return 1e6
            weights_r = np.exp(params[8:])
            weights_r = weights_r / weights_r.sum()
            return soft_worst_error(nodes_r, weights_r, design_cosmologies, (0.1, 3.0), 12, beta=50.0)
        
        x0 = np.concatenate([best_nodes, np.log(best_weights)])
        
        try:
            result = minimize(joint_soft_objective, x0, method='L-BFGS-B',
                             options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-11})
            
            nodes_r = result.x[:8]
            weights_r = np.exp(result.x[8:])
            weights_r = weights_r / weights_r.sum()
            
            if np.all(np.diff(nodes_r) > 0.001) and nodes_r[0] > 0.001 and nodes_r[-1] < 0.999:
                err_r = worst_error(nodes_r, weights_r, design_cosmologies, (0.1, 3.0), 12)
                if err_r < best_err:
                    best_nodes = nodes_r
                    best_weights = weights_r
                    best_err = err_r
                    
        except Exception:
            pass
    
    # Final high-precision polish
    if best_nodes is not None and time.time() < deadline - 5:
        def polish_objective(params):
            nodes_r = params[:8]
            diffs = np.diff(nodes_r)
            if np.any(diffs <= 0.001) or nodes_r[0] <= 0.001 or nodes_r[-1] >= 0.999:
                return 1e6
            weights_r = np.exp(params[8:])
            weights_r = weights_r / weights_r.sum()
            return worst_error(nodes_r, weights_r, design_cosmologies, (0.1, 3.0), 15)
        
        x0 = np.concatenate([best_nodes, np.log(best_weights)])
        
        try:
            result = minimize(polish_objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            
            nodes_r = result.x[:8]
            weights_r = np.exp(result.x[8:])
            weights_r = weights_r / weights_r.sum()
            
            if np.all(np.diff(nodes_r) > 0.001) and nodes_r[0] > 0.001 and nodes_r[-1] < 0.999:
                err_r = worst_error(nodes_r, weights_r, design_cosmologies, (0.1, 3.0), 15)
                if err_r < best_err:
                    best_nodes = nodes_r
                    best_weights = weights_r
                    best_err = err_r
                    
        except Exception:
            pass
    
    if best_nodes is not None:
        # Ensure weights sum to 1 within 1e-12
        best_weights = best_weights / best_weights.sum()
        return {"nodes": best_nodes, "weights": best_weights}
    
    # Fallback: composite midpoint
    n = 8
    return {"nodes": (np.arange(n) + 0.5) / n, "weights": np.full(n, 1.0 / n)}
>>>>>>> REPLACE
