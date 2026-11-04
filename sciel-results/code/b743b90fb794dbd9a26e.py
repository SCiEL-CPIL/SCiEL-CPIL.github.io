"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import time

def _compute_c4_bound(roots):
    """
    Compute the C4 upper bound for given double-root locations.
    
    The Cohn-Gonçalves bound for Fourier sign uncertainty with m double roots
    at positions r_1, ..., r_m involves the L^2 norm of the polynomial
    P(x) = prod_i (x - r_i)^2 weighted by e^{-x}.
    
    The bound is: C4 <= sqrt(2*pi) * ||P||_L2^2 / (prod_i r_i)
    
    We compute this via Gauss-Laguerre quadrature for accuracy.
    """
    roots = np.array(roots, dtype=np.float64)
    m = len(roots)
    
    if np.any(roots <= 0):
        return 1e15
    
    # Use Gauss-Laguerre quadrature with 64 points for good accuracy
    nodes, weights = np.polynomial.laguerre.laggauss(64)
    
    # Compute P(x) = prod_i (x - r_i)^2 at each node
    # Use log-sum for numerical stability
    log_p = np.zeros_like(nodes)
    for r in roots:
        log_p += 2.0 * np.log(np.abs(nodes - r))
    
    # P^2 * e^{-x} = exp(2*log_p - x)
    # But we need ||P||^2 = integral of P^2 * e^{-x} dx
    # So the integrand is exp(2*log_p - x)
    
    # For numerical stability, work in log space
    log_integrand = 2.0 * log_p - nodes
    
    # Find max for stability
    log_max = np.max(log_integrand)
    if log_max > 700:
        # Too large, return a large value
        return 1e15
    
    integrand = np.exp(log_integrand - log_max)
    
    norm_sq = np.sum(weights * integrand) * np.exp(2.0 * log_max)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e15
    
    # The bound involves sqrt(2*pi) * norm_sq / prod(roots)
    log_prod = np.sum(np.log(roots))
    if log_prod > 700:
        return 1e15
    
    prod_roots = np.exp(log_prod)
    
    bound = np.sqrt(2.0 * np.pi) * norm_sq / prod_roots
    
    if not np.isfinite(bound) or bound < 0:
        return 1e15
    
    return bound

def _compute_seed_bound():
    """Compute the C4 bound for the seed configuration."""
    seed_roots = [4.0 * (i + 1) for i in range(12)]
    return _compute_c4_bound(seed_roots)

def _solve_potential_critical_points(coeffs, m=12, max_iter=100, tol=1e-10):
    """
    Solve for critical points of potential V(x) = sum_{k=0}^{K} c_k * x^k
    where V'(x) = sum_{k=1}^{K} k*c_k * x^{k-1}
    
    The critical points satisfy: sum_{j != i} 2/(x_i - x_j) + V'(x_i) = 0
    for i = 1, ..., m
    
    We use Newton-Raphson on the system of m equations.
    """
    x = np.zeros(m)
    x[0] = 1.0
    for i in range(1, m):
        x[i] = x[i-1] + 1.0
    
    for iteration in range(max_iter):
        # Compute V'(x_i) for each i
        v_prime = np.zeros(m)
        for i in range(m):
            v_prime[i] = 0.0
            xi = x[i]
            for k in range(1, len(coeffs)):
                if k > 0:
                    v_prime[i] += k * coeffs[k] * (xi ** (k-1))
        
        # Compute the equations: f_i = sum_{j != i} 2/(x_i - x_j) + V'(x_i)
        f = np.zeros(m)
        for i in range(m):
            f[i] = v_prime[i]
            for j in range(m):
                if i != j:
                    f[i] += 2.0 / (x[i] - x[j])
        
        # Check convergence
        if np.max(np.abs(f)) < tol:
            break
        
        # Compute Jacobian
        J = np.zeros((m, m))
        for i in range(m):
            # Diagonal: derivative of sum_{j != i} 2/(x_i - x_j) w.r.t. x_i
            # = sum_{j != i} 2/(x_i - x_j)^2
            # Plus V''(x_i)
            J[i, i] = v_prime[i]  # placeholder, will compute properly
            J[i, i] = 0.0
            for j in range(m):
                if i != j:
                    J[i, i] += 2.0 / (x[i] - x[j]) ** 2
            
            # V''(x_i)
            vp2 = 0.0
            xi = x[i]
            for k in range(2, len(coeffs)):
                if k >= 2:
                    vp2 += k * (k-1) * coeffs[k] * (xi ** (k-2))
            J[i, i] += vp2
            
            # Off-diagonal: derivative of 2/(x_i - x_j) w.r.t. x_j = 2/(x_i - x_j)^2
            for j in range(m):
                if i != j:
                    J[i, j] = 2.0 / (x[i] - x[j]) ** 2
        
        # Solve J * dx = -f
        try:
            dx = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            dx = np.linalg.lstsq(J, -f, rcond=None)[0]
        
        # Damped update
        alpha = 1.0
        for _ in range(20):
            x_new = x + alpha * dx
            if np.all(x_new > 0):
                # Check if it improves
                f_new = np.zeros(m)
                for i in range(m):
                    vp_new = 0.0
                    xi = x_new[i]
                    for k in range(1, len(coeffs)):
                        if k > 0:
                            vp_new += k * coeffs[k] * (xi ** (k-1))
                    f_new[i] = vp_new
                    for j in range(m):
                        if i != j:
                            f_new[i] += 2.0 / (x_new[i] - x_new[j])
                
                if np.max(np.abs(f_new)) < np.max(np.abs(f)):
                    x = x_new
                    break
                alpha *= 0.5
            else:
                alpha *= 0.5
        else:
            # If no improvement, break
            break
    
    return x

def construct():
    """Construct 12 positive double-root locations minimizing the C4 upper bound."""
    start_time = time.time()
    deadline = start_time + 570.0  # 570s safety margin before 630s kill
    
    m = 12
    
    # Seed configuration
    seed_roots = np.array([4.0 * (i + 1) for i in range(m)])
    seed_bound = _compute_c4_bound(seed_roots)
    
    best_roots = seed_roots.copy()
    best_bound = seed_bound
    
    # Strategy 1: Direct optimization with multiple restarts
    try:
        def objective(x):
            if np.any(x <= 0):
                return 1e15
            return _compute_c4_bound(x)
        
        rng = np.random.default_rng(42)
        
        # Initializations
        inits = []
        
        # Seed
        inits.append(seed_roots.copy())
        
        # Geometric spacing
        inits.append(np.geomspace(2.0, 60.0, m))
        
        # Zeros of Laguerre polynomial L_12 (scaled)
        try:
            zeros = roots_laguerre(m)[0]
            inits.append(zeros * 4.0)
            inits.append(zeros * 8.0)
            inits.append(zeros * 2.0)
        except:
            pass
        
        # Random positive initializations
        for _ in range(3):
            x0 = rng.uniform(1.0, 50.0, m)
            x0.sort()
            inits.append(x0)
        
        # Perturbed seed
        for _ in range(2):
            x0 = seed_roots + rng.normal(0, 2.0, m)
            x0 = np.maximum(x0, 0.5)
            x0.sort()
            inits.append(x0)
        
        for idx, x0 in enumerate(inits):
            if time.time() > deadline:
                break
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                                       'adaptive': True})
                
                if np.all(result.x > 0):
                    candidate_bound = result.fun
                    if candidate_bound < best_bound:
                        best_bound = candidate_bound
                        best_roots = result.x.copy()
                        
            except Exception:
                pass
        
    except Exception:
        pass
    
    # Strategy 2: Potential function approach
    # Use a quartic potential: V(x) = c1*x + c2*x^2 + c3*x^3 + c4*x^4
    # The critical points should be positive and well-spaced
    try:
        # Optimize the potential coefficients
        def potential_objective(params):
            if time.time() > deadline:
                return 1e15
            
            # params: [c1, c2, c3, c4]
            coeffs = np.zeros(5)  # indices 0-4, we use 1-4
            coeffs[1] = params[0]
            coeffs[2] = params[1]
            coeffs[3] = params[2]
            coeffs[4] = params[3]
            
            # Solve for critical points
            x = _solve_potential_critical_points(coeffs, m=m, max_iter=50, tol=1e-8)
            
            if np.any(x <= 0) or not np.all(np.isfinite(x)):
                return 1e15
            
            x.sort()
            return _compute_c4_bound(x)
        
        # Try different initial coefficient sets
        rng = np.random.default_rng(123)
        
        for restart in range(8):
            if time.time() > deadline:
                break
            
            if restart == 0:
                # Start near a configuration that gives roughly uniform spacing
                p0 = np.array([0.0, 0.1, 0.01, 0.001])
            elif restart == 1:
                p0 = np.array([0.0, 0.05, 0.02, 0.0005])
            elif restart == 2:
                p0 = np.array([0.0, 0.2, 0.005, 0.002])
            elif restart == 3:
                p0 = np.array([0.0, 0.08, 0.015, 0.0008])
            else:
                p0 = rng.uniform(-0.1, 0.3, 4)
                p0[0] = 0.0  # Keep linear term at 0 for simplicity
            
            try:
                result = minimize(potential_objective, p0, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                       'adaptive': True})
                
                if result.fun < best_bound:
                    # Extract the roots from the best potential
                    coeffs = np.zeros(5)
                    coeffs[1] = result.x[0]
                    coeffs[2] = result.x[1]
                    coeffs[3] = result.x[2]
                    coeffs[4] = result.x[3]
                    
                    x = _solve_potential_critical_points(coeffs, m=m, max_iter=100, tol=1e-10)
                    
                    if np.all(x > 0) and np.all(np.isfinite(x)):
                        x.sort()
                        candidate_bound = _compute_c4_bound(x)
                        if candidate_bound < best_bound:
                            best_bound = candidate_bound
                            best_roots = x.copy()
                            
            except Exception:
                pass
        
    except Exception:
        pass
    
    # Strategy 3: If we have time, do a final local refinement
    if time.time() < deadline - 30:
        try:
            def objective(x):
                if np.any(x <= 0):
                    return 1e15
                return _compute_c4_bound(x)
            
            result = minimize(objective, best_roots, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            
            if np.all(result.x > 0):
                candidate_bound = result.fun
                if candidate_bound < best_bound:
                    best_bound = candidate_bound
                    best_roots = result.x.copy()
                    
        except Exception:
            pass
    
    # Sort the best roots
    best_roots = np.sort(best_roots)
    
    # Final validation
    if not np.all(best_roots > 0) or not np.all(np.isfinite(best_roots)):
        best_roots = seed_roots
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
