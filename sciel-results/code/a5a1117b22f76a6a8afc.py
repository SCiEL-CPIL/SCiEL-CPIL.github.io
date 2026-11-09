import numpy as np
import time
from scipy.special import genlaguerre

def construct():
    start_time = time.time()
    deadline = start_time + 620  # safety margin before 630s kill
    
    # Precompute Laguerre polynomials L_{2k}^{(-1/2)} for k=0..25
    # We need values and derivatives at 0 and at z_i
    # L_n^{(alpha)}(x) can be evaluated; derivative via recurrence or direct
    
    # We'll build the linear system: for each constraint, evaluate g and g'
    # g(x) = sum_{k=0}^{25} c_k * L_{2k}^{(-1/2)}(x)
    # g(0) = 0, g'(0) = 1, and for each z_i: g(z_i)=0, g'(z_i)=0
    
    # Total constraints: 2 + 2*12 = 26, unknowns: 26 c_k. Good.
    
    # Strategy: search over z_i to minimize B
    # Fast surrogate: numerically compute q(x) roots
    
    # Precompute Laguerre polynomials as functions
    alpha = -0.5
    L_funcs = [genlaguerre(2*k, alpha) for k in range(26)]
    # Derivatives: use the fact that d/dx L_n^{(alpha)}(x) = -L_{n-1}^{(alpha+1)}(x)
    # So L_{2k}^{(-1/2)'}(x) = -L_{2k-1}^{(1/2)}(x) for k>=1, and 0 for k=0
    L_deriv_funcs = [None] + [-genlaguerre(2*k-1, 0.5) for k in range(1, 26)]
    
    def eval_g_and_gprime(x, coeffs):
        """Evaluate g(x) and g'(x) given coefficients c_k"""
        g_val = 0.0
        gp_val = 0.0
        for k in range(26):
            g_val += coeffs[k] * L_funcs[k](x)
            if k == 0:
                gp_val += 0.0  # derivative of constant term is 0
            else:
                gp_val += coeffs[k] * L_deriv_funcs[k](x)
        return g_val, gp_val
    
    def compute_B(z_list):
        """Compute B for given z_i values using numerical root finding"""
        z = np.array(z_list)
        # Build linear system A * c = b
        # Rows: g(0)=0, g'(0)=1, then for each z_i: g(z_i)=0, g'(z_i)=0
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        for k in range(26):
            A[0, k] = float(L_funcs[k](0.0))
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        for k in range(26):
            if k == 0:
                A[1, k] = 0.0
            else:
                A[1, k] = float(L_deriv_funcs[k](0.0))
        b[1] = 1.0
        
        # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
        for i, zi in enumerate(z):
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            for k in range(26):
                A[row_g, k] = float(L_funcs[k](zi))
                if k == 0:
                    A[row_gp, k] = 0.0
                else:
                    A[row_gp, k] = float(L_deriv_funcs[k](zi))
            b[row_g] = 0.0
            b[row_gp] = 0.0
        
        # Solve
        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return float('inf')
        
        # Compute q(x) = g(x) / (x * prod_i (x - z_i)^2)
        # q is a polynomial of degree 25. We need to find its roots.
        # Strategy: sample g(x) at many points, divide by the known factor, find roots of q
        
        # Better: construct the polynomial g explicitly? That's hard with Laguerre basis.
        # Alternative: use the fact that q(x) = g(x)/(x*prod(x-z_i)^2) and find where q has roots.
        # We can evaluate q(x) at sample points and use root finding.
        
        # Sample points: we want to find real roots of q. Let's sample on a grid.
        # The roots of q could be anywhere. Let's use a reasonable range.
        # Since z_i are positive, and g has double zeros at z_i, q is smooth elsewhere.
        
        # Let's evaluate q at many points and look for sign changes
        x_min = 0.01
        x_max = max(z) * 2 + 10
        n_samples = 10000
        x_samples = np.linspace(x_min, x_max, n_samples)
        
        q_vals = np.zeros(n_samples)
        for j, x in enumerate(x_samples):
            g_val, _ = eval_g_and_gprime(x, coeffs)
            denom = x
            for zi in z:
                denom *= (x - zi)**2
            if abs(denom) < 1e-300:
                q_vals[j] = 0.0
            else:
                q_vals[j] = g_val / denom
        
        # Find sign changes
        signs = np.sign(q_vals)
        # Replace zeros with previous non-zero sign
        for j in range(1, n_samples):
            if signs[j] == 0:
                signs[j] = signs[j-1]
        
        # Find intervals where sign changes
        root_intervals = []
        for j in range(n_samples - 1):
            if signs[j] != signs[j+1] and signs[j] != 0 and signs[j+1] != 0:
                root_intervals.append((x_samples[j], x_samples[j+1]))
        
        # Refine each root using bisection
        roots = []
        for lo, hi in root_intervals:
            # Bisection
            for _ in range(50):
                mid = (lo + hi) / 2
                g_mid, _ = eval_g_and_gprime(mid, coeffs)
                denom_mid = mid
                for zi in z:
                    denom_mid *= (mid - zi)**2
                if abs(denom_mid) < 1e-300:
                    break
                q_mid = g_mid / denom_mid
                g_lo, _ = eval_g_and_gprime(lo, coeffs)
                denom_lo = lo
                for zi in z:
                    denom_lo *= (lo - zi)**2
                q_lo = g_lo / denom_lo if abs(denom_lo) > 1e-300 else 0.0
                
                if q_lo * q_mid <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        
        if not roots:
            return 0.0
        return max(roots)
    
    # Search strategy: try different configurations of z_i
    # Start with some reasonable guesses and refine
    
    best_B = float('inf')
    best_z = None
    
    # Try a few initial configurations
    configs = []
    
    # Config 1: equally spaced small values
    configs.append(list(np.linspace(0.1, 1.0, 12)))
    
    # Config 2: logarithmically spaced
    configs.append(list(np.logspace(-1, 0, 12)))
    
    # Config 3: all near 1
    configs.append(list(1.0 + 0.01 * i for i in range(12)))
    
    # Config 4: clustered at small values
    configs.append(list(np.linspace(0.01, 0.5, 12)))
    
    # Config 5: geometric progression
    configs.append([0.1 * (1.2**i) for i in range(12)])
    
    # Evaluate each config
    for config in configs:
        if time.time() > deadline - 60:
            break
        B_val = compute_B(config)
        if B_val < best_B:
            best_B = B_val
            best_z = config[:]
    
    # Local refinement: perturb the best config
    if best_z is not None and time.time() < deadline - 30:
        z = best_z[:]
        B_current = best_B
        step_size = 0.01
        for iteration in range(200):
            if time.time() > deadline - 30:
                break
            improved = False
            for i in range(12):
                if time.time() > deadline - 30:
                    break
                # Try increasing z_i
                z_new = z[:]
                z_new[i] += step_size
                if all(zj > 0 for zj in z_new) and len(set(z_new)) == 12:
                    B_new = compute_B(z_new)
                    if B_new < B_current:
                        z = z_new
                        B_current = B_new
                        improved = True
                # Try decreasing z_i
                z_new = z[:]
                z_new[i] -= step_size
                if all(zj > 0 for zj in z_new) and len(set(z_new)) == 12:
                    B_new = compute_B(z_new)
                    if B_new < B_current:
                        z = z_new
                        B_current = B_new
                        improved = True
            if not improved:
                step_size *= 0.5
                if step_size < 1e-6:
                    break
        
        best_z = z
        best_B = B_current
    
    # Return the best z found
    if best_z is None:
        # Fallback: return something valid
        return [0.1 * (i + 1) for i in range(12)]
    
    return best_z

