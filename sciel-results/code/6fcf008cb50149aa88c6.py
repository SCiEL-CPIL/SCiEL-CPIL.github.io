import numpy as np
import time
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """
    Compute the ratio: L2^2 / (L1 * Sup(AutoConv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s kill
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Variables: x = [h_0, ..., h_{N-1}, M]
    # Objective: Minimize M (x[N])
    # Constraints:
    # 1. sum(h) - 1 == 0
    # 2. h_i >= 0
    # 3. M - conv_k >= 0 for all k
    
    def objective(x):
        return x[N]
        
    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad
        
    def constraints_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        # Return array of constraint violations: M - conv_k
        return M - conv
        
    def constraints_grad(x):
        # Returns a list of gradients for each constraint
        # There are (2*N - 1) inequality constraints + 1 equality constraint
        # SLSQP expects a list of dicts with 'fun' and 'jac'
        # It's more efficient to provide the Jacobian for the whole constraint block
        # But scipy SLSQP usually takes a list of constraint dicts.
        # Let's structure it as a single vector constraint g(x) >= 0 for inequalities and h(x) == 0 for equalities.
        # Actually, scipy minimize with method='SLSQP' accepts constraints as a list of dicts.
        # To provide analytic gradients, we define the Jacobian for the whole constraint vector.
        
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        num_conv = len(conv)
        
        # The constraint vector is [sum(h)-1, M-conv_0, ..., M-conv_{2N-2}]
        # Length 1 + (2N-1)
        
        # We need to return the Jacobian of this vector w.r.t x
        # Shape: (1 + 2N - 1, N + 1)
        
        # This is complex to do in a single 'jac' for a list of constraints in SLSQP interface directly 
        # unless we use a single constraint dict with 'fun' returning vector and 'jac' returning matrix.
        
        pass # We will use the standard SLSQP interface with separate constraints or a combined one.
        # For performance, combining is better.
        
    # Let's define a single constraint function returning a vector of inequalities/equalities
    # SLSQP allows a list of constraints.
    # Constraint 0: Equality sum(h) - 1 = 0
    # Constraint 1..K: Inequalities M - conv_k >= 0
    
    # To use analytic gradients efficiently, we define the Jacobian of the entire constraint vector.
    
    def all_constraints_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        # First element is equality residual
        eq_res = np.sum(h) - 1.0
        # Rest are inequality residuals (must be >= 0)
        ineq_res = M - conv
        return np.concatenate(([eq_res], ineq_res))
        
    def all_constraints_jac(x):
        h = x[:N]
        M = x[N]
        num_conv = 2 * N - 1
        # Jacobian shape: (1 + num_conv, N + 1)
        jac = np.zeros((1 + num_conv, N + 1))
        
        # Row 0: Equality sum(h) - 1 = 0
        # d(sum(h))/dh_j = 1
        jac[0, :N] = 1.0
        # d(sum(h))/dM = 0
        
        # Rows 1..num_conv: Inequalities M - conv_k >= 0
        # d(M - conv_k)/dM = 1
        jac[1:, N] = 1.0
        
        # d(M - conv_k)/dh_j = - d(conv_k)/dh_j
        # d(conv_k)/dh_j = 2 * h_{k-j} (with h_m=0 if out of bounds)
        # We need to compute this for all k, j efficiently.
        # conv_k = sum_i h_i h_{k-i}
        # We can compute the gradient of the whole convolution vector w.r.t h.
        # grad_conv[k, j] = 2 * h_{k-j}
        
        # Let's construct the gradient matrix G where G[k, j] = d(conv_k)/dh_j
        # G has shape (2N-1, N)
        G = np.zeros((2 * N - 1, N))
        for j in range(N):
            # For a fixed j, d(conv_k)/dh_j is non-zero when k-j is in [0, N-1]
            # i.e., k in [j, j+N-1]
            # The value is 2 * h_{k-j}
            # Let's iterate over valid k for this j
            # k ranges from 0 to 2N-2
            # k-j ranges from 0 to N-1 => k from j to j+N-1
            # If j+N-1 > 2N-2, clamp it.
            k_start = j
            k_end = min(j + N, 2 * N - 1)
            if k_start < 2 * N - 1 and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                G[indices_k, j] = 2.0 * h[indices_h]
                
        # jac[1:, :N] = -G
        jac[1:, :N] = -G
        
        return jac
        
    constraints = {
        'type': 'ineq', # SLSQP handles mixed eq/ineq by checking sign? 
        # Actually, SLSQP constraint types are 'eq' or 'ineq'.
        # If we mix them in one vector, we need to be careful.
        # Standard practice: separate them or use 'ineq' for all and handle equality as two inequalities?
        # No, SLSQP supports 'eq' and 'ineq'.
        # If we provide a single dict with 'type': 'ineq', it expects fun(x) >= 0.
        # The equality constraint sum(h)-1=0 is not >= 0.
        # So we must separate them.
    }
    
    # Let's split constraints for SLSQP clarity, but provide analytic grads for each?
    # Or use a single 'ineq' constraint for the peak bounds and a separate 'eq' for sum.
    
    constraint_eq = {
        'type': 'eq',
        'fun': lambda x: np.sum(x[:N]) - 1.0,
        'jac': lambda x: np.concatenate((np.ones(N), [0.0]))
    }
    
    # For the inequality constraints M - conv_k >= 0, we can define a single constraint function
    # that returns the vector of violations, and its Jacobian.
    # SLSQP supports vector-valued constraints.
    
    def ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = np.convolve(h, h, mode='full')
        return M - conv
        
    def ineq_jac(x):
        h = x[:N]
        num_conv = 2 * N - 1
        jac = np.zeros((num_conv, N + 1))
        # d(M - conv_k)/dM = 1
        jac[:, N] = 1.0
        # d(M - conv_k)/dh_j = -2 * h_{k-j}
        G = np.zeros((2 * N - 1, N))
        for j in range(N):
            k_start = j
            k_end = min(j + N, 2 * N - 1)
            if k_start < 2 * N - 1 and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                G[indices_k, j] = 2.0 * h[indices_h]
        jac[:, :N] = -G
        return jac
        
    constraint_ineq = {
        'type': 'ineq',
        'fun': ineq_fun,
        'jac': ineq_jac
    }
    
    bounds = [(0.0, 1.0)] * N + [(0.0, None)]
    
    # Initializations
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse random (Sidon-like)
    np.random.seed(42)
    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Gaussian
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((np.arange(N) - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 4. Triangular
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(np.arange(N) - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)

    for h_init in inits:
        if time.time() > deadline - 10:
            break
            
        x0 = np.concatenate((h_init, [1.0])) # Initial M guess
        
        try:
            res = opt.minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=[constraint_eq, constraint_ineq],
                jac=objective_grad,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if res.success:
                h_candidate = res.x[:N]
                # Ensure non-negativity and normalization (SLSQP might drift slightly)
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception as e:
            pass
            
    return list(best_h)
