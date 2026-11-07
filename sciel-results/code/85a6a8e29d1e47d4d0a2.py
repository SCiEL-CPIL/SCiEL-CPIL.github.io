"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
N = 100
import numpy as np
import time
import math
import sys

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import linprog

def construct():
    """
    Solves the Erdos minimum overlap problem using Linear Programming.
    
    The objective is to minimize max_k C_k(h), where C_k(h) = sum_i h_i * (1 - h_{i+k}).
    We introduce an auxiliary variable t such that C_k(h) <= t for all k.
    We minimize t subject to:
    1. C_k(h) - t <= 0 for all k
    2. sum(h) = N/2
    3. 0 <= h_i <= 1
    
    Variables: x = [h_0, ..., h_{N-1}, t]
    Objective: minimize [0, ..., 0, 1] dot x
    
    This is a Linear Program (LP) because the constraints are linear in h and t.
    Note: C_k(h) is linear in h for fixed k.
    C_k(h) = sum_{i} h_i - sum_i h_i h_{i+k}  <-- Wait, C_k is cross-correlation of h and 1-h.
    C_k(h) = sum_i h_i * (1 - h_{i+k}) = sum_i h_i - sum_i h_i h_{i+k}.
    The term sum_i h_i h_{i+k} is QUADRATIC in h. 
    Therefore, this is NOT a Linear Program. It is a Quadratic Program (QP) or Semi-Definite Program (SDP) if relaxed.
    
    However, the current task contract says "upper bound on constant C5".
    The seed program uses SA because it's non-convex/non-linear.
    
    Let's re-read the inspirations.
    Inspiration 1 & 2 suggest Convex QP or SLSQP.
    Inspiration 3 suggests L-BFGS-B.
    
    My previous thought that it's an LP was wrong because of the h_i * h_j terms.
    C_k(h) = sum_i h_i - sum_i h_i h_{i+k}.
    Minimizing max_k (linear - quadratic) is non-convex.
    
    However, we can reformulate.
    Let u_i = h_i.
    We want to minimize t s.t. sum_i u_i - sum_i u_i u_{i+k} <= t.
    This is non-convex.
    
    Alternative Strategy:
    The problem is equivalent to finding a sequence h with mean 0.5 that minimizes the maximum cross-correlation with its complement.
    
    Let's look at the structure.
    If we use `scipy.optimize.minimize` with method='SLSQP' or 'trust-constr', we can handle the non-smooth max by introducing t and using gradient-based methods on the smooth approximation or directly handling the constraints if we treat the max as a set of inequalities.
    
    Since the objective involves products of variables, it's a Non-Convex Quadratic Program (NQP).
    However, for N=100, we can try to use a convex relaxation or simply a robust local solver like SLSQP with multiple restarts, or better, a global solver like Differential Evolution to find a good starting point for a local solver.
    
    But the directive asks for a STRUCTURAL change. The current program is SA.
    The inspirations mention Convex QP (which might be a relaxation) or SLSQP.
    
    Let's try a different structural approach: **Alternating Minimization** or **Projected Gradient Descent** on the smooth approximation of the max function (Log-Sum-Exp).
    Or, even better, exploit the linearity of the constraints if we fix the non-linear part? No.
    
    Let's try **SLSQP with auxiliary variable t**.
    Minimize t
    Subject to:
    sum(h) = N/2
    0 <= h <= 1
    C_k(h) <= t for all k
    
    This is a standard Non-Linear Programming (NLP) problem. SLSQP is designed for this.
    To avoid local minima, we will run SLSQP from multiple diverse initial points.
    
    Why is this different?
    1. SA is stochastic local search.
    2. SLSQP is deterministic gradient-based local search.
    3. We will use a "Divide and Conquer" or "Multi-start" strategy with specifically designed initializations (e.g., periodic structures, random, constant) to feed into SLSQP.
    
    Let's implement Multi-start SLSQP.
    """
    start_time = time.time()
    time_limit = 145.0
    
    # Pre-compute correlation indices
    # np.correlate(h, 1-h, mode='full') has length 2*N - 1
    # Indices 0 to 2*N-2 correspond to lags -(N-1) to N-1
    # We need C_k(h) for all k.
    
    def compute_correlations(h):
        # Returns array of length 2*N-1
        # corr[k] corresponds to lag k - (N-1)
        return np.correlate(h, 1.0 - h, mode="full")
    
    def objective_and_grads(x):
        """
        x = [h_0, ..., h_{N-1}, t]
        Returns (t, grad)
        """
        h = x[:N]
        t = x[N]
        
        # For SLSQP, we define the objective as t.
        # Constraints will be handled separately.
        return t, np.concatenate([np.zeros(N), [1.0]])
        
    def constraint_defs():
        """
        Define constraints for SLSQP.
        SLSQP takes a list of dictionaries.
        """
        constraints = []
        
        # Equality constraint: sum(h) = N/2
        # h is x[0:N]
        def sum_eq(x):
            return np.sum(x[:N]) - N / 2.0
        
        def sum_eq_grad(x):
            grad = np.zeros(N + 1)
            grad[:N] = 1.0
            return grad
            
        constraints.append({
            'type': 'eq',
            'fun': sum_eq,
            'jac': sum_eq_grad
        })
        
        # Inequality constraints: C_k(h) - t <= 0
        # We need to add one constraint per lag k.
        # There are 2*N - 1 lags.
        for k in range(2 * N - 1):
            # Create a closure for the k-th constraint
            def make_ineq(k_idx):
                def fun(x):
                    h = x[:N]
                    t = x[N]
                    corr = np.correlate(h, 1.0 - h, mode="full")
                    return corr[k_idx] - t
                
                def jac(x):
                    h = x[:N]
                    # Gradient of C_k(h) w.r.t h_i
                    # C_k(h) = sum_i h_i (1 - h_{i+k})
                    # dC_k/dh_i = (1 - h_{i+k}) + h_i * (-1) * d(h_{i+k})/dh_i
                    # d(h_{i+k})/dh_i is 1 if i+k == j? No.
                    # Let's derive carefully.
                    # C_k(h) = sum_{i=0}^{N-1} h_i (1 - h_{i+k}) 
                    # Note: indices are wrapped? No, np.correlate handles bounds.
                    # For a specific lag k (shifted by N-1), the correlation is:
                    # sum_{i} h_i * (1 - h_{i + (k - (N-1))})
                    # Let shift = k - (N-1)
                    # Term is non-zero only if i and i+shift are in [0, N-1].
                    
                    shift = k_idx - (N - 1)
                    grad = np.zeros(N + 1)
                    
                    # d/dh_j [ sum_i h_i (1 - h_{i+shift}) ]
                    # = sum_i [ (delta_{ij} (1 - h_{i+shift}) + h_i (-delta_{i+shift, j}) ) ]
                    # = (1 - h_{j+shift}) - h_{j-shift}  (with appropriate bounds)
                    
                    # Term 1: j is the index of h_i. i=j.
                    # Valid if 0 <= j < N and 0 <= j+shift < N.
                    if 0 <= j < N: # This logic is inside the loop over j? No, we compute vectorized.
                        pass
                    
                    # Let's do it vectorized for speed in jac.
                    # grad_h = (1 - h_shifted_right) - h_shifted_left
                    # where h_shifted_right[j] = h[j+shift] if valid, else 0
                    # and h_shifted_left[j] = h[j-shift] if valid, else 0
                    
                    h_right = np.zeros(N)
                    h_left = np.zeros(N)
                    
                    # j+shift in [0, N-1] => j in [-shift, N-1-shift]
                    if shift >= 0:
                        # j in [0, N-1-shift]
                        h_right[:N-shift] = h[shift:]
                    else:
                        # j in [-shift, N-1]
                        h_right[-shift:] = h[:N+shift]
                        
                    # j-shift in [0, N-1] => j in [shift, N-1+shift]
                    if shift >= 0:
                        # j in [shift, N-1]
                        h_left[shift:] = h[:N-shift]
                    else:
                        # j in [0, N-1+shift]
                        h_left[:N+shift] = h[-shift:]
                        
                    grad[:N] = (1.0 - h_right) - h_left
                    grad[N] = -1.0 # d/dt (C_k - t) = -1
                    
                    return grad
                return fun, jac
            
            fun_k, jac_k = make_ineq(k)
            constraints.append({
                'type': 'ineq',
                'fun': fun_k,
                'jac': jac_k
            })
            
        return constraints

    cons = constraint_defs()
    
    def run_slsqp(h_init):
        x0 = np.concatenate([h_init, [0.5]]) # Initial guess for t
        try:
            res = minimize(
                objective_and_grads,
                x0,
                method='SLSQP',
                bounds=[(0, 1)] * N + [(0, 2)], # t can be up to 2? Max corr is N/2=50, t=50? No, value is scaled.
                # Wait, objective is t. C_k is sum of products. Max sum is 50. So t is around 25-50.
                # Let's set bound for t to (0, 100) to be safe.
                # Actually, value = max_corr / N * 2.
                # If max_corr is 25, value is 0.5.
                # t represents max_corr. So t is around 25.
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            if res.success:
                h_opt = res.x[:N]
                t_opt = res.x[N]
                # Verify validity
                if np.all(np.isfinite(h_opt)) and np.all(h_opt >= -1e-9) and np.all(h_opt <= 1 + 1e-9):
                    h_opt = np.clip(h_opt, 0, 1)
                    # Re-project to sum exactly? SLSQP should handle it, but let's check.
                    if abs(np.sum(h_opt) - N/2.0) > 1e-6:
                         # Simple projection
                         h_opt += (N/2.0 - np.sum(h_opt)) / N
                         h_opt = np.clip(h_opt, 0, 1)
                    val = np.max(np.correlate(h_opt, 1.0 - h_opt, mode="full")) / N * 2.0
                    return h_opt, val
            return None, None
        except Exception:
            return None, None

    best_h = np.full(N, 0.5)
    best_val = np.max(np.correlate(best_h, 1.0 - best_h, mode="full")) / N * 2.0
    
    # Generate diverse initializations
    inits = []
    
    # 1. Constant
    inits.append(np.full(N, 0.5))
    
    # 2. Alternating 0, 1
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    inits.append(h_alt)
    
    # 3. Step
    h_step = np.zeros(N)
    h_step[N//2:] = 1.0
    inits.append(h_step)
    
    # 4. Random
    np.random.seed(42)
    h_rand = np.random.uniform(0, 1, N)
    h_rand += (N/2.0 - np.sum(h_rand)) / N
    h_rand = np.clip(h_rand, 0, 1)
    inits.append(h_rand)
    
    # 5. Sinusoidal
    t = np.linspace(0, 4*np.pi, N)
    h_sin = 0.5 + 0.5 * np.sin(t)
    h_sin += (N/2.0 - np.sum(h_sin)) / N
    h_sin = np.clip(h_sin, 0, 1)
    inits.append(h_sin)
    
    # Run SLSQP on each
    for i, h_init in enumerate(inits):
        if time.time() - start_time > time_limit - 5.0:
            break
            
        h_new, val_new = run_slsqp(h_init)
        
        if h_new is not None and val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            
        # If we have time, we can try to refine the best found so far with more iterations
        if i == 0: # After first run, if good, maybe refine?
            pass

    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
