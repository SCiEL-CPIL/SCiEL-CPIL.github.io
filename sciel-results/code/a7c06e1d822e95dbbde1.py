import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 400

def construct():
    start_time = time.time()
    # Safety margin: stop 10 seconds before the 150s limit
    deadline = start_time + 140.0

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    # 1. Find a good initial guess and the active lag k
    # The current best strategy found a ramp-like structure.
    # Let's start with a linear ramp and find its max autocorrelation lag.
    h_init = np.arange(1, N + 1, dtype=np.float64)
    
    def get_active_lag(h):
        c = np.convolve(h, h)
        # np.max returns the maximum value. We need the index.
        # The convolution length is 2N - 1.
        # The index corresponds to lag k where c[k] is max.
        # Note: np.convolve full mode: c[k] = sum_{i} h[i] h[k-i]
        # For h[i]=i+1, the max is at k=0? Or somewhere else?
        # Actually, for positive h, max is usually at k=0 (lag 0).
        # Let's verify.
        idx = np.argmax(c)
        return idx

    k_best = get_active_lag(h_init)
    
    # We will try to optimize h for this fixed k.
    # Objective: minimize f(h) = 2 * N * (h @ A_k @ h) / (sum(h))^2
    # where A_k is a matrix such that h @ A_k @ h = sum_{i} h[i] h[i+k]
    
    # Construct A_k
    # h @ A_k @ h = sum_i sum_j h[i] A_k[i,j] h[j]
    # We want this to equal sum_i h[i] h[i+k] (with boundaries handled)
    # This is equivalent to the dot product of h and a shifted version of h.
    # Let's define the quadratic form Q(h) = sum_{i=0}^{N-1-k} h[i] h[i+k]
    # Note: The convolution c[k] = sum_{i} h[i] h[i+k] is not symmetric in indices for the quadratic form unless we are careful.
    # Actually, c[k] = sum_{i=0}^{N-1} h[i] h[i+k] (with h[j]=0 if j out of bounds)
    # This is a valid quadratic form. The matrix A_k has A_k[i, i+k] = 1 for i=0..N-1-k.
    # Since h @ A @ h = sum h[i] h[j] A[i,j], we need A[i, i+k] = 1.
    # But the sum is over all i,j.
    # c[k] = sum_i h[i] h[i+k].
    # In matrix form, this is h^T M h where M is not necessarily symmetric, but h^T M h = h^T ((M+M^T)/2) h.
    # So we can use the symmetric part.
    # M[i, j] = 1 if j = i + k.
    # Symmetric part S: S[i, j] = 0.5 if j=i+k, 0.5 if i=j+k.
    
    A = np.zeros((N, N))
    # We only need to construct this if we want to be fast, but N=400 is small.
    # However, we can compute the gradient analytically without building A explicitly in the loop if we use shifts.
    
    def objective_and_grad(h):
        s = np.sum(h)
        if s < 1e-12:
            return np.inf, np.zeros(N)
        
        # Compute c[k] = sum h[i] h[i+k]
        # Using np.convolve is easy but maybe slow in gradient?
        # c = np.convolve(h, h)
        # q = c[k]
        
        # Efficient computation of q and grad q
        # q = sum_{i=0}^{N-1-k} h[i] h[i+k]
        # grad_q[i] = d/dh_i (sum_j h[j] h[j+k])
        # If i is in [0, N-1-k], term h[i] h[i+k] contributes h[i+k] to grad[i].
        # If i+k is in [0, N-1], i.e., i in [k, N-1], term h[i-k] h[i] contributes h[i-k] to grad[i].
        
        q = np.dot(h[:N-k], h[k:])
        
        grad_q = np.zeros(N)
        # Contribution from first index i (0 to N-1-k)
        grad_q[:N-k] += h[k:]
        # Contribution from second index i (k to N-1)
        grad_q[k:] += h[:N-k]
        
        # f(h) = 2 * N * q / s^2
        val = 2.0 * N * q / (s * s)
        
        # grad_f = 2 * N * ( grad_q * s^2 - q * 2 * s * 1 ) / s^4
        # grad_f = 2 * N * ( grad_q * s - 2 * q ) / s^3
        
        grad_f = 2.0 * N * (grad_q * s - 2.0 * q) / (s ** 3)
        
        return val, grad_f

    # Optimization
    # We need to handle the constraint sum(h) > 0.
    # L-BFGS-B supports bounds.
    # Since the objective is scale invariant (if we scale h by c, q scales by c^2, s by c, f is invariant),
    # the optimizer might drift. We can fix one variable or add a constraint.
    # However, L-BFGS-B with bounds is robust.
    # Let's use bounds [1e-6, None] to keep h positive?
    # The problem allows negative heights.
    # If h has negative values, s can be small or zero.
    # The "Third autocorrelation inequality" context usually implies positive functions or specific structures.
    # But the task says "real, possibly negative".
    # However, if s is close to 0, value explodes.
    # Let's start with positive h and see if it goes negative.
    # To be safe, we can normalize h to sum=1 during optimization?
    # No, L-BFGS-B doesn't do equality constraints easily.
    # We can use the fact that f is homogeneous of degree 0.
    # We can project back to sum(h)=1 after each step? No, that breaks the gradient.
    # Better: Just let it optimize. If it hits s~0, it will fail.
    # We can add a small regularization or just rely on the initial positive start.
    
    # Let's try a few different k values and initializations.
    
    best_val = np.inf
    best_h = h_init.copy()
    
    # Initial candidate: linear ramp
    val_init = compute_value(h_init)
    if val_init < best_val:
        best_val = val_init
        best_h = h_init.copy()
    
    # Try optimizing for the active lag of the initial ramp
    # And also try a few other lags
    
    lags_to_try = [k_best, 0, N//4, N//2]
    
    for k in lags_to_try:
        if time.time() > deadline - 20:
            break
            
        if k >= N:
            continue
            
        # Initial guess for this k: 
        # If k is large, a ramp might not be optimal.
        # Let's use the current best_h as a starting point, or a simple ramp.
        # Using best_h is good if the structure is similar.
        x0 = best_h.copy()
        
        # Run optimization
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            
            if res.success or res.fun < best_val:
                h_opt = res.x
                # Check validity
                s = np.sum(h_opt)
                if abs(s) < 1e-6:
                    continue
                val_opt = compute_value(h_opt)
                
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    
                    # Update active lag for next iterations if needed
                    # k_best = get_active_lag(best_h)
                    
        except Exception as e:
            pass

    # Refine: The active lag might have changed during optimization.
    # Let's do one more round with the new active lag.
    if time.time() < deadline - 10:
        k_new = get_active_lag(best_h)
        if k_new != k_best:
            try:
                res = minimize(
                    objective_and_grad, # Note: k is captured in closure? 
                    # Wait, objective_and_grad uses k from outer scope.
                    # We need to pass k or redefine.
                    # Let's redefine a factory.
                    None, # placeholder
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
                )
            except:
                pass
            
            # Redo properly with correct k
            def make_obj(k_val):
                def obj_and_grad(h):
                    s = np.sum(h)
                    if s < 1e-12:
                        return np.inf, np.zeros(N)
                    q = np.dot(h[:N-k_val], h[k_val:])
                    grad_q = np.zeros(N)
                    grad_q[:N-k_val] += h[k_val:]
                    grad_q[k_val:] += h[:N-k_val]
                    val = 2.0 * N * q / (s * s)
                    grad_f = 2.0 * N * (grad_q * s - 2.0 * q) / (s ** 3)
                    return val, grad_f
                return obj_and_grad
            
            obj_k = make_obj(k_new)
            x0 = best_h.copy()
            res = minimize(
                obj_k, 
                x0, 
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            if res.fun < best_val:
                h_opt = res.x
                s = np.sum(h_opt)
                if abs(s) > 1e-6:
                    val_opt = compute_value(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()

    # Final check
    if time.time() < deadline - 5:
        # Maybe try a random perturbation of the best solution to escape local minima?
        # Or just return.
        pass

    return best_h
