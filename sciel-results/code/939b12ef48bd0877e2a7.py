import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    a = np.arange(1, n + 1, dtype=np.float64) / n
    w = (1 - np.arange(M, dtype=np.float64) / M) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    
    # Start from the Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        
    def objective_and_grad(x, t):
        # x shape (n,)
        # L shape (M, n)
        L = x[None, :] + a[None, :] * y[:, None]
        
        # Sort L along axis 1
        # idx shape (M, n)
        idx = np.argsort(L, axis=1)
        L_sorted = np.take_along_axis(L, idx, axis=1)
        
        # d shape (M, n-1)
        d = L_sorted[:, 1:] - L_sorted[:, :-1]
        
        # Soft-min derivative
        # v = (w - d) / t
        # w shape (M,) -> (M, 1)
        w_expanded = w[:, None]
        v = (w_expanded - d) / t
        # sigmoid(v)
        # Use stable sigmoid
        # For v > 0: 1 / (1 + exp(-v))
        # For v < 0: exp(v) / (1 + exp(v))
        # Or just use scipy.special.expit if available, but let's implement manually for safety
        # Actually, np.exp(-v) can overflow.
        # Let's use the piecewise definition or a stable log-sum-exp approach for the value too.
        
        # Value of S(d, w) = -t * log(exp(-d/t) + exp(-w/t))
        # = -t * log(exp(-max(d,w)/t) * (exp(-(d-max)/t) + exp(-(w-max)/t)))
        # = max(d,w) - t * log(exp(-(d-max)/t) + exp(-(w-max)/t))
        
        m = np.maximum(d, w_expanded)
        # term1 = exp(-(d-m)/t), term2 = exp(-(w-m)/t)
        # One of these is 1, the other is <= 1.
        # log(1 + exp(-|d-w|/t))
        diff = np.abs(d - w_expanded)
        log_term = np.log1p(np.exp(-diff / t))
        S_val = m - t * log_term
        
        # Derivative of S w.r.t d
        # sigma((w-d)/t)
        # If w >= d, v >= 0. sigma(v) = 1/(1+exp(-v))
        # If w < d, v < 0. sigma(v) = exp(v)/(1+exp(v))
        
        # We can compute this as:
        # deriv = np.where(v >= 0, 1 / (1 + np.exp(-v)), np.exp(v) / (1 + np.exp(v)))
        # But v can be large.
        # Alternatively, deriv = 1 - sigmoid((d-w)/t)
        # Let's use the stable sigmoid implementation.
        
        # Actually, we can just use the fact that:
        # dS/dd = exp(-d/t) / (exp(-d/t) + exp(-w/t))
        # = 1 / (1 + exp(-(w-d)/t))
        
        # Let's compute u = (w - d) / t
        # If u > 0, deriv = 1 / (1 + exp(-u))
        # If u < 0, deriv = exp(u) / (1 + exp(u))
        
        # To avoid overflow in exp(-u) when u is large negative (i.e. w-d large negative),
        # we can use:
        # deriv = np.exp(-np.logaddexp(0, -u)) ? No.
        
        # Let's just clip u or use a stable sigmoid.
        # scipy.special.expit is stable.
        # Since we are in a function that might be called many times, let's check if we can import it.
        # The prompt says SciPy is available.
        
        try:
            from scipy.special import expit
            deriv = expit(v)
        except ImportError:
            # Fallback
            deriv = np.zeros_like(v)
            mask = v >= 0
            deriv[mask] = 1.0 / (1.0 + np.exp(-v[mask]))
            deriv[~mask] = np.exp(v[~mask]) / (1.0 + np.exp(v[~mask]))
            
        # Union length U(y)
        # U = w + sum(S_val)
        U = w + np.sum(S_val, axis=1)
        
        # Area
        area = np.mean(U)
        
        # Gradient
        # grad_x_k = (1/M) sum_m sum_j deriv[j,m] * (delta_{sigma(j+1),k} - delta_{sigma(j),k})
        # We can accumulate this by scattering.
        # idx shape (M, n)
        # For each m, we have idx[m, j] and idx[m, j+1]
        # grad contribution from pair j at height m is deriv[j,m] * (e_{idx[m, j+1]} - e_{idx[m, j]})
        
        grad = np.zeros(n, dtype=np.float64)
        
        # Vectorized scatter
        # We want to add deriv[j,m] to grad[idx[m, j+1]] and subtract from grad[idx[m, j]]
        # idx_plus = idx[:, 1:]
        # idx_minus = idx[:, :-1]
        
        idx_plus = idx[:, 1:]
        idx_minus = idx[:, :-1]
        
        # Use np.add.at for accumulation
        np.add.at(grad, idx_plus.ravel(), deriv.ravel())
        np.add.at(grad, idx_minus.ravel(), -deriv.ravel())
        
        grad /= M
        
        return area, grad

    start_time = time.time()
    deadline = 600.0
    current_x = x0.copy()
    
    # Annealing schedule
    # Start with t=1.0, decrease to 0.01
    # Number of L-BFGS steps?
    # L-BFGS-B takes iterations. We can run multiple minimize calls with decreasing t.
    
    temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    for t in temperatures:
        if time.time() > deadline - 30:
            break
            
        # Run minimize
        # We need to wrap objective_and_grad to return (f, g)
        def fun(x):
            f, g = objective_and_grad(x, t)
            return f, g
            
        try:
            res = minimize(fun, current_x, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'maxls': 50})
            if res.success or res.fun < np.mean([np.mean((w + np.sum(np.minimum(np.sort(x[None,:] + a[None,:]*y[:,None], axis=1)[:,1:] - np.sort(x[None,:] + a[None,:]*y[:,None], axis=1)[:,:-1], w[:,None]), axis=1))) for _ in [0]]):
                current_x = res.x
        except Exception:
            pass
            
    return current_x
