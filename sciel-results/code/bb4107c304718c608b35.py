"""Second autocorrelation inequality: maximize (L2^2)/(L1 * sup|autoconv|)."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(x):
    """Compute the objective value for nonnegative heights x."""
    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-12):
        return -np.inf
    x = np.maximum(x, 0.0)
    l2_sq = float(np.dot(x, x))
    l1 = float(np.sum(x))
    if l1 <= 1e-15:
        return -np.inf
    # Autoconvolution (linear convolution of x with itself)
    ac = np.convolve(x, x, mode='full')
    sup_ac = float(np.max(np.abs(ac)))
    if sup_ac <= 1e-15:
        return -np.inf
    return l2_sq / (l1 * sup_ac)

def _objective_and_grad(x):
    """Return (-score, gradient of score) for use in minimization."""
    x = np.maximum(x, 0.0)
    l2_sq = float(np.dot(x, x))
    l1 = float(np.sum(x))
    if l1 <= 1e-15:
        return 1e10, np.zeros_like(x)
    ac = np.convolve(x, x, mode='full')
    sup_ac = float(np.max(np.abs(ac)))
    if sup_ac <= 1e-15:
        return 1e10, np.zeros_like(x)
    
    score = l2_sq / (l1 * sup_ac)
    
    # Gradient of score w.r.t. x_i
    # score = l2_sq / (l1 * sup_ac)
    # dscore/dx_i = (2*x_i * l1 * sup_ac - l2_sq * (sum(dx_ac/dx_i) * sign * delta + sup_ac * 1)) / (l1 * sup_ac)^2
    # Simplify: let D = l1 * sup_ac
    # dscore = (2*x_i*D - l2_sq*(sup_ac * 1 + l1 * d_sup_ac/dx_i)) / D^2
    # d_sup_ac/dx_i = sum_j (x_j * delta_{i,j} + x_j * delta_{i, N+1-j}) where ac[k] = sum_j x_j x_{k-j}
    # ac[k] = sum_{j=0}^{N-1} x_j x_{k-j} for k in 0..2N-2
    # d ac[k]/d x_i = x_{k-i} + x_{k-i} if k-i is in range, more precisely:
    # d ac[k]/d x_i = x_{k-i} (if 0<=k-i<N) + x_{k-i} (if 0<=k-i<N) -- no, it's just the convolution
    # Actually ac[k] = sum_{j} x[j]*x[k-j], so d ac[k]/d x[i] = x[k-i] + x[k-i] only if both terms involve x[i]
    # Wait: ac[k] = sum_{j=0}^{N-1} x[j] x[k-j]. 
    # d/dx[i] ac[k] = x[k-i] (from j=i term: d x[i] x[k-i]/dx[i] = x[k-i]) + x[k-i] (from j=k-i term: d x[i] x[k-i]/dx[i] = x[k-i])
    # But only if i != k-i, i.e., 2i != k. If 2i == k, then it's 2*x[i].
    # More carefully: d/dx[i] ac[k] = x[k-i] * [1 if 0<=k-i<N else 0] + x[k-i] * [1 if 0<=k-i<N else 0]
    # No wait. Let me redo. ac[k] = sum_{j=0}^{N-1} x[j] x[k-j].
    # Term j contributes x[j] x[k-j]. 
    # d/dx[i] of this sum: for each j, d/dx[i] (x[j] x[k-j]).
    # If j == i: x[k-i] (assuming 0 <= k-i < N)
    # If k-j == i, i.e., j = k-i: x[j] = x[k-i] (assuming 0 <= k-i < N)
    # If both (j==i and k-j==i, i.e., k=2i): the term is x[i]^2, derivative is 2*x[i].
    # So d ac[k]/d x[i] = (1 if 0<=k-i<N else 0)*x[k-i] + (1 if 0<=i<N and 0<=k-i<N else 0)*x[k-i]
    # = 2*x[k-i] if k=2i and 0<=i<N, else x[k-i] if 0<=k-i<N, else 0.
    
    # For simplicity and robustness, compute gradient numerically if analytical is too complex.
    # Given N=50, numerical gradient is feasible but let's try analytical.
    
    # Actually, let's use a simpler approach: the sup is at some index k*. 
    # d sup_ac / d x[i] = d ac[k*]/d x[i] (assuming k* is unique and fixed)
    
    # Find where sup is achieved
    k_star = int(np.argmax(np.abs(ac)))
    sign_ac = np.sign(ac[k_star])
    
    # d ac[k_star]/d x[i]
    d_ac_dx = np.zeros(N)
    for i in range(N):
        # d ac[k_star]/d x[i]
        val = 0.0
        # j = i term
        if 0 <= k_star - i < N:
            val += x[k_star - i]
        # j = k_star - i term  
        j2 = k_star - i
        if 0 <= j2 < N and j2 != i:
            val += x[j2]
        elif j2 == i:
            val += x[j2]  # already counted? No, when j==i and k-j==i, it's the same term x[i]^2, derivative 2*x[i]
            # Actually when j=i and k-j=i means k=2i, the single term x[i]*x[i] has derivative 2*x[i]
            # My formula above: first term gives x[k-i]=x[i], second term gives x[k-i]=x[i], total 2*x[i]. Correct.
            # But I have a condition j2 != i to avoid double counting... let me redo.
        # Let me just do it cleanly:
        d_ac_dx[i] = 0.0
        # Contribution from j=i: d/dx[i] x[i]*x[k-i] = x[k-i] if 0<=k-i<N
        if 0 <= k_star - i < N:
            d_ac_dx[i] += x[k_star - i]
        # Contribution from j=k_star-i: d/dx[i] x[j]*x[i] = x[j] if 0<=j<N, j != i
        j2 = k_star - i
        if 0 <= j2 < N and j2 != i:
            d_ac_dx[i] += x[j2]
    
    D = l1 * sup_ac
    D2 = D * D
    
    # dscore/dx[i] = (2*x[i]*D - l2_sq*(sup_ac + l1*sign_ac*d_ac_dx[i])) / D2
    grad = (2.0 * x * D - l2_sq * (sup_ac + l1 * sign_ac * d_ac_dx)) / D2
    
    return -score, -grad

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, keep safety margin
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Try several initializations
    inits = [
        np.ones(N),
        np.array([1.0 if i % 2 == 0 else 0.1 for i in range(N)], dtype=float),
        np.array([1.0 if i % 3 == 0 else 0.0 for i in range(N)], dtype=float),
        np.linspace(1.0, 0.1, N),
        np.linspace(0.1, 1.0, N),
        np.array([1.0] * (N//2) + [0.0] * (N - N//2)),
        np.array([0.0] * (N//2) + [1.0] * (N - N//2)),
    ]
    
    # Add some random inits
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.rand(N))
    
    for x0 in inits:
        if time.time() > deadline - 10.0:
            break
        
        x0 = np.maximum(x0, 0.0)
        # Normalize to have sum ~ 1 for better conditioning
        s = np.sum(x0)
        if s > 1e-12:
            x0 = x0 / s
        
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500},
            )
            x_opt = np.maximum(res.x, 0.0)
            sc = _score(x_opt)
            if sc > best_score:
                best_score = sc
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Final polish: local refinement around best solution
    if time.time() < deadline - 5.0 and best_score > 0:
        try:
            # Use the best solution as starting point for a more careful optimization
            x0 = best_x.copy()
            s = np.sum(x0)
            if s > 1e-12:
                x0 = x0 / s
            
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000},
            )
            x_opt = np.maximum(res.x, 0.0)
            sc = _score(x_opt)
            if sc > best_score:
                best_score = sc
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # Ensure nonnegative
    best_x = np.maximum(best_x, 0.0)
    
    # Check deadline
    if time.time() > deadline - 2.0:
        # Return quickly
        pass
    
    return list(best_x)
