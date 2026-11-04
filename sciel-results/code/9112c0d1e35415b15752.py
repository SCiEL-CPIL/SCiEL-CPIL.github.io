"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy:
    1. Generate diverse initial points.
    2. Use smooth surrogate (log-sum-exp) + L-BFGS-B gradient optimization.
    3. Final exact evaluation.
    """
    start_time = time.time()
    DEADLINE = 145  # 5s safety margin before 150s kill
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    # First evaluate the seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    rng = np.random.RandomState(42)
    
    def smooth_objective_and_grad(h, beta):
        """
        Smooth surrogate for |2N * max(conv) / sum(h)^2|.
        """
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        
        conv = np.convolve(h, h)
        
        # Log-sum-exp for numerical stability
        max_conv = np.max(conv)
        shifted = beta * (conv - max_conv)
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        if sum_exp == 0:
            soft_max = max_conv
            weights = np.zeros_like(conv)
            weights[np.argmax(conv)] = 1.0
        else:
            logsumexp = max_conv + np.log(sum_exp) / beta
            soft_max = logsumexp
            weights = exp_shifted / sum_exp
        
        # f = 2N * soft_max / s^2
        f = 2.0 * N * soft_max / (s * s)
        
        # Gradient of soft_max w.r.t. h
        # d(soft_max)/d(h_k) = sum_i weights_i * d(conv_i)/d(h_k)
        # d(conv_i)/d(h_k) = 2 * h[i-k] if 0 <= i-k < N else 0
        # So grad_soft_max[k] = 2 * sum_i weights[i] * h[i-k]
        
        grad_soft_max = np.zeros(N)
        # Efficient computation:
        # grad_soft_max[k] = 2 * dot(weights[k:k+N], h)
        # Note: weights has length 2N-1.
        for k in range(N):
            # i ranges from k to k+N-1
            # h index is i-k, so 0 to N-1
            grad_soft_max[k] = 2.0 * np.dot(weights[k:k+N], h)
            
        # Gradient of f
        # f = 2N * soft_max * s^(-2)
        # df/dh_k = 2N * (d(soft_max)/dh_k * s^(-2) + soft_max * d(s^(-2))/dh_k)
        # d(s^(-2))/dh_k = -2 * s^(-3)
        # df/dh_k = 2N * (grad_soft_max[k] / s^2 - 2 * soft_max / s^3)
        
        grad_f = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * soft_max / (s ** 3))
        
        return f, grad_f
    
    # Generate initial points
    init_points = []
    
    # Seed
    init_points.append(np.ones(N).astype(float))
    
    # Random uniform
    for trial in range(100):
        if time.time() - start_time > DEADLINE:
            break
        h = rng.uniform(-1, 1, N)
        init_points.append(h)
    
    # Random normal
    for scale in [0.5, 1.0, 2.0]:
        for trial in range(50):
            if time.time() - start_time > DEADLINE:
                break
            h = rng.normal(0, scale, N)
            init_points.append(h)
    
    # Alternating
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    init_points.append(h_alt)
    
    # Cosine
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10]:
        init_points.append(np.cos(2 * np.pi * freq * t))
    
    # Evaluate all init points and keep top candidates
    evals = []
    for h in init_points:
        if time.time() - start_time > DEADLINE:
            break
        val, valid = evaluate(h)
        if valid:
            evals.append((val, h))
    
    evals.sort(key=lambda x: x[0])
    
    # Take top candidates for optimization
    n_optimize = min(10, len(evals))
    
    # For each top candidate, run L-BFGS with increasing beta
    for beta in [10, 50, 100, 200, 500]:
        if time.time() - start_time > DEADLINE:
            break
        
        for idx in range(n_optimize):
            if time.time() - start_time > DEADLINE:
                break
            
            h0 = evals[idx][1].copy()
            
            try:
                result = minimize(
                    lambda h, b=beta: smooth_objective_and_grad(h, b)[0],
                    h0,
                    jac=lambda h, b=beta: smooth_objective_and_grad(h, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 1e10:
                    h_opt = result.x
                    val, valid = evaluate(h_opt)
                    if valid and val < best_value:
                        best_value = val
                        best_h = h_opt.copy()
                        
            except Exception:
                pass
    
    # Final polish: small perturbations around best_h
    if time.time() - start_time < DEADLINE:
        h_curr = best_h.copy()
        lr = 0.01
        for iteration in range(100):
            if time.time() - start_time > DEADLINE:
                break
            improved = False
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE:
                    break
                h_try = h_curr.copy()
                h_try[coord] += lr * rng.uniform(-1, 1)
                val_try, valid = evaluate(h_try)
                if valid and val_try < best_value:
                    best_value = val_try
                    best_h = h_try.copy()
                    h_curr = h_try
                    improved = True
                    if best_value < 0.5:
                        break
            if best_value < 0.5:
                break
            if not improved:
                lr *= 0.7
    
    return best_h
# EVOLVE-BLOCK-END
