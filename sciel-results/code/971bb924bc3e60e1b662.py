"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We want to minimize max | (f*f)[k] | / (sum(f)^2 )
    # Let s = sum(f). We can fix s = 1 by scaling, since the ratio is scale-invariant.
    # Actually, if we scale f by c, numerator scales by c^2, denominator by c^2.
    # So we can fix sum(f) = 1.
    
    # Let x be the vector of length N with sum(x) = 1.
    # Objective: minimize max_k | (x*x)[k] |
    
    # Autoconvolution (x*x)[k] = sum_{i+j=k} x_i x_j.
    # This is a quadratic form.
    
    # We can use a smooth approximation: minimize (sum_k | (x*x)[k] |^p)^(1/p) for large p,
    # or use log-sum-exp for max.
    
    # Alternatively, use an epigraph formulation:
    # Minimize t
    # Subject to: -t <= (x*x)[k] <= t for all k
    #              sum(x) = 1
    
    # The constraints are quadratic. We can use SLSQP or trust-constr.
    
    # Let's try a simpler approach first: random initialization + gradient descent with soft-max.
    
    # Soft-max approximation: f(t) = (1/beta) * log(sum(exp(beta * |z_k|)))
    # This is differentiable if we use soft-absolute: sqrt(z^2 + eps)
    
    # Let's define the objective as:
    # J(x) = (1/beta) * log( sum_k exp(beta * sqrt((x*x)_k^2 + eps)) )
    
    # We need to project onto sum(x)=1 after each step, or use Lagrange multipliers.
    
    # Let's use scipy.optimize.minimize with SLSQP, fixing sum(x)=1.
    
    # Initial guess: alternating signs? Or random?
    # Let's try a few initializations.
    
    best_val = np.inf
    best_x = None
    
    # Define the autoconvolution function
    def autoconv(x):
        # Full convolution
        return np.convolve(x, x, mode='full')
    
    # Objective function for optimization
    beta = 10.0
    eps = 1e-10
    
    def objective(x):
        ac = autoconv(x)
        # Soft max of absolute values
        return (1.0/beta) * np.log(np.sum(np.exp(beta * np.sqrt(ac**2 + eps))))
    
    def objective_grad(x):
        ac = autoconv(x)
        mag = np.sqrt(ac**2 + eps)
        exp_term = np.exp(beta * mag)
        sum_exp = np.sum(exp_term)
        # d/dac_k [ (1/beta) log(sum exp(beta*mag)) ] = exp(beta*mag_k) / sum_exp
        weights = exp_term / sum_exp
        # d mag_k / d ac_k = ac_k / mag_k
        d_mag_d_ac = ac / mag
        # d ac_k / d x_i : ac_k = sum_{i+j=k} x_i x_j
        # d ac_k / d x_i = x_{k-i} + x_{i-k} (if within bounds)
        # Actually, ac = convolve(x, x, 'full')
        # ac_k = sum_{i} x_i x_{k-i} where x is zero-padded.
        # d ac_k / d x_i = x_{k-i} + x_{k-i} if i != k-i? No.
        # ac_k = sum_{i=0}^{N-1} x_i x_{k-i} (with x_j=0 for j<0 or j>=N)
        # d ac_k / d x_i = x_{k-i} + x_{k-i} if k-i != i, else 2*x_i? 
        # Wait, if i = k-i, i.e., k=2i, then the term is x_i^2, derivative is 2*x_i.
        # If i != k-i, the term x_i x_{k-i} appears once in the sum for ac_k? 
        # No, the sum is over i. For a fixed k, the pairs (i, k-i) and (k-i, i) are the same term in the sum over i?
        # Actually, convolve(x, x) computes sum_i x_i y_{k-i}. If x=y, it's sum_i x_i x_{k-i}.
        # So for each i, we have one term. The derivative w.r.t x_i is x_{k-i} (from the first factor) + x_{k-i} (from the second factor if k-i = i? No).
        # Let's be precise. ac_k = sum_{i=0}^{N-1} x_i * x_{k-i} (where x_j=0 if j<0 or j>=N).
        # d ac_k / d x_i = x_{k-i} (derivative of x_i) + sum_j x_j * d(x_{k-j})/d(x_i)
        # d(x_{k-j})/d(x_i) is 1 if k-j=i, i.e., j=k-i.
        # So d ac_k / d x_i = x_{k-i} + x_{k-i} = 2*x_{k-i} if k-i is in [0, N-1], else x_{k-i} if i=k-i?
        # Wait, if k-i = i, i.e., k=2i, then the term is x_i * x_i = x_i^2. Derivative is 2*x_i.
        # If k-i != i, the term is x_i * x_{k-i}. Derivative w.r.t x_i is x_{k-i}.
        # But also, when j = k-i, the term x_j * x_{k-j} = x_{k-i} * x_i. Derivative w.r.t x_i is x_{k-i}.
        # So total derivative is 2*x_{k-i} if k-i is valid and k-i != i?
        # Actually, no. The sum is over i. Let's change variable to j.
        # ac_k = sum_j x_j x_{k-j}.
        # d ac_k / d x_i = x_{k-i} (from j=i) + sum_j x_j * d(x_{k-j})/d(x_i)
        # d(x_{k-j})/d(x_i) = 1 if k-j=i => j=k-i.
        # So the second term is x_{k-i} if k-i is in [0, N-1].
        # So d ac_k / d x_i = x_{k-i} + x_{k-i} = 2*x_{k-i} if k-i is in [0, N-1].
        # But if k-i = i, i.e., k=2i, then both terms are x_i, so 2*x_i. Correct.
        # So d ac_k / d x_i = 2 * x_{k-i} if 0 <= k-i < N, else 0.
        
        # So grad_x_i = sum_k weights[k] * d_mag_d_ac[k] * d ac_k / d x_i
        #             = sum_k weights[k] * d_mag_d_ac[k] * 2 * x_{k-i} (if 0 <= k-i < N)
        # Let l = k-i, so k = l+i.
        # grad_x_i = 2 * sum_l weights[l+i] * d_mag_d_ac[l+i] * x_l (if 0 <= l < N and 0 <= l+i < 2N-1)
        
        # This is a convolution!
        # grad = 2 * convolve(x, weights * d_mag_d_ac, mode='same')? 
        # Let's check. 
        # grad_i = 2 * sum_{l} x_l * w_{l+i} where w_k = weights[k] * d_mag_d_ac[k].
        # This is the cross-correlation of x and w, or convolution with reversed w.
        # convolve(x, w, 'full')[i+N-1] = sum_l x_l w_{i+N-1-l}. Not quite.
        # Let's just compute it directly for now, N=400 is small.
        
        grad = np.zeros(N)
        for i in range(N):
            # sum over k such that 0 <= k-i < N
            k_min = max(0, i)
            k_max = min(2*N-1, i+N-1)
            ks = np.arange(k_min, k_max+1)
            valid = (ks - i >= 0) & (ks - i < N)
            ks = ks[valid]
            if len(ks) > 0:
                grad[i] = 2.0 * np.sum(weights[ks] * d_mag_d_ac[ks] * x[ks - i])
        return grad
    
    # Constraints: sum(x) = 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(N)}]
    
    # Try a few initializations
    inits = [
        np.ones(N) / N,  # uniform
        np.random.randn(N),  # random
        np.sign(np.random.randn(N)) * (1.0/N),  # random signs
    ]
    
    for init in inits:
        init = init.copy()
        # Normalize to sum=1
        init = init / np.sum(init)
        
        try:
            res = minimize(objective, init, method='SLSQP', jac=objective_grad,
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
            
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    if best_x is None:
        best_x = np.ones(N) / N
    
    # Final polish: compute the actual max absolute autoconvolution
    ac = autoconv(best_x)
    max_abs = np.max(np.abs(ac))
    integral = np.sum(best_x)
    score = max_abs / (integral ** 2)
    
    # The task says "score is the upper bound on C3'". The evaluator measures the raw value.
    # We should return the sequence of heights. The score is computed by the evaluator.
    # But we want to minimize the score.
    
    # Maybe we can do a local search to reduce the max absolute value further.
    # Simple gradient descent on the soft-max objective with projection.
    
    x = best_x.copy()
    lr = 0.01
    for step in range(1000):
        if time.time() > deadline - 5.0:
            break
        grad = objective_grad(x)
        # Project gradient onto the tangent space of sum(x)=1
        grad = grad - np.mean(grad)
        x = x - lr * grad
        # Normalize to sum=1
        x = x / np.sum(x)
    
    ac = autoconv(x)
    max_abs = np.max(np.abs(ac))
    integral = np.sum(x)
    final_score = max_abs / (integral ** 2)
    
    # Return the sequence of signed step heights
    return x

# EVOLVE-BLOCK-END
