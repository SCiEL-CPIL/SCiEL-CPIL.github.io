import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    """
    Construct the sequence using a differentiable soft-domain-wall approach.
    """
    
    def get_h(p, eps):
        i = np.arange(N)
        p_mod = p % N
        h = np.zeros(N)
        # Sum over adjacent periods to handle wrap-around
        for m in [-1, 0, 1]:
            u1 = (i - p_mod - m*N) / eps
            u2 = (i - (p_mod + N/2.0) - m*N) / eps
            # Clip for stability
            u1 = np.clip(u1, -20, 20)
            u2 = np.clip(u2, -20, 20)
            h += (1.0 / (1.0 + np.exp(-u1))) - (1.0 / (1.0 + np.exp(-u2)))
        return h

    def loss_and_grad(params):
        p, eps = params
        eps = max(eps, 0.1)
        h = get_h(p, eps)
        
        C = np.zeros(N)
        for k in range(1, N):
            shifted = np.roll(h, -k)
            C[k] = np.sum(h * (1.0 - shifted))
            
        beta = 10.0
        C_max = np.max(C[1:])
        exp_C = np.exp(beta * (C[1:] - C_max))
        sum_exp = np.sum(exp_C)
        loss = C_max + np.log(sum_exp) / beta
        
        # Penalty for sum
        sum_h = np.sum(h)
        penalty = 100.0 * (sum_h - N/2.0)**2
        total_loss = loss + penalty
        
        return total_loss

    # Optimize
    # Start with a few random initial guesses
    best_res = None
    best_loss = float('inf')
    
    for _ in range(5):
        p0 = np.random.uniform(0, N)
        eps0 = np.random.uniform(0.5, 2.0)
        res = minimize(loss_and_grad, [p0, eps0], method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_loss:
            best_loss = res.fun
            best_res = res
            
    p_opt, eps_opt = best_res.x
    eps_opt = max(eps_opt, 0.1)
    
    h = get_h(p_opt, eps_opt)
    
    # Correct sum
    h += (N/2.0 - np.sum(h)) / N
    h = np.clip(h, 0.0, 1.0)
    
    return h
