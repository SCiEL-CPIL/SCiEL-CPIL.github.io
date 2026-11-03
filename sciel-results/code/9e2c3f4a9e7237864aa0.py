"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _compute_score(h):
    """
    Compute the raw value: max |autoconv| / (integral)^2.
    h is a 1D array of length N.
    integral = sum(h)
    autoconv = np.convolve(h, h, mode='full')
    We want to minimize max(|autoconv|) / (sum(h))^2
    
    Note: The task says "signed equal-width step functions".
    The integral is the sum of heights (assuming width 1).
    """
    integral = np.sum(h)
    if abs(integral) < 1e-12:
        return 1e10
    ac = np.convolve(h, h, mode='full')
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / (integral * integral)

def _soft_max_surrogate(h, t, temp=1.0):
    """
    Smooth surrogate for max(|ac|) using log-sum-exp.
    ac = convolve(h, h, 'full')
    We want to approximate max(|ac|).
    Let z = |ac|.
    log-sum-exp(z/temp) * temp approximates max(z).
    """
    ac = np.convolve(h, h, mode='full')
    z = np.abs(ac)
    # To avoid overflow in exp, subtract max
    z_max = np.max(z)
    if z_max == 0:
        return 0.0
    # log(sum(exp((z - z_max)/temp))) * temp + z_max
    # Actually, log-sum-exp(x) = max(x) + log(sum(exp(x - max(x))))
    # So soft_max = temp * (z_max/temp + log(sum(exp((z - z_max)/temp))))
    #             = z_max + temp * log(sum(exp((z - z_max)/temp)))
    exp_term = np.exp((z - z_max) / temp)
    log_sum = np.log(np.sum(exp_term))
    soft_max_val = z_max + temp * log_sum
    
    integral = np.sum(h)
    if abs(integral) < 1e-12:
        return 1e10
    return soft_max_val / (integral * integral)

def _objective(x, temp):
    """
    Objective function for the optimizer.
    x is the full vector of length N.
    """
    return _soft_max_surrogate(x, None, temp)

def _gradient(x, temp):
    """
    Analytical gradient of the soft-max surrogate.
    Let L = log-sum-exp(z/temp) * temp / I^2 where z = |ac|, I = sum(x).
    This is complex to derive analytically for the whole chain.
    Instead, we can use finite differences or rely on L-BFGS-B's numerical gradient
    if analytical is too complex. However, the prompt mentions analytical gradients
    helped. Let's try to derive or use a simpler approach.
    
    Actually, let's just use numerical gradient first to ensure correctness, 
    but the prompt says analytical was better. 
    Let's try to implement a simple numerical gradient check or just pass None 
    to let scipy handle it, but for speed, analytical is preferred.
    
    Given time constraints, I will implement a finite difference gradient 
    if analytical is too hard, but let's see if we can use the fact that 
    conv is linear.
    
    Let's stick to numerical gradient for robustness in this iteration, 
    but optimize the function evaluation speed.
    """
    # For now, return None to let scipy compute numerical gradient
    # This might be slow. Let's try to implement a simple finite diff 
    # inside the objective? No, scipy's numerical diff is fine for N=400 
    # if the function is fast.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = 1e10
    best_h = np.ones(N)
    
    # Initial random binary-ish vectors
    # Strategy: Random initialization with signs
    np.random.seed(42)
    
    # We can try a few different initializations
    inits = []
    # 1. All ones
    inits.append(np.ones(N))
    # 2. Random +/- 1
    for _ in range(5):
        inits.append(np.random.choice([-1, 1], N))
    # 3. Random continuous in [-1, 1]
    for _ in range(5):
        inits.append(np.random.uniform(-1, 1, N))
    
    # We will optimize each initialization with L-BFGS-B
    # using a soft-max surrogate with decreasing temperature
    
    temps = [10.0, 1.0, 0.1, 0.01]
    
    for h0 in inits:
        if time.time() > deadline - 5:
            break
            
        current_h = h0.copy()
        
        # Normalize to avoid huge/small numbers? 
        # The objective is scale-invariant? 
        # max(|ac|) / (sum)^2. 
        # If we scale h by k, ac scales by k^2, sum scales by k.
        # (k^2 * max_ac) / (k^2 * sum^2) = max_ac / sum^2. 
        # So it is scale invariant. We can fix sum(h) = 1 or similar to remove scale ambiguity?
        # But the problem doesn't require a specific integral. 
        # However, optimization might drift in scale. 
        # Let's add a constraint or just let it be.
        
        for temp in temps:
            if time.time() > deadline - 5:
                break
                
            # Define objective for this temp
            def obj(x):
                return _soft_max_surrogate(x, None, temp)
            
            # Use L-BFGS-B
            # We don't have bounds, but we can add them to prevent explosion
            # h in [-10, 10] should be enough
            res = minimize(
                obj,
                current_h,
                method='L-BFGS-B',
                bounds=[(-10, 10)] * N,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < 1e10:
                current_h = res.x
            
            # Check actual score (not surrogate)
            actual_score = _compute_score(current_h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = current_h.copy()
                
    # Final polish with exact max (no soft) using a local search?
    # Or just return the best found.
    
    return best_h

# EVOLVE-BLOCK-END
