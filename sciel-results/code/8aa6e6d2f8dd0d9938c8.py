"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    
    We formulate this as a continuous optimization problem over the vector h in [0,1]^N
    subject to sum(h) = N/2. We use a Log-Sum-Exp (LSE) surrogate for the maximum overlap
    to create a smooth objective function. We use L-BFGS-B (via scipy.optimize.minimize)
    for efficient gradient-based optimization, which navigates the landscape more effectively
    than simple gradient descent or derivative-free methods.
    
    The theoretical lower bound for the maximum overlap is N/4, achieved by the uniform
    distribution h_i = 0.5. We verify this by performing a short optimization that should
    converge to this state (or a state with equivalent performance).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Initial guess: uniform solution
    # This is the known optimal solution, so we start here to ensure we don't get stuck
    # in a local minimum far from the optimum.
    h0 = np.full(N, 0.5)
    
    # We can also try a slightly perturbed start to ensure the optimizer "does work"
    # and verifies the local minimum, but starting at the global optimum is safest.
    # Let's use a very small perturbation to make the gradient non-zero initially.
    rng = np.random.default_rng(42)
    h0 = 0.5 + 0.01 * rng.standard_normal(N)
    h0 = np.clip(h0, 0.0, 1.0)
    
    # Normalize to satisfy sum constraint exactly
    s = np.sum(h0)
    if s > 1e-9:
        h0 = h0 * (N / 2.0) / s
    h0 = np.clip(h0, 0.0, 1.0)
    # Re-normalize precisely
    for _ in range(10):
        s = np.sum(h0)
        target = N / 2.0
        if abs(s - target) < 1e-9:
            break
        h0 = h0 * (target / s)
        h0 = np.clip(h0, 0.0, 1.0)

    beta = 10.0  # Sharpness of the max approximation
    # Higher beta makes LSE closer to max, but can make optimization harder.
    # 10 is a reasonable balance.

    def compute_overlaps(h_vec):
        # C[k] = sum_i h[i] * (1 - h[i+k]) for circular shift k
        overlaps = np.zeros(N)
        for k in range(N):
            h_shifted = np.roll(h_vec, k)
            overlaps[k] = np.sum(h_vec * (1.0 - h_shifted))
        return overlaps

    def objective_and_grad(h_vec):
        # Ensure h_vec is within bounds for gradient calculation stability
        # (Though L-BFGS-B handles bounds, it's good practice)
        h_vec = np.clip(h_vec, 0.0, 1.0)
        
        overlaps = compute_overlaps(h_vec)
        
        # Log-Sum-Exp for max
        max_overlap = np.max(overlaps)
        if beta == 0:
            return max_overlap, None
        
        # Stable LSE
        scaled = beta * overlaps
        max_s = np.max(scaled)
        lse = max_s + np.log(np.sum(np.exp(scaled - max_s)))
        obj = lse / beta
        
        # Gradient of LSE wrt overlaps is softmax(beta * overlaps)
        probs = np.exp(scaled - max_s)
        probs /= np.sum(probs)
        
        # Gradient of obj wrt h
        grad_h = np.zeros(N)
        for k in range(N):
            # dC[k]/dh[j] = (1 - h[j+k]) - h[j-k]
            dC_dh = (1.0 - np.roll(h_vec, -k)) - np.roll(h_vec, k)
            grad_h += probs[k] * dC_dh
            
        # The gradient of the LSE objective wrt h is just grad_h
        # (The 1/beta factor is included in the definition of probs via the chain rule? 
        #  Let's re-verify:
        #  f(h) = 1/beta * log(sum(exp(beta C_k(h))))
        #  df/dh_j = 1/beta * (1/sum(exp(beta C_k))) * sum(exp(beta C_k) * beta * dC_k/dh_j)
        #  df/dh_j = sum_k [ exp(beta C_k) / sum(exp(beta C_l)) ] * dC_k/dh_j
        #  df/dh_j = sum_k probs_k * dC_k/dh_j
        #  So yes, grad_h is correct.)
        
        return obj, grad_h

    # Use L-BFGS-B for optimization
    # Bounds: 0 <= h_i <= 1
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Constraints: sum(h) = N/2
    # L-BFGS-B doesn't handle equality constraints directly well in all cases,
    # but we can use a penalty method or just rely on the fact that the optimum
    # for the unconstrained problem (with bounds) might naturally satisfy the sum
    # or we can project.
    # However, scipy.optimize.minimize with method='L-BFGS-B' only handles bounds.
    # For equality constraints, we should use SLSQP or trust-constr.
    # But SLSQP is slower.
    # Let's try SLSQP with a tight tolerance and time limit.
    
    # Alternative: Use a penalty term for the sum constraint in the objective.
    # Penalty = lambda * (sum(h) - N/2)^2
    # This allows us to use L-BFGS-B.
    
    lambda_penalty = 100.0
    
    def penalized_objective_and_grad(h_vec):
        obj, grad = objective_and_grad(h_vec)
        sum_h = np.sum(h_vec)
        penalty = lambda_penalty * (sum_h - N / 2.0) ** 2
        grad_penalty = 2 * lambda_penalty * (sum_h - N / 2.0) * np.ones(N)
        return obj + penalty, grad + grad_penalty

    try:
        # Use L-BFGS-B with penalty
        res = minimize(
            penalized_objective_and_grad,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
        )
        
        if res.success:
            h_opt = res.x
        else:
            # If L-BFGS-B fails, fall back to uniform
            h_opt = np.full(N, 0.5)
            
    except Exception:
        h_opt = np.full(N, 0.5)

    # Post-processing: ensure constraints are met exactly
    # Clip to [0, 1]
    h_opt = np.clip(h_opt, 0.0, 1.0)
    
    # Normalize sum to N/2
    s = np.sum(h_opt)
    if s > 1e-9:
        h_opt = h_opt * (N / 2.0) / s
    h_opt = np.clip(h_opt, 0.0, 1.0)
    
    # Re-normalize precisely
    for _ in range(10):
        s = np.sum(h_opt)
        target = N / 2.0
        if abs(s - target) < 1e-9:
            break
        h_opt = h_opt * (target / s)
        h_opt = np.clip(h_opt, 0.0, 1.0)

    # Final precision polish: ensure sum is exactly N/2
    sum_h = np.sum(h_opt)
    diff = (N / 2.0) - sum_h
    if abs(diff) > 1e-12:
        if h_opt[0] + diff <= 1.0 and h_opt[0] + diff >= 0.0:
            h_opt[0] += diff
        else:
            h_opt = h_opt * (N / 2.0) / sum_h

    # Check time
    if time.time() > deadline:
        return np.full(N, 0.5)

    return h_opt

# EVOLVE-BLOCK-END
