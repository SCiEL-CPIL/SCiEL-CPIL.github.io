import numpy as np
import time

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for ACO on multiple-constraint knapsack.
    Uses Lagrangian relaxation to estimate dual prices for constraints,
    then scores items by reduced cost: prize - lambda^T * weight.
    """
    t0 = time.time()
    DEADLINE = t0 + 320.0  # Safety margin before 330s kill

    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Ensure shapes: prizes (5,100), weights (5,100)
    if prizes.ndim == 1:
        # Fallback: treat as single prize vector
        P = prizes.copy()
        W = weights if weights.ndim == 2 else weights[None, :]
    else:
        P = prizes.sum(axis=0)  # Total prize per item, shape (100,)
        W = weights  # shape (5, 100)

    n_constraints, n_items = W.shape
    capacity = np.full(n_constraints, 25.0)

    # Lagrangian dual estimation via subgradient method
    # Minimize g(lambda) = sum_i max(0, P_i - lambda^T W_i) + lambda^T capacity
    # Actually, the Lagrangian for max sum P_i x_i s.t. W x <= cap, x in {0,1}
    # is L(x, lambda) = sum P_i x_i - lambda^T (W x - cap) = sum (P_i - lambda^T W_i) x_i + lambda^T cap
    # Dual function g(lambda) = max_x L(x,lambda) = sum_i max(0, P_i - lambda^T W_i) + lambda^T cap
    # We want to minimize g(lambda) over lambda >= 0 to get a tight bound.
    # The subgradient of g(lambda) is: sum_{i: P_i > lambda^T W_i} W_i - cap
    # Wait, let's re-derive.
    # g(lambda) = sum_i max(0, r_i) + lambda^T cap, where r_i = P_i - lambda^T W_i
    # If r_i > 0, item i is "selected" in the relaxed problem.
    # Subgradient wrt lambda_k: sum_{i: r_i > 0} W_{ki} - cap_k
    # We perform projected subgradient descent on lambda.

    lambda_ = np.zeros(n_constraints)
    # Step size parameter
    # Standard subgradient method: lambda_{t+1} = max(0, lambda_t - alpha_t * grad_t)
    # alpha_t = gamma / ||lambda_t|| or similar. Let's use a simple fixed step or decaying.
    # Since we only need a good heuristic, not the exact optimum, a few iterations suffice.

    # Precompute for speed
    # We will run ~50-100 iterations of subgradient descent.
    max_iter = 100
    # Initial step size
    alpha = 0.1
    
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5.0:
            break
            
        # Compute reduced costs r_i = P_i - lambda^T W_i
        # W is (C, N), lambda is (C,). lambda^T W is (N,)
        r = P - W.T @ lambda_
        
        # Determine which items are "selected" (r_i > 0)
        selected = r > 0
        
        # Subgradient: sum_{i: selected} W_i - capacity
        # W is (C, N). Sum over items axis=1 for selected items.
        # Use boolean indexing or multiplication
        # grad = W[:, selected].sum(axis=1) - capacity
        # To avoid creating a slice, use: W * selected[None, :]
        grad = (W * selected[None, :]).sum(axis=1) - capacity
        
        # Update lambda
        # lambda = max(0, lambda - alpha * grad)
        # We might need to adapt alpha. Let's try a simple diminishing step.
        # Or fixed step if it converges fast enough.
        # Given small N=100, C=5, this is very fast.
        
        lambda_ = np.maximum(0, lambda_ - alpha * grad)
        
        # Optional: simple step size adaptation
        # If we want to be more robust, we could track the best dual value.
        # But for a heuristic, the resulting lambda_ is what matters.
        
        # Decay alpha slightly to help convergence
        alpha *= 0.95

    # Final reduced costs
    # r_i = P_i - lambda^T W_i
    reduced_costs = P - W.T @ lambda_
    
    # The heuristic should be proportional to the "attractiveness".
    # Items with high reduced cost are good candidates.
    # However, the ACO heuristic must be positive.
    # If reduced cost is negative, the item is not worth taking in the relaxed problem.
    # We can map negative values to a small positive epsilon, or use a transformation.
    # The prompt says: "Heuristic entries that are not positive become 1e-9".
    # So we can just return the reduced costs, and let the evaluator handle the clipping?
    # No, the evaluator clips non-positive to 1e-9.
    # But we want to guide the ants. If reduced cost is negative, it's bad.
    # If we return negative values, they become 1e-9.
    # If we return positive values, they are used as is.
    # This seems reasonable.
    
    # However, the scale might matter. 
    # Let's normalize or scale? 
    # The prompt says "larger is more attractive".
    # Relative ordering is what matters for probability proportional to pheromone * heuristic.
    # But pheromone starts at 1. So initially, probability is proportional to heuristic.
    # If heuristics are very small (e.g., 0.001), and pheromone is 1, it's fine.
    # If heuristics are very large, it's fine.
    # But if some are negative -> 1e-9, they are effectively excluded.
    
    # Let's ensure the output is strictly positive for valid items.
    # The evaluator does: h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    # So we can just return the reduced costs.
    
    # One potential issue: if ALL reduced costs are negative, all become 1e-9.
    # Then all items have equal heuristic. The choice depends entirely on pheromone (which starts uniform).
    # This is a degenerate case.
    # To avoid this, we might shift the values so that the max is positive?
    # But the Lagrangian dual tells us that if lambda is optimal, sum max(0, r_i) + lambda^T cap is the bound.
    # If we shift, we change the meaning.
    # Let's stick to the reduced costs.
    
    h = reduced_costs.copy()
    
    # Sanitize: ensure finite and shape is correct
    h = np.asarray(h, dtype=float)
    if h.shape != (n_items,):
        # This should not happen, but just in case
        h = np.ones(n_items)
        
    # Replace non-finite with 1e-9 (evaluator does this too, but good to be safe)
    h = np.where(np.isfinite(h), h, 1e-9)
    
    # The evaluator will clip non-positive to 1e-9.
    # We return h.
    
    return h
