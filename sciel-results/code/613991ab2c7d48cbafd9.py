"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE_SECONDS = 300.0  # 300s limit per evaluation, leave margin

def _check_deadline():
    """Return True if we should stop and return current best."""
    return (time.time() - _START_TIME) > (_DEADLINE_SECONDS - 5.0)

def _compute_cost(x):
    """Compute cost for points x of shape (n, 4)."""
    x_scaled = x / 10.0  # scale to [0, 1]
    return 1.0 + np.mean(x_scaled, axis=1)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Pareto Front Selection.
    
    This strategy treats the problem as a multi-objective optimization where
    we want to minimize the predicted objective value (mu) and the evaluation
    cost. We identify the Pareto front of these two objectives. Points on the
    front are the most efficient: you cannot lower the predicted value without
    increasing cost, or lower cost without increasing predicted value.
    
    To select a single point from this set (or rank all points), we use a 
    utility score that penalizes deviation from the ideal point (min mu, min cost)
    in a normalized space. This encourages a balanced exploration of cheap, 
    promising points and avoids the greedy exploitation of expensive points 
    or the pure exploration of cheap, unpromising points.
    
    Specifically:
    1. Normalize mu and cost to [0, 1] range based on current candidates.
    2. Calculate a composite score: - (normalized_mu + normalized_cost).
    3. However, simple summing might be dominated by one scale. 
       We use a weighted sum where weights are adaptive or fixed to balance.
       Given the cost range [1, 2] and mu range potentially larger, we should
       be careful.
       
    A more robust approach for "Pareto-like" selection in single-objective 
    context with cost is **Cost-Adjusted Expected Improvement** or 
    **Cost-Adjusted Lower Confidence Bound**.
    
    Let's stick to a structural change: **Pareto Front Ranking**.
    1. Identify candidates that are not dominated by any other candidate in 
       terms of (mu, cost). A candidate A dominates B if A.mu <= B.mu and 
       A.cost <= B.cost (at least one strict).
    2. Candidates on the Pareto front get a high base score.
    3. Candidates off the front are penalized based on their "distance" to the 
       nearest Pareto point.
       
    However, computing the full Pareto front and distances for 2000 points 
    in O(N^2) or O(N log N) is feasible but might be slow if not optimized.
    N=2000 is small enough for O(N^2) in Python if vectorized, but let's try 
    a simpler proxy that captures the spirit: **Normalized Weighted Sum**.
    
    Score = -(w1 * norm_mu + w2 * norm_cost)
    
    To make it "Pareto-like", we can use the **Non-Dominated Sorting Genetic Algorithm (NSGA-II)** 
    crowding distance concept or just rank them.
    
    Actually, a simpler and very effective structural alternative to Thompson Sampling 
    is **Expected Hypervolume Improvement (EHVI)** approximation or just 
    **Cost-Weighted EI**.
    
    Let's try **Cost-Weighted Lower Confidence Bound (LCB)**.
    LCB = mu - kappa * sigma.
    Cost-aware LCB = (mu - kappa * sigma) / cost.
    This is a deterministic rule.
    
    The prompt asks for a MATERIAL DIFFERENCE. 
    Current: Thompson Sampling (Stochastic).
    New: **Pareto Front Selection** (Deterministic, Multi-objective logic).
    
    Implementation of Pareto Front Selection:
    1. Compute mu and cost arrays.
    2. For each point, determine if it is dominated.
    3. Points on the Pareto front get a score of 1.0 (or high value).
    4. Points not on the front get a score based on how close they are to the front.
    
    Since we need to return scores for ALL 2000 points to let the evaluator pick the best,
    we can assign:
    - Pareto points: High score.
    - Non-Pareto points: Lower score, scaled by their "regret" relative to the front.
    
    To avoid O(N^2) complexity in pure Python loops which might be slow, we can use 
    a vectorized approach or a simplified heuristic that approximates Pareto selection.
    
    Heuristic: 
    Normalize mu to [0, 1] (0 is best/min, 1 is worst/max).
    Normalize cost to [0, 1] (0 is best/min, 1 is worst/max).
    Score = - (norm_mu + norm_cost) / 2.
    
    This is a weighted sum. It is NOT the same as Pareto front selection, but it is 
    a common scalarization.
    
    Let's try a slightly more sophisticated scalarization: **Multiplicative Weighted Sum**.
    Score = - (norm_mu * norm_cost).
    This favors points where BOTH are low. It creates a "corner" preference.
    
    Let's implement the Multiplicative Weighted Sum (Geometric Mean approach).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Check deadline
    if _check_deadline():
        # Return a simple deterministic score to finish quickly
        return -mu / cost
    
    # Normalize mu and cost to [0, 1] range
    # We want low values to be "good" (close to 0 in normalized space for penalty)
    mu_min, mu_max = np.min(mu), np.max(mu)
    cost_min, cost_max = np.min(cost), np.max(cost)
    
    # Handle division by zero in normalization
    mu_range = mu_max - mu_min
    cost_range = cost_max - cost_min
    
    if mu_range < 1e-10:
        norm_mu = np.zeros_like(mu)
    else:
        norm_mu = (mu - mu_min) / mu_range
        
    if cost_range < 1e-10:
        norm_cost = np.zeros_like(cost)
    else:
        norm_cost = (cost - cost_min) / cost_range
        
    # Multiplicative Weighted Sum (Geometric)
    # We want to minimize the product of normalized objectives.
    # Score should be high for low product.
    # product = norm_mu * norm_cost
    # Score = - product
    
    # To add some exploration, we can mix in a small amount of uncertainty
    # but the directive says structural change. Pure Pareto/Scalarization is the change.
    
    # However, if mu is very low but cost is high, product might be higher than 
    # a point with slightly higher mu but much lower cost. This is the desired behavior.
    
    # Let's also consider the "distance to ideal point" (0,0) in normalized space.
    # Euclidean distance: sqrt(norm_mu^2 + norm_cost^2)
    # This is also a valid scalarization.
    
    # Let's use the Euclidean distance to the ideal point (0,0) in normalized space.
    # This is equivalent to minimizing the L2 norm of the normalized regrets.
    
    distances = np.sqrt(norm_mu**2 + norm_cost**2)
    
    # We want higher scores for lower distances.
    # Score = - distances
    
    # To make it more robust against scale issues, we can also consider 
    # the "Pareto optimality" explicitly.
    # But Euclidean distance is a smooth approximation that is fast.
    
    # Let's refine: 
    # If we just use -distances, we might pick a point that is "average" in both 
    # rather than "excellent" in one. 
    # The product (norm_mu * norm_cost) favors the corner (0,0) more strongly.
    # Let's use the product.
    
    product = norm_mu * norm_cost
    scores = -product
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    if not np.any(np.isfinite(scores)):
        return np.zeros_like(mu)
        
    return scores
# EVOLVE-BLOCK-END
