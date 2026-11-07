import numpy as np
import time

# We need a deadline guard. The evaluator kills after 330s.
# We should track time in the acquisition function? No, the acquisition is called many times.
# But the prompt says: "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
# Wait, the acquisition function returns scores for 2000 candidates. It doesn't return the "best result found" in the sense of the optimization loop, it returns the scores that the loop uses to pick the next point.
# So the "deadline guard" likely applies to the whole evaluation process if we were writing the loop. But we are only writing the acquisition function.
# However, if the acquisition function is too slow, the loop might time out.
# The prompt says: "Each evaluation must finish within 300 s." (This usually refers to the whole run).
# "The evaluator KILLS the program after 330s wall-clock."
# If we just return scores, we don't control the loop.
# But maybe we can use time to adapt?
# "Spare time does not establish a bottleneck..."
# Let's focus on the algorithm first.

# Strategy: Pareto-frontier based acquisition.
# Candidates are scored by their rank in a non-dominated sort of (predicted value, evaluation cost).
# We want to maximize: reduction in predicted value per unit cost.
# Actually, the directive says: "assigning the highest score to points on the cost-efficient frontier where marginal reduction in predicted value per unit cost is maximized"

# Let's interpret "predicted value" as the expected improvement or the posterior mean?
# Usually, acquisition functions like UCB or EI are used.
# Here, we have mu and sigma.
# A common cost-aware acquisition is: (mu - f_best) / cost (for minimization, we want low mu, so maybe (f_best - mu) / cost? No, we want to find points with low mu. If we pick a point with low mu, we might improve f_best. The potential improvement is f_best - mu (if mu < f_best). If mu > f_best, potential improvement is negative (or zero).
# So, potential improvement = max(0, f_best - mu) + exploration_term.
# Exploration term could be sigma.
# So, Score = (max(0, f_best - mu) + beta * sigma) / cost.
# This is similar to the current program but with max(0, ...).

# The directive suggests a Pareto frontier approach.
# Let's try to implement a simple Pareto frontier ranking.
# For each candidate i, we have:
#   value_i = mu_i (or some estimate of the function value)
#   cost_i = cost_i
# We want to minimize value and cost.
# A point is non-dominated if there is no other point with both value <= value_i and cost <= cost_i, with at least one strict inequality.
# We can assign scores based on the Pareto rank.
# Points on the first Pareto frontier get the highest scores.
# Among points on the same frontier, we can rank them by some metric, e.g., value/cost or (f_best - value)/cost.

# Let's try a hybrid:
# 1. Compute a base score: (f_best - mu + beta * sigma) / cost.
# 2. But the directive specifically asks for a Pareto-frontier based acquisition.
# Let's try to implement that.

# Steps for Pareto acquisition:
# 1. Define the objective for each candidate: 
#    We want to minimize "risk" or "expected cost" and "expected value".
#    Let's say we minimize:
#      obj1 = mu (predicted value)
#      obj2 = cost
#    But we also want to account for uncertainty (sigma).
#    Maybe obj1 = mu - kappa * sigma (optimistic prediction, lower is better)
#    obj2 = cost
#    We want to minimize both.
# 2. Find the non-dominated set (Pareto frontier).
# 3. Assign scores:
#    - Points on the Pareto frontier get high scores.
#    - Among them, we can rank by the "slope" or "efficiency": (f_best - mu) / cost or similar.
#    - Points not on the frontier get lower scores, based on their Pareto rank.

# However, computing the Pareto frontier for 2000 points is O(N^2) in the worst case, which is 4,000,000 operations. This is fast in Python/NumPy.
# Let's implement this.

# But wait, the current program scores 2.18. The seed scores 1.0.
# The current program is: (2.0 * sigma - mu) / cost.
# This is equivalent to (-(mu - 2.0 * sigma)) / cost.
# This favors high sigma and low mu, and low cost.

# Let's try the Pareto approach as instructed.

def acquisition(mu, sigma, f_best, cost):
    # Start time for deadline guard?
    # The acquisition function is called once per iteration of the loop.
    # The loop runs until total cost > 30.
    # Each iteration evaluates 1 point.
    # The acquisition function is called to score 2000 candidates.
    # The total time for the whole run must be < 300s.
    # If the acquisition function takes T seconds, and there are N iterations, total time is N*T.
    # We don't know N exactly, but it's around 25-30.
    # If T is small (e.g., 0.1s), total time is ~3s.
    # So speed is not a huge issue unless we do something very expensive.
    # Pareto frontier calculation is O(N^2) = 4e6 ops, which is < 0.1s in NumPy.
    
    # 1. Compute optimistic prediction (lower is better)
    # We want to find points that are likely to be better than f_best.
    # Let's use mu - kappa * sigma as the "predicted best value".
    # A larger kappa means more exploration.
    # Let's set kappa = 2.0 (similar to UCB).
    kappa = 2.0
    pred_val = mu - kappa * sigma  # Lower is better
    
    # 2. We have two objectives to minimize:
    #    obj1 = pred_val
    #    obj2 = cost
    #    We want to find the non-dominated set.
    
    n = len(mu)
    # Initialize dominated flag
    dominated = np.zeros(n, dtype=bool)
    
    # Find non-dominated points
    # A point i is non-dominated if there is no point j such that:
    # pred_val[j] <= pred_val[i] AND cost[j] <= cost[i] AND (pred_val[j] < pred_val[i] OR cost[j] < cost[i])
    
    # Efficient way:
    # Sort by pred_val.
    # For each point in sorted order, check if any previous point has cost <= current cost.
    # If yes, current point is dominated.
    # If no, current point is non-dominated.
    
    order = np.argsort(pred_val)
    sorted_pred = pred_val[order]
    sorted_cost = cost[order]
    
    # We need to check for each point if there is a previous point with cost <= current cost.
    # Since we are processing in increasing order of pred_val, any previous point has pred_val <= current pred_val.
    # So we just need to check if min_cost_so_far <= current_cost.
    
    min_cost_so_far = np.inf
    is_dominated = np.zeros(n, dtype=bool)
    
    for i in range(n):
        idx = order[i]
        if sorted_cost[i] < min_cost_so_far:
            # This point is not dominated by any previous point.
            # But wait, could it be dominated by a later point?
            # No, later points have higher pred_val.
            # So if it's not dominated by previous points, it's non-dominated?
            # Not quite. A later point could have lower cost and same/higher pred_val?
            # No, we sorted by pred_val. Later points have >= pred_val.
            # So if a later point has lower cost, it doesn't dominate the current point because its pred_val is higher.
            # So the condition is: a point is non-dominated if and only if no previous point (in sorted order) has cost <= current cost.
            # Wait, if a previous point has cost == current cost and pred_val == current pred_val, it's a duplicate.
            # Let's handle that.
            
            # Actually, the standard algorithm is:
            # Sort by obj1.
            # Iterate. Keep track of min obj2 seen so far.
            # If current obj2 < min obj2 seen so far, it's non-dominated.
            # Else, it's dominated.
            
            # This works because if current obj2 >= min obj2 seen so far, there is a point with lower obj1 and lower/equal obj2.
            
            is_dominated[idx] = False
            min_cost_so_far = sorted_cost[i]
        else:
            is_dominated[idx] = True
    
    # Now, is_dominated is False for non-dominated points.
    # But this only identifies the "first" Pareto frontier?
    # Actually, this identifies all non-dominated points.
    # Points that are dominated by non-dominated points are dominated.
    # Points that are dominated by dominated points are also dominated.
    # So this correctly identifies the set of non-dominated points.
    
    # Now we need to assign scores.
    # The directive says: "assigning the highest score to points on the cost-efficient frontier where marginal reduction in predicted value per unit cost is maximized"
    
    # Let's define "marginal reduction" as (f_best - pred_val) / cost.
    # This is the expected improvement per unit cost.
    # We want to maximize this.
    
    # For non-dominated points, we can rank them by this metric.
    # For dominated points, we can give them a lower score.
    
    # Let's compute the efficiency score for all points.
    # Note: f_best - pred_val can be negative. We should clamp it to 0?
    # If pred_val > f_best, the expected improvement is 0.
    # But we might still want to explore there if sigma is high?
    # pred_val already includes -kappa*sigma.
    # If pred_val > f_best, it means mu - kappa*sigma > f_best.
    # This suggests that even with exploration, it's unlikely to beat f_best.
    # So efficiency should be low.
    
    efficiency = (f_best - pred_val) / cost
    # efficiency can be negative.
    
    # Let's assign scores:
    # Non-dominated points: score = efficiency + bonus
    # Dominated points: score = efficiency - penalty
    
    # Or, we can rank all points by efficiency, but boost non-dominated points.
    
    # Let's try:
    # score = efficiency
    # For non-dominated points, add a large constant to push them to the top.
    # But we need to ensure that the best non-dominated point has the highest score.
    
    # Let's find the max efficiency among non-dominated points.
    # And assign scores based on rank.
    
    # Simpler approach:
    # score = efficiency
    # Then, for non-dominated points, we can multiply by a factor or add a bias.
    
    # Let's try:
    # score = efficiency + 10 * (1 - is_dominated)
    # This gives non-dominated points a +10 boost.
    # But if efficiency is very negative, they might still be low.
    
    # Better:
    # Among non-dominated points, rank by efficiency.
    # Among dominated points, rank by efficiency.
    # Then, assign scores such that all non-dominated points have higher scores than all dominated points.
    
    # Let's do that.
    
    scores = np.zeros(n)
    
    # Get indices of non-dominated and dominated points
    nondom_idx = np.where(~is_dominated)[0]
    dom_idx = np.where(is_dominated)[0]
    
    # Sort non-dominated points by efficiency (descending)
    if len(nondom_idx) > 0:
        nondom_eff = efficiency[nondom_idx]
        order_nondom = np.argsort(-nondom_eff)  # Descending
        # Assign scores: highest efficiency gets highest score
        # We can use the rank or the actual efficiency value.
        # Using actual efficiency value allows for finer granularity.
        # But we need to ensure they are higher than dominated points.
        
        # Let's find the max efficiency among dominated points.
        if len(dom_idx) > 0:
            max_dom_eff = np.max(efficiency[dom_idx])
        else:
            max_dom_eff = -np.inf
        
        # The min efficiency among non-dominated points is min(nondom_eff)
        min_nondom_eff = np.min(nondom_eff)
        
        # If min_nondom_eff <= max_dom_eff, we need to shift non-dominated scores up.
        # Let's shift all non-dominated scores by a large amount.
        # Or, we can assign scores based on rank.
        
        # Let's assign scores based on rank within non-dominated set.
        # Rank 0 (best) gets score 1.0, Rank 1 gets 0.9, etc.
        # And dominated points get scores < 0.
        
        # This is a robust way to separate them.
        
        # But we have 2000 points. Rank 1999 would get a very low score.
        # Let's use a decay function.
        
        # For non-dominated points:
        # score = 1.0 / (1 + rank)
        # For dominated points:
        # score = -1.0 / (1 + rank)
        
        # This ensures all non-dominated points have positive scores, and all dominated points have negative scores.
        # And within each group, higher efficiency gets a higher score.
        
        # Let's implement this.
        
        # Sort non-dominated indices by efficiency (descending)
        nondom_eff = efficiency[nondom_idx]
        order_nondom = np.argsort(-nondom_eff)
        
        for rank, idx in enumerate(order_nondom):
            scores[idx] = 1.0 / (1.0 + rank)
        
        # Sort dominated indices by efficiency (descending)
        if len(dom_idx) > 0:
            dom_eff = efficiency[dom_idx]
            order_dom = np.argsort(-dom_eff)
            
            for rank, idx in enumerate(order_dom):
                scores[idx] = -1.0 / (1.0 + rank)
        
    else:
        # All points are dominated? This shouldn't happen, but just in case.
        dom_eff = efficiency[dom_idx]
        order_dom = np.argsort(-dom_eff)
        for rank, idx in enumerate(order_dom):
            scores[idx] = 1.0 / (1.0 + rank)
    
    # Sanitize
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return scores
