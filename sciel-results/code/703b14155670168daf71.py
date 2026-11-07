"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores using a Lagrangian Relaxation approach.
    
    Strategy:
    1. Estimate the "scarcity" or "shadow price" of each constraint.
       We assume constraints are roughly equally tight, but we can refine
       this by looking at the distribution of weights. A simple heuristic
       for shadow prices in a uniform [0,1] setting with capacity 25/100 items
       (25% fill rate) is that all constraints are similarly tight.
       
    2. However, a more robust structural change is to compute the 'net value'
       of an item. In multi-constraint knapsack, an item is valuable if its
       prize is high relative to the 'cost' it imposes on the tightest constraints.
       
    3. We approximate the tightness by the inverse of the average weight.
       If a constraint has high average weight, it is likely to be a bottleneck
       sooner, so its shadow price should be higher.
       
    4. Score = Prize - lambda * (Weight Vector).
       To ensure positivity and scale invariance, we normalize.
       
    5. Specifically, we compute a 'penalty' for each item based on how much
       it consumes of the most 'expensive' (scarce) resources.
    """
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # 1. Estimate shadow prices (dual variables) for each constraint.
    # Heuristic: Constraints with higher average weight are more restrictive.
    # Since capacity is fixed (25) and weights are uniform [0,1],
    # the 'tightness' is proportional to the mean weight.
    # Let's normalize mean weights to get relative scarcity.
    mean_weights = weights.mean(axis=1)  # shape (5,)
    
    # If mean weight is high, the constraint is tighter (fewer items fit).
    # We want the shadow price to be higher for tighter constraints.
    # Use a simple scaling: lambda_i = 1 / (mean_weight_i + epsilon)
    # Or simply lambda_i = mean_weight_i if we assume linear cost?
    # Actually, in Lagrangian relaxation for knapsack, lambda is the price per unit weight.
    # If the total capacity is C and total weight is W, the "average price" is roughly 
    # (Total Prize) / (Total Weight). 
    # Let's compute a global price-to-weight ratio to set a baseline.
    
    total_prize = prizes.sum()
    total_weight = weights.sum()
    
    # Baseline "market price" of weight
    # If we assume we fill ~25% of capacity, the effective weight limit is 25.
    # Let's estimate the shadow price per constraint as:
    # lambda_i = (Total Prize / 5) / (Capacity * Average Weight Fraction?)
    
    # Simpler, more robust structural approach:
    # Compute the 'opportunity cost' of each item.
    # An item consumes w_ij of constraint j.
    # The value of that consumption is proportional to the scarcity of j.
    
    # Scarcity estimate:
    # If mean weight is ~0.5, we can fit ~50 items before hitting capacity 25? No, 25/0.5 = 50 items.
    # But we only have 100 items. So we are not constrained by count, but by sum.
    # The binding constraint is the sum.
    
    # Let's use the inverse of the mean weight as a proxy for shadow price.
    # High mean weight -> lower capacity in terms of item count -> higher price per unit weight?
    # Actually, if weight is high, each unit of weight is "cheaper" in terms of items?
    # No, if weight is high, you can fit FEWER items. So the constraint is TIGHTER.
    # Therefore, the shadow price (value of relaxing the constraint by 1 unit) should be HIGHER.
    
    # Let's try: lambda_i = 1 / mean_weight_i.
    # If mean weight is 0.1, we can fit 250 items. Loose. Low price.
    # If mean weight is 1.0, we can fit 25 items. Tight. High price.
    
    eps = 1e-9
    lambda_vec = 1.0 / (mean_weights + eps)  # shape (5,)
    
    # Normalize lambdas to have mean 1 to avoid scale issues?
    # Or keep raw. Let's normalize to mean 1 for stability.
    lambda_vec = lambda_vec / (np.mean(lambda_vec) + eps)
    
    # 2. Calculate the "cost" of each item in "prize units".
    # Cost_ij = lambda_j * w_ij
    # Total Cost_i = sum_j (lambda_j * w_ij)
    # This represents how much "prize" this item effectively consumes from the budget.
    
    # weights is (5, 100), lambda_vec is (5,)
    # We need to broadcast lambda_vec across items.
    # cost_matrix = lambda_vec[:, np.newaxis] * weights  # shape (5, 100)
    # item_cost = cost_matrix.sum(axis=0)  # shape (100,)
    
    item_cost = np.dot(lambda_vec, weights)  # shape (100,)
    
    # 3. Calculate "Net Prize" or "Profit".
    # Profit_i = Prize_i - Item_Cost_i
    # But Prize_i is the sum of prizes? Or is it a single prize?
    # The task says "prizes ... shape (5, 100)".
    # "mean ... of the highest total prize found".
    # Usually in MCKP, there is ONE prize per item, and MULTIPLE weights.
    # Here, prizes is (5, 100). This implies there might be a prize associated with each constraint?
    # OR, more likely, the "prize" is the sum of the 5 values?
    # "Item scores for ant colony optimization of the multiple-constraint knapsack"
    # "prizes and weights, of shape (5, 100)"
    # "best prize of that iteration"
    # If prizes were a single vector, it would be (100,).
    # Given the shape (5, 100), it's highly probable that the "total prize" of an item is the sum of its 5 prize components, OR the evaluator sums them.
    # Let's assume Total Prize_i = sum(prizes[:, i]).
    
    total_prize_per_item = prizes.sum(axis=0)  # shape (100,)
    
    # 4. Heuristic Score.
    # We want to prioritize items with high (Prize - Cost).
    # However, if Cost > Prize, the score becomes negative.
    # The ant algorithm uses: prob prop to pheromone * heuristic.
    # Heuristic must be positive.
    
    # A common heuristic in ACO for knapsack is Prize / Weight.
    # Here, we use (Prize / Cost) as a "return on investment" metric.
    # This is the reciprocal of the "cost per prize".
    
    # If Cost is 0 (item has no weight), score is inf.
    # If Prize is 0, score is 0.
    
    # Let's use: h = total_prize_per_item / (item_cost + eps)
    
    # This measures "how much prize we get per unit of 'shadow cost'".
    # Items that use scarce resources efficiently will have high scores.
    
    h = total_prize_per_item / (item_cost + eps)
    
    # Normalize or scale?
    # The magnitude matters for the initial pheromone distribution if pheromone starts at 1.
    # If h is very large, it dominates. If small, it's ignored.
    # Let's normalize h to be in a reasonable range, e.g., mean 1.
    # Or simply ensure it's positive and finite.
    
    # Let's check the scale.
    # Prize ~ U[0,1] sum of 5 -> ~2.5
    # Cost ~ sum(lambda * w). lambda ~ 1. w ~ 0.5. Cost ~ 2.5.
    # Ratio ~ 1.
    
    # To be safe, let's normalize by the mean.
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
    else:
        h = np.ones_like(h)
        
    # Ensure positivity
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h

# EVOLVE-BLOCK-END
