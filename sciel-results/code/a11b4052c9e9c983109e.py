"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the ant colony heuristic.
    
    Strategy: Shadow-price marginal utility blended with soft-min efficiency.
    
    We estimate the "shadow price" of each constraint as inversely proportional
    to its total capacity slack (i.e., 1 / sum of weights for that constraint,
    normalized). Items that consume scarce resources are penalized.
    
    Score per item = prize_total - sum_c (shadow_price[c] * weight[c, item])
    
    This is blended with a soft-min of per-constraint prize/weight ratios
    (using log-sum-exp for numerical stability) to capture items that are
    efficient across ALL constraints, not just on average.
    
    The combined score is sharpened via a power transform.
    """
    start_time = time.time()
    deadline = start_time + 280.0
    
    try:
        prizes = np.asarray(prizes, dtype=float)
        weights = np.asarray(weights, dtype=float)
        
        n_constraints, n_items = 5, 100
        capacity = 25.0
        eps = 1e-9
        
        # --- Component 1: Shadow-price marginal utility ---
        # For each constraint c, estimate scarcity as 1 / total_weight[c]
        # Items with high weight in scarce constraints get penalized.
        total_weights = np.sum(weights, axis=1)  # (5,)
        total_weights = np.maximum(total_weights, eps)
        
        # Shadow price: higher when total weight is lower (scarcer resource)
        # Normalize so that sum of shadow prices is reasonable
        shadow_prices = 1.0 / total_weights  # (5,)
        # Normalize shadow prices to have mean 1
        shadow_prices = shadow_prices / np.mean(shadow_prices)
        
        # Total prize per item (sum across constraints? No, prizes shape is (5,100))
        # Actually prizes is (5, 100) - each row is a different prize dimension?
        # Wait, re-reading: "prizes and weights, of shape (5, 100)"
        # And "Every constraint has capacity 25"
        # So prizes[c, i] is the prize contribution of item i to constraint c?
        # Or is it that there are 5 prize dimensions and 5 weight dimensions?
        # 
        # Looking at the task: "prizes and weights, of shape (5, 100), are uniform in [0, 1]"
        # "Every constraint has capacity 25"
        # "Each ant adds items that still fit in all constraints"
        # 
        # So for each item i and constraint c:
        #   weight[c, i] is how much of constraint c's capacity item i consumes
        #   prize[c, i] is the prize value... but wait, prize should be scalar per item
        # 
        # Actually, I think prizes might be (1, 100) broadcast to (5, 100)? 
        # No, the task says shape (5, 100). Let me re-read.
        #
        # "prizes and weights, of shape (5, 100)" - both are (5, 100)
        # "best prize of that iteration" - suggests prize is a scalar per item
        # 
        # I think the prize for an item is the sum (or some aggregation) of prizes[:, i]
        # Or perhaps prizes is actually the same value repeated? No, it says uniform in [0,1]
        #
        # Let me assume: total_prize[i] = sum_c prizes[c, i] or mean_c prizes[c, i]
        # And weight[c, i] is the consumption of constraint c by item i
        #
        # For the shadow price approach:
        # marginal_utility[i] = total_prize[i] - sum_c(shadow_price[c] * weight[c, i])
        
        # Total prize per item: use sum across constraint dimensions
        total_prize = np.sum(prizes, axis=0)  # (100,)
        
        # Weighted cost: sum_c(shadow_price[c] * weight[c, i])
        weighted_cost = np.sum(shadow_prices[:, None] * weights, axis=0)  # (100,)
        
        # Marginal utility: prize minus cost
        h_marginal = total_prize - weighted_cost  # (100,)
        
        # --- Component 2: Soft-min of per-constraint efficiency ratios ---
        # For each item, compute prize/weight ratio per constraint
        safe_w = np.maximum(weights, eps)
        ratios = prizes / safe_w  # (5, 100)
        
        # Soft-min using log-sum-exp trick:
        # softmin(x) = -logsumexp(-x) = -log(sum(exp(-x)))
        # This is a smooth approximation to min(x)
        neg_ratios = -ratios
        # For numerical stability, subtract max
        max_neg = np.max(neg_ratios, axis=0, keepdims=True)  # (1, 100)
        exp_neg = np.exp(neg_ratios - max_neg)  # (5, 100)
        sum_exp = np.sum(exp_neg, axis=0)  # (100,)
        log_sum_exp = np.log(sum_exp + eps)  # (100,)
        soft_min_ratio = -(log_sum_exp - max_neg[0, :])  # (100,)
        
        # This gives a smooth approximation to min_c(prizes[c,i]/weights[c,i])
        
        # --- Component 3: Geometric slack (rewards items using less capacity) ---
        slack = 1.0 - weights / capacity  # (5, 100)
        slack = np.clip(slack, eps, 1.0)
        log_slack = np.log(slack)
        mean_log_slack = np.mean(log_slack, axis=0)  # (100,)
        h_slack = np.exp(mean_log_slack)  # (100,)
        
        # --- Normalize each component to [0, 1] ---
        def normalize_positive(arr):
            arr = np.where(arr > 0, arr, eps)
            mn = np.min(arr)
            mx = np.max(arr)
            if mx > mn:
                return (arr - mn) / (mx - mn)
            return np.ones_like(arr)
        
        h1 = normalize_positive(h_marginal)  # (100,)
        h2 = normalize_positive(soft_min_ratio)  # (100,)
        h3 = normalize_positive(h_slack)  # (100,)
        
        # --- Blend components ---
        # Weight marginal utility and soft-min heavily
        alpha1 = 0.45  # shadow-price marginal utility
        alpha2 = 0.40  # soft-min efficiency
        alpha3 = 0.15  # geometric slack
        
        h = alpha1 * h1 + alpha2 * h2 + alpha3 * h3
        
        # Ensure positive values
        h = np.where(h > 0, h, eps)
        
        # Sharpen: power transform to emphasize high-efficiency items
        # Use 1.5 exponent as suggested by knowledge
        h = np.power(h + eps, 1.5)
        
        # Cleanup: replace non-finite or non-positive with eps
        h = np.where(np.isfinite(h) & (h > 0), h, eps)
        
        # Scale to reasonable range
        max_h = np.max(h)
        if max_h > 0:
            h = h / max_h
        
        # Ensure minimum value
        h = np.maximum(h, 1e-9)
        
        # Ensure shape is (100,)
        h = h.reshape(n_items)
        
        # Time check
        if time.time() > deadline:
            return h
            
        return h
        
    except Exception:
        # Fallback: return a valid array of small positive numbers
        return np.full((100,), 1e-9)
# EVOLVE-BLOCK-END
