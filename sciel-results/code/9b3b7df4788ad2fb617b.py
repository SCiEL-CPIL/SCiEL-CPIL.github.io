import numpy as np

def heuristics(prizes, weights):
        # prizes: (5, 100), weights: (5, 100)
        # Capacity C = 25
        C = 25.0
        K, N = weights.shape
        
        # 1. Calculate tightness of each constraint
        # Sum of weights for each constraint
        W_sum = np.sum(weights, axis=1) # shape (5,)
        
        # Tightness score: higher means tighter
        # Using W_sum / C as a proxy for how "full" the knapsack would be if we took all items
        # Or just W_sum is fine, relative scale matters
        tightness = W_sum / C
        
        # Normalize tightness to get weights lambda
        # Avoid division by zero
        total_tightness = np.sum(tightness)
        if total_tightness > 0:
            lambdas = tightness / total_tightness
        else:
            lambdas = np.ones(K) / K
            
        # 2. Compute heuristic
        # eta_i = p_i / exp(sum_j lambda_j * ln(w_ji))
        # Using log-space for stability
        # w_ji > 0 since uniform [0,1] but might be very close to 0.
        # Add epsilon to avoid log(0)
        eps = 1e-9
        w_safe = weights + eps
        
        log_w = np.log(w_safe) # shape (5, 100)
        
        # Weighted sum of log weights
        # lambdas shape (5,), log_w shape (5, 100)
        # Result shape (100,)
        weighted_log_w = np.sum(lambdas[:, np.newaxis] * log_w, axis=0)
        
        # Denominator is exp(weighted_log_w)
        # eta = p_i / exp(weighted_log_w)
        # p_i is prizes[i] ? No, prizes shape is (5, 100). 
        # Wait, the task says "prizes and weights, of shape (5, 100)".
        # This implies there are 5 prizes per item? Or 5 constraints and 100 items, and prizes is also (5, 100)?
        # "Item scores... prizes and weights... shape (5, 100)".
        # Usually in MCKP, each item has ONE prize and K weights.
        # But here prizes is (5, 100). This is unusual.
        # Let's re-read carefully: "prizes and weights, of shape (5, 100), are uniform in [0, 1]."
        # "Every constraint has capacity 25".
        # "20 ants... adds items... until none fits".
        # "best prize of that iteration".
        # If prizes is (5, 100), does each item have 5 prizes?
        # Or is it a typo in my interpretation?
        # "mean over instances of the highest total prize".
        # If an item is selected, what is its prize?
        # If prizes is (5, 100), maybe the "total prize" is the sum of all 5 prize components?
        # Or maybe the problem is actually 5 independent knapsacks? No, "multiple-constraint knapsack".
        # In MCKP, usually $p_i$ is scalar.
        # If `prizes` is (5, 100), it's possible that the "prize" of an item is the sum of its entries in the `prizes` matrix?
        # Or maybe the 5 rows correspond to the 5 constraints, and there's a specific prize associated with each constraint?
        # Let's assume the "value" of item $i$ is the sum of its prizes across the 5 rows, or perhaps the prizes matrix is actually just the weights for a different objective?
        # No, "prizes" and "weights" are distinct.
        # Let's assume the total prize of an item $i$ is $P_i = \sum_{k=0}^4 prizes[k, i]$.
        # This makes the most sense for a scalar objective "total prize".
        
        item_prizes = np.sum(prizes, axis=0) # shape (100,)
        
        # Calculate eta
        # eta_i = item_prizes[i] / exp(weighted_log_w[i])
        eta = item_prizes / np.exp(weighted_log_w)
        
        # Handle non-positive or non-finite
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        return eta
