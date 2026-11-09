import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins is an array of remaining capacities
    # item is the size of the current item
    
    # Identify feasible bins
    # The prompt says "bins with enough remaining capacity" are passed?
    # "returns np.ndarray of one score per bin with enough remaining capacity"
    # This implies the input `bins` might already be filtered?
    # "scores each bin with enough remaining capacity"
    # If `bins` contains ALL bins, we must handle infeasible ones.
    # The seed code `return -(bins - item)` would give high scores to infeasible bins (negative remaining -> positive score).
    # This is DANGEROUS.
    # If bins = [10, 20] and item = 15.
    # 10 - 15 = -5. Score = 5.
    # 20 - 15 = 5. Score = -5.
    # Max score is 5 (Bin 1). But Bin 1 is infeasible!
    
    # The prompt says: "scores each bin with enough remaining capacity"
    # Does the evaluator filter `bins` before passing?
    # "the item goes to the highest-scoring bin"
    # If I return a high score for an infeasible bin, the item might go there?
    # The contract says: "returns np.ndarray of one score per bin with enough remaining capacity"
    # This phrasing is ambiguous.
    # 1. The input `bins` contains ONLY feasible bins.
    # 2. I must return scores for ALL bins, but only feasible ones are considered?
    # 3. I must return scores for ALL bins, and I must ensure infeasible bins have very low scores.
    
    # Given the seed code `return -(bins - item)`, it does NOT mask infeasible bins.
    # If the seed scores 1.0, and it doesn't mask, then either:
    # a) The input `bins` is pre-filtered to only feasible bins.
    # b) The evaluator ignores infeasible bins despite high scores? Unlikely.
    # c) The seed is actually flawed but happens to score 1.0 because the test cases don't trigger infeasibility with high scores?
    #    Or maybe `bins` is always feasible?
    
    # Let's assume the safest interpretation:
    # The input `bins` contains ALL currently open bins.
    # I MUST ensure that infeasible bins (bins < item) get a very low score (e.g., -np.inf).
    
    # Let's check the seed again.
    # `return -(bins - item)`
    # If bins < item, score is positive.
    # If bins > item, score is negative.
    # Max score will be from the most infeasible bin (smallest bin).
    # This would be a catastrophic failure if `bins` is not pre-filtered.
    
    # However, the prompt says: "scores each bin with enough remaining capacity"
    # This might imply that the `bins` array passed to the function ALREADY ONLY CONTAINS FEASIBLE BINS.
    # "each arriving item goes to the highest-scoring bin"
    # If `bins` only contains feasible bins, then `bins - item >= 0`.
    # Then `-(bins - item)` is <= 0.
    # The highest score is the one closest to 0, i.e., smallest remaining space.
    # This is Best Fit.
    
    # I will assume `bins` contains ONLY FEASIBLE BINS based on the prompt description "scores each bin with enough remaining capacity".
    # If I am wrong and `bins` contains all bins, I should mask.
    # To be safe, I will mask infeasible bins with -np.inf.
    # This is strictly better than the seed if `bins` is not pre-filtered.
    # If `bins` IS pre-filtered, masking does nothing (all are feasible).
    
    # So, Step 1: Mask infeasible bins.
    # Step 2: Apply Heuristic.
    
    feasible = bins >= item
    scores = np.full_like(bins, -np.inf)
    
    if not np.any(feasible):
        # No feasible bin? The harness should handle this (open new bin).
        # But we must return something.
        # If we return -inf for all, max is -inf.
        # Let's just return the raw scores or -inf.
        return scores
    
    # Heuristic: Hybrid Best/Worst Fit
    # If item is small, use Worst Fit (maximize remaining).
    # If item is large, use Best Fit (minimize remaining).
    
    # Threshold: C/3? We don't know C.
    # We can estimate C from max(bins)? No, bins are remaining.
    # We can use a relative threshold?
    # Or just stick to Best Fit but with a twist.
    
    # Let's try Best Fit with a quadratic penalty for remaining space.
    # This makes it prefer "tighter" fits more.
    remaining = bins[feasible] - item
    
    # Score: minimize remaining^2?
    # This is equivalent to maximizing (C - remaining)^2?
    # Let's just use -remaining^2.
    # This is a "smooth" surrogate for min(remaining).
    
    scores[feasible] = -remaining**2
    
    return scores
