import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remainders
    remainders = bins - item
    
    # Valid bins are those with enough capacity
    valid = bins >= item
    
    # Initialize scores to -inf for invalid bins
    scores = np.full_like(bins, -np.inf)
    
    if not np.any(valid):
        return scores
        
    r = remainders[valid]
    
    # Model the packing process as a spectral clustering problem where the goal is to minimize 
    # the variance of the remaining capacities across all open bins.
    # Score each candidate bin by the negative of the resulting global variance of remainders.
    
    # To do this efficiently, we compute the change in variance if we place the item in a specific bin.
    # Let R be the vector of remainders for valid bins.
    # Let mean_R = mean(R).
    # Let var_R = var(R).
    # If we pick bin i, its remainder becomes r_i - item.
    # The new mean will be (sum(R) - item) / N.
    # The new variance will be sum((R_j - new_mean)^2) / N.
    
    # However, the bins that are NOT valid (bins < item) are not candidates. 
    # The prompt says "scores each bin with enough remaining capacity".
    # The bins that don't have enough capacity are effectively "closed" or "blocked" for this item.
    # Do we consider them in the variance calculation?
    # "variance of the remaining capacities across all open bins"
    # Usually, "open bins" means bins that are currently in use. 
    # In the online packing context, `bins` array likely contains the *current* remaining capacity of all bins that have been opened so far.
    # Bins with 0 capacity might be in the array? Or only active bins?
    # The input `bins` is an np.ndarray.
    # If a bin has 0 capacity, it is not "open" in the sense that it can't take items. 
    # But if it's in the array, it contributes to the statistics.
    # Let's assume `bins` contains all currently allocated bins.
    # The strategy suggests minimizing variance of remainders.
    
    # Let's try to compute the score based on the change in variance of the *valid* bins only?
    # Or all bins?
    # "across all open bins". If a bin is full (0), is it open? 
    # If we include full bins (0), their variance contribution is fixed.
    # Let's assume `bins` contains the state of all bins that have been created.
    
    # Let's calculate the variance of the *current* remainders of all bins in `bins`.
    # Then for each valid bin i, calculate the new variance if `bins[i]` becomes `bins[i] - item`.
    
    # Efficient calculation:
    # Let S = sum(bins)
    # Let S2 = sum(bins**2)
    # Let N = len(bins)
    # Current Variance V = (S2 - S**2/N) / N  (using population variance formula for stability? or sample?)
    # Let's use population variance (divide by N) as it's common in these heuristic contexts, or just proportional.
    
    # New state for bin i:
    # S_new = S - item
    # S2_new = S2 - bins[i]**2 + (bins[i] - item)**2
    # S2_new = S2 - bins[i]**2 + bins[i]**2 - 2*bins[i]*item + item**2
    # S2_new = S2 - 2*bins[i]*item + item**2
    
    # V_new_i = (S2_new - S_new**2/N) / N
    
    # We want to minimize V_new_i. So score = -V_new_i.
    
    # Let's check if this is equivalent to minimizing the term dependent on i.
    # V_new_i = (S2 - 2*bins[i]*item + item**2 - (S - item)**2/N) / N
    # Constant terms: (S2 + item**2 - S**2/N + 2*S*item/N - item**2/N) / N
    # Variable term: -2*bins[i]*item / N
    
    # So minimizing V_new_i is equivalent to maximizing bins[i].
    # This suggests that minimizing variance leads to a "Worst Fit" strategy (pick the bin with the largest remainder/capacity).
    
    # Let's re-read the inspiration: "minimize the variance of the remaining capacities... encourages a homogeneous set of gaps".
    # If we pick the largest bin, we make it smaller, reducing the spread.
    # If we pick the smallest bin (Best Fit), we make it even smaller, potentially increasing the spread if it was already small.
    # So "Minimize Variance" -> "Worst Fit".
    
    # However, the current program is a blend of Best Fit and Neutral.
    # The directive says: "KEEP this design and investigate improvements within its family."
    # "Family" = Adaptive weight between Best Fit and Neutral?
    # Or "Family" = The specific heuristic structure?
    # The strategy to try is: "Model the packing process as a spectral clustering problem... Score each candidate bin by the negative of the resulting global variance".
    
    # If I implement "Score = -V_new", it effectively becomes Worst Fit (maximize bins[i]).
    # Is Worst Fit in the family of "Best Fit / Neutral"? No, it's the opposite of Best Fit.
    # But the directive says "Strategy to try in this change: ...".
    # So I should implement the Variance Minimization strategy.
    
    # Let's implement the variance minimization score.
    # Score_i = -V_new_i
    
    # We can just compute the term that varies with i, since constants don't affect argmax.
    # V_new_i proportional to -bins[i]
    # So Score_i proportional to bins[i]
    # This is simply Worst Fit.
    
    # Wait, is it that simple?
    # Let's check the math again.
    # V = E[X^2] - (E[X])^2
    # V_new = (1/N) * sum_{j!=i} (x_j - mean_new)^2 + (1/N) * (x_i - item - mean_new)^2
    # This is slightly more complex because mean_new depends on the choice?
    # No, mean_new = (S - item) / N. It is independent of WHICH bin i we pick, as long as we subtract `item` from SOME bin.
    # Yes, the sum S decreases by `item` regardless of which bin.
    # So mean_new is constant for all candidates.
    
    # Let's expand V_new_i:
    # V_new_i = (1/N) * [ sum_{j!=i} (x_j - m)^2 + (x_i - item - m)^2 ]
    # where m = (S - item) / N
    
    # Let C = sum_{j!=i} (x_j - m)^2. This depends on i because the set {j!=i} changes.
    # C = sum_{all j} (x_j - m)^2 - (x_i - m)^2
    
    # So V_new_i = (1/N) * [ sum_{all j} (x_j - m)^2 - (x_i - m)^2 + (x_i - item - m)^2 ]
    
    # Let K = sum_{all j} (x_j - m)^2. This is constant for all i (since m is constant).
    # V_new_i = (1/N) * [ K - (x_i - m)^2 + (x_i - item - m)^2 ]
    
    # Let d = x_i - m.
    # Term = -d^2 + (d - item)^2 = -d^2 + d^2 - 2*d*item + item^2 = -2*item*(x_i - m) + item^2
    
    # So V_new_i = (1/N) * [ K - 2*item*(x_i - m) + item^2 ]
    
    # To minimize V_new_i, we need to maximize the term being subtracted: 2*item*(x_i - m).
    # Since item > 0, we need to maximize (x_i - m), which means maximizing x_i.
    
    # So, minimizing the new variance is exactly equivalent to picking the bin with the largest current remainder (Worst Fit).
    
    # So the "Spectral Clustering / Variance Minimization" strategy reduces to Worst Fit.
    
    # The current program is: score = -alpha * r, where alpha = item/100.
    # This is a blend of Best Fit (score=-r) and Neutral (score=0).
    # Worst Fit would be score = +r.
    
    # The directive says "KEEP this design... investigate improvements within its family".
    # If I switch to Worst Fit, am I staying in the family?
    # The family is "Adaptive weight".
    # Maybe I can blend Best Fit and Worst Fit?
    # Or maybe the "Variance" insight is just a justification for a specific weight?
    
    # Let's look at the "Neutral" part.
    # If we consider the variance of the *entire* system including closed bins?
    # If we assume the "family" allows us to change the target score from "Best Fit" to "Worst Fit" or a blend, we could try:
    # score = beta * r - (1-beta) * r = (2*beta - 1) * r
    # If beta=0.5, score=0 (Neutral).
    # If beta=1, score=r (Worst Fit).
    # If beta=0, score=-r (Best Fit).
    
    # The current program uses alpha = item/100 to blend Best Fit (-r) and Neutral (0).
    # score = -alpha * r.
    # This is biased towards Best Fit for large items.
    
    # If the Variance argument holds, and Variance Minimization = Worst Fit,
    # perhaps we should blend Worst Fit and Neutral?
    # score = alpha * r.
    # For large items, Worst Fit. For small items, Neutral.
    
    # Why would Worst Fit be better for large items?
    # If I have a large item, putting it in a large bin leaves a smaller remainder than putting it in a small bin?
    # No.
    # Bin 1: 90, Bin 2: 10. Item: 50.
    # BF: Bin 2 (10-50 invalid? No, 10<50). Only Bin 1 valid.
    # Let's say Bin 1: 90, Bin 2: 80. Item: 50.
    # BF: Bin 2 (rem 30) or Bin 1 (rem 40)? BF picks Bin 2 (rem 30).
    # WF: Bin 1 (rem 40).
    # Variance of {40, 80} vs {90, 30}.
    # Mean of {40, 80} = 60. Var = ((40-60)^2 + (80-60)^2)/2 = (400+400)/2 = 400.
    # Mean of {90, 30} = 60. Var = ((90-60)^2 + (30-60)^2)/2 = (900+900)/2 = 900.
    # So BF (picking Bin 2) resulted in LOWER variance?
    # Wait.
    # Initial: {90, 80}. Mean 85. Var = (25+25)/2 = 25.
    # Option BF (Bin 2 -> 30): State {90, 30}. Mean 60. Var = 900.
    # Option WF (Bin 1 -> 40): State {40, 80}. Mean 60. Var = 400.
    # So WF resulted in lower variance than BF in this case.
    # My previous derivation said Min Variance -> Max x_i (Worst Fit).
    # Here x_1=90, x_2=80. Max is 90 (Bin 1).
    # Picking Bin 1 (WF) gave Var 400.
    # Picking Bin 2 (BF) gave Var 900.
    # So yes, WF minimizes variance.
    
    # So the "Variance Minimization" strategy is Worst Fit.
    
    # The current program is Best Fit for large items.
    # The proposed strategy is Worst Fit for large items (if we map alpha to weight of WF).
    # Or maybe we should just switch to Worst Fit?
    
    # Let's try blending Worst Fit and Neutral.
    # score = alpha * r, where alpha = item / 100.
    # Large item -> Worst Fit.
    # Small item -> Neutral.
    
    # This is a direct inversion of the current program's logic regarding the fit direction, but keeps the "adaptive weight" structure.
    
    # Let's implement this.
    
    alpha = item / 100.0
    scores[valid] = alpha * r
    
    return scores
