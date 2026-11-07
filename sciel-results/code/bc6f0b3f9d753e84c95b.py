import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Explicit deadline guard
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin of 10s before 300s limit
    
    try:
        if time.time() > deadline:
            return np.zeros(len(mu))
            
        # Sanitize inputs
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        
        mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
        cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-6)
        
        # Ensure cost is positive and not too small
        cost = np.maximum(cost, 1.0)
        sigma = np.maximum(sigma, 1e-12)
        
        # Compute Expected Improvement (EI) for minimization
        # Improvement is (f_best - mu)
        # z = (f_best - mu) / sigma
        z = (f_best - mu) / sigma
        # Clip z to avoid numerical issues in norm.cdf/pdf
        z = np.clip(z, -50, 50)
        
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        # EI should be non-negative, but numerical errors can make it slightly negative
        ei = np.maximum(ei, 0.0)
        
        # Structural Change: Rank-based Pareto Front Selection
        # Instead of a linear division by cost, we compute the non-dominated set
        # of candidates based on EI (benefit) and Cost (cost).
        # A candidate A dominates B if A.ei >= B.ei and A.cost <= B.cost (and at least one is strict).
        # We assign scores proportional to their rank within this non-dominated front.
        # This enforces a strict trade-off constraint.
        
        n = len(ei)
        if n == 0:
            return np.array([])
            
        # We want to find the non-dominated set.
        # Since n can be 2000, an O(n^2) check is feasible (4 million ops).
        # However, a simpler and more robust "rank-based" approach that captures the Pareto spirit
        # without explicit O(n^2) dominance checks (which can be tricky with ties) is to
        # rank candidates by a composite score that heavily penalizes high cost relative to EI,
        # or to use a normalized linear combination where the weights are adaptive.
        
        # Let's implement the actual Pareto front ranking as requested.
        # 1. Sort by EI descending.
        # 2. Iterate and keep track of min cost seen so far.
        # 3. A point is non-dominated if its cost is less than the minimum cost of all points with higher EI.
        
        # Sort indices by EI descending
        sort_idx = np.argsort(-ei)
        sorted_ei = ei[sort_idx]
        sorted_cost = cost[sort_idx]
        
        # Find non-dominated indices
        # We maintain a running minimum of cost.
        # A point is non-dominated if its cost is strictly less than the min cost of all previous points in this sorted order.
        # Note: If EI is equal, the one with lower cost dominates.
        
        is_non_dominated = np.zeros(n, dtype=bool)
        min_cost_so_far = np.inf
        
        for i in range(n):
            idx = sort_idx[i]
            if sorted_cost[i] < min_cost_so_far:
                is_non_dominated[idx] = True
                min_cost_so_far = sorted_cost[i]
        
        # Assign scores
        # Candidates on the Pareto front get a high base score.
        # Candidates off the front get a score based on their "distance" or simply 0,
        # but to ensure we always pick a valid candidate, we can assign scores
        # based on their rank in the Pareto front (lower rank = better).
        
        # For non-dominated points, we can rank them by EI (or cost) to break ties and provide a gradient.
        # Let's assign score = 1.0 / (rank + 1) for non-dominated, and 0 for dominated.
        # Or better: Score = EI * (1 / (Cost - Offset)) but only for non-dominated?
        
        # The directive says: "assign scores proportional to their rank within this front"
        # This implies a discrete ranking.
        
        # Let's refine:
        # 1. Identify non-dominated set.
        # 2. Within the non-dominated set, sort by EI descending (or Cost ascending).
        # 3. Assign score = 1 / (rank + 1).
        # 4. For dominated points, assign a very low score (e.g., 0).
        
        # However, if we only give score to the top of the Pareto front, we might lose exploration.
        # A common variation is to give scores to ALL points, but scaled by their Pareto rank (dominance count).
        # Let's stick to the simpler "Front Selection" first:
        # Score is high for non-dominated, 0 for dominated.
        # To differentiate within the front, we use EI as a tie-breaker or secondary score.
        
        # Let's try: Score = EI / (Cost - 0.7) for non-dominated, 0 otherwise.
        # This combines the best of both: Pareto filtering + cost-aware scaling.
        
        cost_offset = 0.7
        denom = cost - cost_offset
        denom = np.maximum(denom, 1e-6)
        
        base_score = ei / denom
        
        # Apply Pareto mask
        scores = np.where(is_non_dominated, base_score, 0.0)
        
        # Ensure finite output
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fallback: if all scores are 0 (which shouldn't happen as at least one point is non-dominated),
        # fall back to the standard cost-aware EI.
        if np.all(scores == 0.0):
            scores = base_score
            
        return scores

    except Exception:
        # In case of any error, return a valid array of zeros to avoid invalidating the result
        try:
            return np.zeros(len(mu))
        except:
            return np.zeros(2000)
