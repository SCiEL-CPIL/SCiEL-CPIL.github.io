import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Computes acquisition scores using Rank-Based Pareto Front Selection.
    
    Strategy:
    1. Compute Expected Improvement (EI) for all candidates.
    2. Identify the non-dominated set (Pareto front) in the (EI, Cost) space.
       A candidate A dominates B if EI_A >= EI_B and Cost_A <= Cost_B (and at least one is strict).
       Since we want to maximize EI and minimize Cost, the Pareto front consists of points 
       that are not dominated by any other point.
    3. Assign scores based on the rank of the candidate within the Pareto front.
       To ensure a smooth transition and handle the fact that only a few points are on the 
       strict front, we define a "dominance rank" for all points:
       - Points on the Pareto front get the highest ranks (closest to 1.0).
       - Points dominated by many points get lower ranks.
       - Specifically, we compute the number of points that dominate each candidate.
       - Score = 1 / (1 + num_dominating_points).
       
       This creates a strict trade-off: a point with slightly higher EI but much higher cost 
       might be dominated by a point with lower EI and much lower cost, thus receiving a 
       lower score. This is different from EI/Cost which can be fooled by extreme values.
       
       To make it more robust and ensure we don't ignore the magnitude of EI entirely, 
       we can combine the rank score with a normalized EI term, but the directive asks for 
       rank-based Pareto front selection. Let's stick to the dominance structure primarily.
       
       Actually, pure dominance rank might be too sparse. Let's refine:
       Compute the "Pareto Rank" (NSGA-II style) for all points.
       Rank 0 = Non-dominated (best trade-off).
       Rank 1 = Dominated only by Rank 0 points.
       Rank 2 = Dominated only by Rank 0 and 1 points.
       ...
       Score = 1 / (1 + Rank).
       
       This ensures that the best trade-off points are selected first. If there's a tie in rank,
       we can break it by EI magnitude to ensure higher EI is preferred among non-dominated points.
       
       However, calculating full NSGA-II ranks for 2000 points is O(N^2) which is 4,000,000 operations.
       This is feasible in Python/NumPy if vectorized, but might be slow in pure Python loops.
       Let's try to vectorize the dominance check.
       
       Dominance check:
       A dominates B if:
       (EI_A > EI_B AND Cost_A < Cost_B) OR
       (EI_A == EI_B AND Cost_A < Cost_B) OR
       (EI_A > EI_B AND Cost_A == Cost_B)
       
       For 2000 points, we can create a 2000x2000 matrix of dominance relationships.
       This is 4 million booleans. Memory is fine. Time might be tight if not optimized.
       
       Alternative: 
       Since we only have 2 objectives, the Pareto front is a monotonic curve.
       We can sort by EI descending. Then, a point is on the Pareto front if its cost is 
       lower than all previous points in the sorted list (which have higher or equal EI).
       
       Let's implement the efficient Pareto front extraction for 2D:
       1. Sort candidates by EI descending.
       2. Iterate through the sorted list. Keep track of the minimum cost seen so far.
       3. If current cost < min_cost_so_far, it's on the Pareto front. Update min_cost.
       
       This gives us the set of non-dominated points (Rank 0).
       
       For the rest, we can assign ranks based on how many Pareto front points dominate them?
       Or simply: 
       Score for Pareto front points: High score (e.g., 1.0).
       Score for others: Lower score, potentially based on their distance to the Pareto front.
       
       But the directive says: "assign scores proportional to their rank within this front".
       This implies we should only care about the front? Or rank *within* the front?
       "compute the non-dominated set... then assign scores proportional to their rank within this front"
       This suggests points NOT in the front get 0 or a very low score.
       
       If we only select from the Pareto front, we might miss good points that are slightly 
       dominated but have much higher EI. However, for cost-aware BO, the Pareto front 
       approach is a standard robust method.
       
       Let's calculate the Pareto front.
       For points on the front, we assign scores based on their rank in the front.
       How to rank within the front? 
       The front is a set of points. We can rank them by their "quality" in the front.
       Since they are non-dominated, we can't say one is strictly better than another in both objectives.
       We can use a utility function on the front, e.g., normalized EI + normalized (1-Cost).
       Or simply assign them all a high score, and penalize non-front points heavily.
       
       Let's try:
       1. Compute EI.
       2. Find Pareto Front (Non-dominated set).
       3. Assign score = 1.0 to points on the front.
       4. Assign score = 0.0 to points not on the front.
       
       This is very aggressive. It might lead to instability if the front is very small.
       
       Better: 
       Assign scores based on the "Pareto Rank" (NSGA-II).
       Rank 0: Front.
       Rank 1: Next layer.
       ...
       Score = 1 / (1 + Rank).
       
       To compute ranks efficiently for N=2000:
       We can do it iteratively.
       1. Find Front 0 (Rank 0).
       2. Remove Front 0 from the set.
       3. Find Front 1 (Rank 1) from the remaining set (considering dominance by ALL points, 
          but effectively, a point is Rank 1 if it is dominated only by Rank 0 points).
       
       This is O(N * R * N) in worst case, but usually fast.
       
       Let's implement a simpler version:
       Compute the number of points that dominate each point.
       Dominance count D_i = sum_j (1 if j dominates i).
       Score = 1 / (1 + D_i).
       
       This is O(N^2) to compute D_i.
       Vectorized:
       EI_diff = EI[:, None] - EI[None, :]  # N x N
       Cost_diff = Cost[:, None] - Cost[None, :]
       Dominates = (EI_diff > 0) & (Cost_diff < 0) | (EI_diff == 0) & (Cost_diff < 0) | (EI_diff > 0) & (Cost_diff == 0)
       # Actually, strict dominance:
       # A dominates B if (EI_A > EI_B AND Cost_A < Cost_B) OR (EI_A == EI_B AND Cost_A < Cost_B) OR (EI_A > EI_B AND Cost_A == Cost_B)
       # Let's use: (EI_A >= EI_B AND Cost_A <= Cost_B) AND NOT (EI_A == EI_B AND Cost_A == Cost_B)
       
       This matrix is 2000x2000 = 4M elements. 
       Creating it and summing is fast in NumPy.
       
       Let's try this approach.
    """
    start_time = time.time()
    deadline = 300.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)

    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # Check time
    if time.time() > deadline - 10.0:
        # Fallback: simple EI/Cost
        z = (f_best - mu) / sigma
        z_clipped = np.clip(z, -10.0, 10.0)
        phi_z = norm.pdf(z_clipped)
        Phi_z = norm.cdf(z_clipped)
        ei = (f_best - mu) * Phi_z + sigma * phi_z
        ei = np.maximum(ei, 0.0)
        return ei / cost

    # 1. Compute EI
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10.0, 10.0)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute Dominance Counts
    # We want to count how many points dominate each point.
    # Point j dominates point i if:
    # (EI_j > EI_i AND Cost_j < Cost_i) OR
    # (EI_j == EI_i AND Cost_j < Cost_i) OR
    # (EI_j > EI_i AND Cost_j == Cost_i)
    
    # Vectorized calculation
    # EI_j > EI_i: ei[None, :] > ei[:, None]
    # Cost_j < Cost_i: cost[None, :] < cost[:, None]
    
    # Create matrices
    # Note: memory is 2000x2000 floats/bools. 2000*2000*8 bytes = 32MB. Fine.
    
    ei_diff = ei[None, :] - ei[:, None]  # ei_j - ei_i
    cost_diff = cost[None, :] - cost[:, None]  # cost_j - cost_i
    
    # Dominance condition
    # Strictly better EI and Cost, or equal in one and better in other.
    # Let's use:
    # (ei_diff > 0 AND cost_diff < 0) OR
    # (ei_diff == 0 AND cost_diff < 0) OR
    # (ei_diff > 0 AND cost_diff == 0)
    
    # To avoid floating point equality issues, we can use a small epsilon or just rely on > and <.
    # Given the continuous nature, exact equality is rare.
    
    dominates = ((ei_diff > 0) & (cost_diff < 0)) | \
                ((ei_diff == 0) & (cost_diff < 0)) | \
                ((ei_diff > 0) & (cost_diff == 0))
    
    # Count how many points dominate each point i
    # dominates[j, i] is True if j dominates i.
    # We want sum over j for each i.
    # dominates is N x N where rows are j (dominator) and cols are i (dominated).
    # So we sum along axis 0.
    dominance_count = np.sum(dominates, axis=0)
    
    # 3. Assign Scores
    # Score = 1 / (1 + dominance_count)
    # This gives 1.0 to non-dominated points (count=0).
    # Points dominated by 1 other get 0.5, etc.
    
    scores = 1.0 / (1.0 + dominance_count)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Optional: If all scores are 0 (shouldn't happen), fallback
    if np.all(scores == 0):
        scores = ei / cost
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
        
    return scores
