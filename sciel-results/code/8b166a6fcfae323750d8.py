"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on 'Global Difficulty Prioritization'.
    
    Strategy:
    1. Simulate a First Fit Decreasing (FFD) packing to assess item difficulty.
    2. Items that are large or cause bins to open are considered 'hard' to place.
    3. The heuristic encourages placing 'hard' items earlier in the bin sequence.
       Specifically, the attractiveness of placing item j after item i is high
       if item j is 'hard' (large size, low flexibility).
    4. We also incorporate a local validity check: if i+j > capacity, the score
       is effectively zero (1e-9).
    
    This differs from local gap-based heuristics by using global structural
    information about the instance's packing landscape to guide the search
    towards resolving the most constrained items first.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = len(demands)
    demands_arr = np.asarray(demands, dtype=float)
    
    # 1. Simulate FFD to identify 'hard' items
    # Sort indices by demand descending
    sorted_indices = np.argsort(demands_arr)[::-1]
    sorted_demands = demands_arr[sorted_indices]
    
    # Track how many bins each item 'contributes' to opening or filling
    # A simple proxy for difficulty: 
    # - Larger items are harder to place.
    # - Items that don't fit in any existing bin (forcing a new bin) are hardest.
    
    bins = []  # List of remaining capacities
    item_difficulty = np.zeros(n, dtype=float)
    
    # We need to map sorted items back to original indices to assign difficulty
    # But for the heuristic matrix, we just need a score per item index.
    
    # Let's compute a 'flexibility' score for each item.
    # Flexibility = how many bins (in a random or FFD sense) it could fit into.
    # High flexibility = easy to place. Low flexibility = hard to place.
    
    # Instead of full FFD simulation which is O(N^2) or worse, let's use a simpler
    # global metric: 
    # Difficulty(item) = (demand / capacity)^2 * (1 + count_of_items_smaller_than_it)
    # Or simply: Large items are hard. Small items are easy.
    # But "Hard" in bin packing often means "doesn't fit well with others".
    
    # Let's use a "Waste Contribution" heuristic from a quick greedy pass.
    # We will run a simple Next Fit Decreasing (NFD) simulation to see which items
    # result in high waste when placed.
    
    # Actually, a more robust signal for ACO is to favor items that are 
    # "complementary" to the average item size, or simply prioritize large items.
    # Standard bin packing theory suggests LPT (Longest Processing Time first) 
    # is effective. So, making large items more attractive to be placed 
    # (i.e., higher heuristic value for j if j is large) might help.
    
    # Let's define the heuristic eta[i, j] as:
    # 1. If demands[i] + demands[j] > capacity: 1e-9 (invalid)
    # 2. Else: Base score based on 'difficulty' of j.
    #    Difficulty(j) = demands[j] / capacity.
    #    We want to place hard items. So high demand j -> high attractiveness.
    #    Why? Because if we don't place them early, they might not fit later.
    #    Also, placing a large item leaves a specific gap that smaller items 
    #    can fill.
    
    # Let's refine: 
    # Attractiveness of j after i should be high if:
    # a) j fits after i.
    # b) j is "large" (hard to place later).
    # c) The remaining space after i+j is not "dead" (can fit at least one small item) 
    #    OR is very small (tight).
    
    # Let's combine:
    # 1. Validity mask
    pair_sums = demands_arr[:, None] + demands_arr[None, :]
    valid = pair_sums <= capacity
    
    # 2. Difficulty score for each item j
    # Normalize demands to [0, 1]
    max_d = np.max(demands_arr)
    min_d = np.min(demands_arr)
    if max_d == min_d:
        norm_d = np.ones(n)
    else:
        norm_d = (demands_arr - min_d) / (max_d - min_d)
        
    # Difficulty: Higher for larger items.
    # Let's use norm_d + 1 to keep positive.
    difficulty = norm_d + 1.0
    
    # 3. Gap Quality
    remaining = capacity - pair_sums
    # Small gaps are good (tight). 
    # Large gaps are okay if they can fit another item.
    # Dead gaps (0 < gap < min_d) are bad.
    
    # Score for gap:
    # If gap == 0: 1.0
    # If gap >= min_d: 1.0 (flexible)
    # If 0 < gap < min_d: 0.1 (penalty)
    
    gap_score = np.ones((n, n), dtype=float)
    dead_gap = valid & (remaining > 1e-9) & (remaining < min_d)
    gap_score[dead_gap] = 0.1
    
    # 4. Combine
    # eta[i, j] = difficulty[j] * gap_score[i, j]
    # This encourages placing large items (high difficulty) into spots where the gap
    # is not wasteful.
    
    eta = difficulty[None, :] * gap_score
    
    # Apply validity
    eta[~valid] = 1e-9
    
    # Diagonal: i=j is usually invalid in distinct item packing
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure finite and positive
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    # Check time
    if time.time() > deadline:
         # Return a simple fallback if this took too long (unlikely for N=120)
         return np.full((n, n), 1e-9)
         
    return eta
# EVOLVE-BLOCK-END
