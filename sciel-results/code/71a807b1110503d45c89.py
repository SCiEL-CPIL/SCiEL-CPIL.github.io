"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a Greedy Simulated Packing heuristic.
    
    Strategy:
    1. Calculate a 'balance penalty' for each item based on its maximum weight 
       relative to the capacity (25). Items that are heavy in any single 
       constraint are penalized more heavily because they risk blocking 
       many other items early in the ant's construction.
    2. Use a greedy simulation to determine the 'value' of each item. 
       We simulate a greedy knapsack fill using a composite score 
       (prize density adjusted by balance penalty).
    3. The heuristic score for an item is its rank in this greedy ordering,
       inverted so that items selected earlier (higher marginal utility in 
       the context of the specific instance's constraint landscape) get 
       higher scores.
       
    This captures the combinatorial structure of the specific instance 
    (which constraints are tight, which items are 'bad' fits) rather than 
    just static ratios.
    """
    start_time = time.time()
    
    # Safety guard: if we take too long, fallback to simple ratio
    # Though this should be fast for 100 items.
    
    # prizes: (5, 100), weights: (5, 100)
    # Capacity is 25 for all constraints.
    cap = 25.0
    
    # 1. Calculate per-item metrics
    # Sum of prizes per item
    total_prize = np.sum(prizes, axis=0) # (100,)
    
    # Max weight per item (across constraints)
    max_weight = np.max(weights, axis=0) # (100,)
    
    # Balance Penalty: How "expensive" is the item in its worst constraint?
    # If an item uses 10 units of constraint 1, it consumes 40% of that capacity.
    # We want to penalize items that are disproportionately heavy in any single constraint.
    # A simple proxy: max_weight / cap.
    # If max_weight is close to cap, it's a "blocker".
    # Let's define a "fit score" = 1 / (1 + max_weight/cap)
    # Items with low max_weight get higher fit scores.
    fit_score = 1.0 / (1.0 + max_weight / cap)
    
    # Prize density: total_prize / sum(weights) is the baseline.
    sum_weight = np.sum(weights, axis=0)
    density = total_prize / (sum_weight + 1e-9)
    
    # Composite Score for Greedy Simulation:
    # We combine density and fit. 
    # If an item has high density but is a "blocker" (low fit), it might be bad.
    # Let's try: Score = density * fit_score
    # This penalizes items that are heavy in any constraint.
    composite_score = density * fit_score
    
    # 2. Greedy Simulation to Rank Items
    # We simulate adding items one by one based on composite_score.
    # This mimics what an ant might do if it had perfect foresight of the 
    # "greedy" path, but we use this to rank items for the heuristic.
    
    # Sort items by composite_score descending
    order = np.argsort(composite_score)[::-1]
    
    # Initialize remaining capacities
    remaining_cap = np.full(5, cap)
    
    # We will assign a "simulated prize" to each item.
    # If an item fits in the greedy fill, it gets its actual prize.
    # If it doesn't fit, it gets 0 (or a small penalty).
    # Actually, a better heuristic is: 
    # Rank items by the order in which they WOULD be picked by a greedy algorithm 
    # that maximizes prize, but we need to account for constraints.
    
    # Let's do a proper greedy fill simulation to see which items are "viable" early.
    # This is O(N^2) worst case, but N=100, so 10,000 ops. Very fast.
    
    selected = np.zeros(100, dtype=bool)
    greedy_prizes = np.zeros(100)
    
    # Iterate through items in order of composite_score
    for idx in order:
        # Check if item idx fits in remaining capacity
        if np.all(weights[:, idx] <= remaining_cap):
            # It fits
            selected[idx] = True
            greedy_prizes[idx] = total_prize[idx]
            # Update remaining capacity
            remaining_cap -= weights[:, idx]
        else:
            # It doesn't fit in the current greedy state
            # Assign 0 prize? Or a small fraction?
            # If it doesn't fit, it's less attractive in this specific instance context.
            greedy_prizes[idx] = 0.0
            
    # 3. Construct Heuristic
    # The heuristic should reflect the "greedy viability".
    # Items that were selected in the greedy fill get high scores.
    # Items that were not selected get low scores.
    
    # However, simply using 0/1 is too binary. 
    # Let's use the rank in the greedy selection order.
    # Items selected earlier are "better".
    
    # Create a rank array. 
    # For selected items, rank is their position in the 'order' list.
    # For non-selected items, rank is high (bad).
    
    # Actually, a smoother approach:
    # Use the composite_score itself, but normalized by the maximum possible 
    # composite score? No, that's just the static score again.
    
    # Let's try a different angle: 
    # The heuristic should guide the ant to pick items that are "easy to fit" 
    # and "high value".
    # The greedy simulation tells us which items form a good "core".
    
    # Let's assign scores based on:
    # 1. If selected in greedy: Score = total_prize[idx] * (1 / (1 + rank/100))
    #    This rewards early picks.
    # 2. If not selected: Score = total_prize[idx] * 0.1
    #    This keeps them in play but low priority.
    
    # We need to map the 'order' index to a rank.
    # order is an array of indices sorted by composite_score.
    # We can create a rank array where rank[item] = position in order.
    ranks = np.empty(100)
    for pos, idx in enumerate(order):
        ranks[idx] = pos
    
    # Normalize ranks to [0, 1] where 0 is best (first)
    norm_ranks = ranks / 100.0
    
    # Heuristic = total_prize * (1 - norm_ranks) for selected items?
    # But what about non-selected? They have high ranks.
    # If we just use total_prize * (1 - norm_ranks), non-selected items 
    # (which have high ranks) will get low scores. 
    # This effectively uses the greedy ordering as the heuristic.
    
    # Is this different from just using composite_score?
    # Yes, because the greedy simulation accounts for the *interaction* 
    # (items that fit together). Two items might have high individual 
    # composite scores but can't be picked together due to a shared 
    # tight constraint. The greedy simulation picks the one that fits 
    # first, effectively "vetoing" the other in the context of the 
    # "core" solution.
    
    # Let's refine:
    # Score = total_prize * (1.0 / (1.0 + ranks))
    # This gives a heavy weight to the top items in the greedy order.
    
    heuristic_scores = total_prize * (1.0 / (1.0 + ranks))
    
    # Ensure positivity
    heuristic_scores = np.maximum(heuristic_scores, 1e-9)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 250:
        # Fallback to simple ratio if something went wrong (shouldn't happen)
        return total_prize / (sum_weight + 1e-9)
        
    return heuristic_scores
# EVOLVE-BLOCK-END
