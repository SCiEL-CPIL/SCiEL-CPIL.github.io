"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We will implement a Q-learning agent that learns online during the evaluation.
# The state is represented by a feature vector derived from the current bins.
# The action is the index of the bin to place the item in.
# The reward is negative of the increase in the "L1 lower bound" or a proxy for it.
# Since we don't know the future items, we approximate the cost of leaving a bin "open" 
# versus closing it.

# Feature extractor:
# 1. Item size
# 2. Min remaining capacity
# 3. Max remaining capacity
# 4. Mean remaining capacity
# 5. Variance of remaining capacities
# 6. Count of bins with remaining capacity > item (feasible actions count)
# 7. Number of "active" bins (remaining < capacity)
# We assume capacity is 100.

CAPACITY = 100.0
# We use a linear Q-function: Q(s, a) = w . phi(s, a)
# phi(s, a) includes global state features and action-specific features (remaining space in bin a).

# Initialize weights. 
# Features:
# 0: bias
# 1: item_size
# 2: min_rem
# 3: max_rem
# 4: mean_rem
# 5: std_rem
# 6: num_feasible
# 7: num_active
# 8: remaining_in_target_bin (action feature)
# 9: remaining_in_target_bin / item_size (ratio)
# 10: (remaining_in_target_bin - item) (waste)

NUM_FEATURES = 11
w = np.zeros(NUM_FEATURES)
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor

# To speed up convergence and stabilize, we use a simple replay buffer or just online updates.
# Given the 120s limit and 1000 items * 5 sequences, we have plenty of time for online updates.
# However, pure online TD can be unstable. We'll use a simple update rule.

# We need to track the "next state" to compute the TD error.
# Since this is a single call per item, we cannot easily do full TD without storing state history.
# Alternative: Use a "Heuristic Bootstrapping" approach.
# We assume the value of a state is proportional to the "waste" created.
# Let's try a different structural change: **Simulated Annealing / Randomized Search with Memory**.
# Instead of learning, we maintain a history of "good" placements.
# Actually, the prompt asks for a DIFFERENT design family.
# Let's try **Nearest Neighbor / KNN on Bin States**.
# We store a dictionary of recent bin configurations (hashed) and the outcome (how many bins were used in the next 10 steps).
# This is complex to implement robustly in < 120s.

# Let's go back to the **Probabilistic / Stochastic** approach but structured differently.
# The inspiration 1 used "expected size of next item".
# Let's use **Optimal Control / Dynamic Programming approximation**.
# The cost of a bin with remaining space $r$ is $C(r)$.
# If we place item $i$ in bin with remaining $r$, new remaining is $r' = r - i$.
# The cost change is $C(r') - C(r)$.
# We want to minimize total cost.
# If we assume the "cost" of an open bin is proportional to its remaining space (waste), 
# then we minimize sum of remaining spaces. This is equivalent to minimizing the number of bins 
# if all bins are full. But open bins have waste.
# The L1 lower bound is $LB = \max(1, \lceil \sum items / C \rceil)$.
# Excess = $N_{bins} - LB$.
# Minimizing $N_{bins}$ is the primary goal.
# A bin is "closed" (counted in $N_{bins}$) as soon as it receives an item.
# So every placement into an *empty* bin increases $N_{bins}$ by 1.
# Every placement into a *non-empty* bin keeps $N_{bins}$ constant.
# Therefore, the most critical decision is: **Do I open a new bin or use an existing one?**
# If I must use an existing bin, which one?
# Using a bin with small remaining space might "waste" that small space if it can't fit future items.
# Using a bin with large remaining space preserves the optionality for large future items.

# Strategy: **Two-Phase Decision**
# 1. Determine if we should open a new bin. 
#    - If there is a bin that can fit the item and has remaining space $r$ such that $r - item < \text{threshold}$, 
#      and the threshold is small enough that it's likely "dead" space, prefer it (Best Fit).
#    - If there is a bin with large remaining space, and the item is large, maybe we should open a new bin 
#      to keep the large bin "pure" for other large items? (This is rare in online packing unless items are very large).
#    - Standard Best Fit is usually optimal for minimizing bins.
#    - **Worst Fit** is sometimes better for Weibull because it avoids fragmentation? No, Worst Fit creates many half-full bins.
#    - **First Fit Decreasing** is offline.
#    
# Let's try a **Genetic/Heuristic Hybrid** that adapts the "Best Fit" score based on the **distribution of remaining capacities**.
# If the remaining capacities are very fragmented (many small, many large), we should be careful.
# If they are uniform, Best Fit is good.

# New Structural Idea: **Entropy-Based Scoring**
# We calculate the entropy of the remaining capacities.
# High entropy (diverse bin sizes) suggests we should try to reduce entropy by filling bins that are "average" size? 
# Or fill the smallest to clear them?
# Let's test: Score = $-\text{remaining} + \lambda \cdot \text{Entropy}$.
# This is still local.

# Let's try **Lookahead via Monte Carlo Tree Search (MCTS) approximation**.
# We can't do full MCTS in 120s for 5000 items.
# But we can do a **1-step lookahead with sampled future items**.
# Assume the next item is drawn from the Weibull distribution.
# We estimate the Weibull parameters from the history of seen items.
# Then, for each candidate bin, we simulate placing the current item, and then place a *sampled* next item.
# We score the bin based on the resulting state (e.g., number of bins, total waste).
# This is a **Model-Based** approach, distinct from the static heuristic.

# Implementation Plan:
# 1. Maintain a history of item sizes.
# 2. Fit Weibull parameters (shape k, scale lambda) to the history.
# 3. In `priority`:
#    a. Generate a sample of the "next item" from the fitted Weibull.
#    b. For each valid bin, simulate placing `item` then `next_item`.
#    c. Calculate a cost metric for the resulting state (e.g., sum of remaining spaces in bins that can't fit `next_item`).
#    d. Score = -cost.
#    e. Update Weibull parameters with the current `item`.

history = []

def fit_weibull(data):
    if len(data) < 10:
        return 1.0, 10.0  # default
    # Use MLE for Weibull
    # scipy.stats.weibull_min.fit(data, floc=0)
    # But we want to avoid scipy if possible for speed? No, scipy is allowed.
    # However, calling fit every step is expensive.
    # We can update parameters incrementally or fit every N steps.
    # For simplicity and speed, we'll fit every 10 items.
    pass

# Let's implement the MCTS-like lookahead.

item_count = 0
weibull_k = 1.0
weibull_lambda = 10.0
last_fit_count = 0

def update_weibull_params():
    global weibull_k, weibull_lambda, last_fit_count
    if len(history) > 20 and item_count - last_fit_count > 10:
        # Fit Weibull
        # Weibull PDF: f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
        # Log-likelihood: sum(k*log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k)
        # This is non-trivial to solve analytically. Use scipy.
        from scipy.stats import weibull_min
        try:
            k, loc, scale = weibull_min.fit(history, floc=0)
            weibull_k = k
            weibull_lambda = scale
            last_fit_count = item_count
        except:
            pass

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global item_count, history, weibull_k, weibull_lambda
    
    # Update history
    history.append(item)
    item_count += 1
    
    # Update Weibull params periodically
    update_weibull_params()
    
    scores = np.full(len(bins), -1e18)
    
    # Calculate remaining capacity
    remaining = bins - item
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    
    # Sample a "next item" from the Weibull distribution
    # We sample multiple next items to get a robust estimate (Monte Carlo)
    n_samples = 10
    # Weibull random variates: x = lambda * (-log(u))^(1/k)
    u = np.random.uniform(0.0001, 1.0, n_samples)
    next_items = weibull_lambda * (-np.log(u)) ** (1.0 / weibull_k)
    
    # For each valid bin, estimate the cost
    # Cost metric: After placing `item` and `next_item`, what is the "waste"?
    # We want to minimize the number of bins.
    # If we place `item` in bin i, remaining is r_i = bins[i] - item.
    # If `next_item` fits in bin i, it goes there. Remaining becomes r_i - next_item.
    # If `next_item` does NOT fit in bin i, it might go to another bin or open a new one.
    # Opening a new bin is bad (increases N_bins).
    # So, we prefer bins where `next_item` is likely to fit, OR where the remaining space is so small 
    # that it's "dead" anyway.
    
    # Let's define the score as the probability that the bin remains "useful" for future items,
    # minus the waste.
    
    # Simple proxy: 
    # Score = - (remaining_after_item) 
    # But penalize if remaining_after_item is "awkward" (e.g., can't fit the expected next item).
    
    # Let's calculate the expected "waste" of the bin after placing the item.
    # If the remaining space is > expected_next_item, it's "open" and might be used.
    # If the remaining space is < expected_next_item, it's "fragmented" and likely wasted.
    
    # We want to AVOID creating fragments that are too small for typical items but too large to ignore.
    # So, if r_i is close to the median of next_items, it's a "bad" gap?
    # Actually, if r_i is small, it's good (bin is full).
    # If r_i is large, it's good (bin is ready for large items).
    # The worst case is r_i in the "middle" where it can't fit many small items but wastes space for large ones.
    
    # Let's use the distribution of next_items to score the gap r_i.
    # Score_i = -r_i + bonus_if_r_i_is_very_small_or_very_large
    
    # For each valid bin i:
    # r_i = bins[i] - item
    # Count how many of the sampled next_items fit in r_i.
    # If many fit, the bin is "active" and will be used, reducing waste.
    # If few fit, the bin is "stuck" with r_i waste.
    
    # We want to maximize the number of future items that can fit in this bin's remaining space.
    
    valid_remaining = remaining[valid_indices]
    
    # Vectorized calculation:
    # next_items shape (n_samples,)
    # valid_remaining shape (n_valid,)
    # Can next_items fit in valid_remaining?
    # fit_matrix[j, k] = 1 if next_items[k] <= valid_remaining[j]
    
    # This creates a (n_valid, n_samples) matrix.
    # n_valid is at most len(bins). len(bins) can be large (up to 1000?).
    # 1000 * 10 = 10000 comparisons. Very fast.
    
    fit_matrix = (valid_remaining[:, None] >= next_items[None, :])
    # Number of next items that fit in each bin
    num_fitting = np.sum(fit_matrix, axis=1)
    
    # Score:
    # Base: -valid_remaining (Best Fit)
    # Bonus: num_fitting (Worst Fit-like, prefer bins that can hold more future items)
    # We want to balance these.
    # If we prefer bins that hold more future items, we are doing "Worst Fit" for the next item.
    # But we also want to minimize current waste.
    
    # Let's try: Score = -valid_remaining + 0.5 * num_fitting
    # Or: Score = -valid_remaining * (1 - num_fitting/n_samples)
    # This penalizes waste less if the bin is "useful" for future items.
    
    # Let's use the probability that the bin is "alive" (can fit at least one future item)
    # P_alive = num_fitting / n_samples
    
    # If P_alive is high, the bin will be used, so the "waste" valid_remaining is not really waste, 
    # it's "capacity".
    # If P_alive is low, the valid_remaining is "dead" waste.
    
    # So, effective_waste = valid_remaining * (1 - P_alive)
    # We want to minimize effective_waste.
    
    p_alive = num_fitting / n_samples
    effective_waste = valid_remaining * (1.0 - p_alive)
    
    scores[valid_indices] = -effective_waste
    
    return scores

# EVOLVE-BLOCK-END
