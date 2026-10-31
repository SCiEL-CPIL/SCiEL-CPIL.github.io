"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 120.0
_start_time = time.time()

# State variables for online adaptation
# We track the running mean of item sizes to adapt our "usefulness" thresholds.
# We also track the number of items to weight the adaptation.
_item_count = 0
_item_sum = 0.0
# Initial estimate for Weibull(2, 12) mean is ~10.6
_current_mean_item = 10.6

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _item_count, _item_sum, _current_mean_item
    
    # Update running statistics
    _item_count += 1
    _item_sum += item
    # Use a smoothed update to avoid noise in early stages
    # alpha = 1 / _item_count would be exact, but we want stability.
    # Let's use a simple moving average or exponential moving average.
    # For robustness, we'll use a simple average that converges quickly.
    _current_mean_item = _item_sum / _item_count
    
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # Safety fallback: Best Fit Decreasing (static)
        remaining = bins - item
        feasible = remaining >= 0
        return np.where(feasible, -remaining, -np.inf)

    remaining = bins - item
    feasible = remaining >= 0

    # The core strategy: Dynamic Fragmentation Penalty
    # A gap is "dead space" if it's smaller than the expected next item (can't fit anything).
    # A gap is "waste" if it's very large (bin is mostly empty, could have been avoided).
    #
    # We define two thresholds based on the current mean item size (mu):
    # 1. Dead Space Threshold (T_dead): Gaps < T_dead are likely useless.
    #    Let T_dead = 0.5 * mu. If the gap is half the average item size, it's risky.
    # 2. Waste Threshold (T_waste): Gaps > T_waste are inefficient.
    #    Let T_waste = 3.0 * mu. If the gap is 3x the average item, we've left too much room.
    
    mu = _current_mean_item
    T_dead = 0.5 * mu
    T_waste = 3.0 * mu
    
    # We want to maximize the score.
    # The ideal gap is somewhere between T_dead and T_waste.
    # Let's model the "utility" of a gap as a function that is:
    # - Negative (penalty) if gap < T_dead (dead space)
    # - Positive (reward) if T_dead <= gap <= T_waste (useful space)
    # - Negative (penalty) if gap > T_waste (waste)
    
    # A simple piecewise linear or quadratic function can achieve this.
    # Let's use a "bump" function centered at the midpoint of the useful range.
    # Center = (T_dead + T_waste) / 2 = 1.75 * mu
    # Width = (T_waste - T_dead) / 2 = 1.25 * mu
    
    center = 1.75 * mu
    width = 1.25 * mu
    
    # Score is a Gaussian-like bump around the center, but with hard penalties outside.
    # Actually, a simpler approach:
    # Score = - (gap - center)^2 / width^2
    # This naturally penalizes deviations from the "sweet spot" which adapts to the data.
    
    # However, we must ensure that very small gaps (near 0) are not overly penalized if they
    # are close to 0, because a near-full bin is generally good.
    # Wait, if gap < T_dead, it's dead space. So gap=0 is dead space?
    # No, gap=0 means the bin is full. That is NOT dead space.
    # Dead space is when the gap is > 0 but too small to fit the next item.
    # So, gap=0 is ideal (perfect fit).
    # Gap in (0, T_dead) is bad (dead space).
    # Gap in [T_dead, T_waste] is good (useful).
    # Gap > T_waste is bad (waste).
    
    # Let's refine the score function:
    # 1. If gap == 0: Score = +Infinity (or very high) -> Perfect fit.
    # 2. If 0 < gap < T_dead: Score = -gap (penalize dead space, linearly)
    # 3. If T_dead <= gap <= T_waste: Score = 0 (neutral, acceptable)
    # 4. If gap > T_waste: Score = -(gap - T_waste) (penalize waste, linearly)
    
    # This is a piecewise linear penalty. Let's implement it.
    
    scores = np.full_like(remaining, -np.inf, dtype=float)
    
    # Perfect fit
    perfect_fit = np.isclose(remaining, 0.0, atol=1e-9)
    scores[perfect_fit] = 1000.0  # High reward for perfect fit
    
    # Dead space: 0 < gap < T_dead
    dead_space = (remaining > 1e-9) & (remaining < T_dead)
    scores[dead_space] = -remaining[dead_space]  # Linear penalty
    
    # Useful range: T_dead <= gap <= T_waste
    useful = (remaining >= T_dead) & (remaining <= T_waste)
    scores[useful] = 0.0  # Neutral
    
    # Waste: gap > T_waste
    waste = remaining > T_waste
    scores[waste] = -(remaining[waste] - T_waste)  # Linear penalty
    
    # Apply feasibility mask
    scores = np.where(feasible, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
