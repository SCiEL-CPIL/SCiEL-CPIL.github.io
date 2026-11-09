import numpy as np
import time

# State for the UCB bandit strategy
# We categorize bins into 'fit zones' based on how much space is left after placement.
# Zones:
# 0: Very Tight (remaining < 5% of capacity)
# 1: Tight (5-15%)
# 2: Medium (15-30%)
# 3: Loose (30-60%)
# 4: Very Loose (>60%)
#
# We track:
# - counts[z]: number of times zone z was selected
# - rewards[z]: sum of "reward" received. 
#   Reward definition: We want to minimize total bins. 
#   A good move is one that keeps the bin "useful" without wasting too much space.
#   However, since we don't know the future, we use a heuristic reward:
#   If we pick a bin, the "cost" is the new remaining space. 
#   But to learn *which* fit is best, we can use the fact that 
#   Best-Fit Decreasing (BFD) is a strong baseline. 
#   We will use a simpler reward: 
#   Reward = 1.0 if the fit was "tight" (good for consolidation), 
#   but we actually want to maximize the UCB of the *choice*.
#   
# Actually, a simpler and more robust "learning" approach for bin packing 
# in online settings is to track the *average remaining space* for each fit zone 
# and prefer zones that historically led to lower remaining space (tighter packing).
#
# Let's use a Contextual Bandit where the context is the item size ratio.
# But to keep it simple and fast, we'll use a fixed set of "fit strategies" 
# and learn which one performs best in terms of "efficiency".
#
# Strategy: 
# For each incoming item, we look at all feasible bins. 
# We classify each feasible bin into one of 5 fit zones.
# We maintain an UCB score for each zone.
# We pick the bin from the zone with the highest UCB score. 
# If multiple bins are in that zone, we pick the tightest one (Best Fit).
#
# Reward signal: 
# Since we can't get a global reward immediately, we use a local proxy.
# A "good" placement is one that minimizes the remaining space in the bin.
# However, if we always pick the tightest, we just do Best Fit. 
# The "learning" aspect needs to allow deviating from Best Fit if Best Fit 
# leads to fragmentation.
#
# Alternative Structural Change: 
# Instead of UCB, let's try a "Lookahead" heuristic that is cheap.
# But the directive says "different design family".
#
# Let's stick to the UCB idea but refine the reward.
# We can simulate a "virtual" reward based on the density.
# If we place an item in a bin, the new fill ratio is (bins[i] + item) / 100.
# We want to maximize the *minimum* fill ratio of the used bins? No.
# We want to minimize the number of bins.
#
# Let's try a different family: **Entropy-based packing**.
# We aim to keep the distribution of bin fill levels "compact" (low variance) 
# or "skewed" (high variance, full bins and empty bins).
# For bin packing, "full bins and empty bins" (high variance) is better 
# because it means we have fewer partially filled bins.
#
# So, the goal is to maximize the variance of the bin fill levels 
# (among non-empty bins) or minimize the number of bins with intermediate fill.
#
# Heuristic: 
# Score(bin) = - (Remaining Space) + Lambda * (Current Fill)^2
# This is similar to the current program (quadratic term).
#
# Let's go back to the UCB/Bandit approach but make it distinct.
# The current program is a *static* greedy heuristic.
# The new program will be an *adaptive* heuristic that adjusts its 
# preference for tightness based on the current state of the bins.
#
# State: 
# - `tightness_preference`: A scalar that shifts over time.
#   Initially 0.5 (neutral).
#   If we detect many "loose" bins (fragmentation), we increase preference for tight fits.
#   If we detect many "tight" bins but no space for medium items, we might decrease it?
#   This is hard to detect online without memory.
#
# Let's use a **Fuzzy Logic** approach.
# Define linguistic variables for "Fit Quality" and "Bin State".
#
# Actually, the most distinct and potentially effective structural change 
# for Weibull distributions (which have a heavy tail of small items) is 
# **Size-Sorted Best Fit** emulation via adaptive thresholds.
#
# But we can't sort.
#
# Let's implement a **Regret-Minimizing** strategy using a simple 
# **Multiplicative Weights Update** (MWU) algorithm.
# We have 5 "arms" corresponding to fit zones.
# We maintain weights w[0..4].
# At each step, we select the zone with the highest weight.
# Among bins in that zone, pick the tightest.
# After selection, we update weights based on a "loss" signal.
# Loss = 1 if the resulting remaining space is "bad" (e.g., > 20% and < 80%), 
#        0 otherwise.
# We want to avoid "wasted" space that is too small for most future items 
# (Weibull small items) but too large to be ignored.
#
# This is a distinct optimization method (Online Convex Optimization) 
# vs. the current static heuristic.

import numpy as np
import time

# MWU State
# Zones: 0: <5%, 1: 5-15%, 2: 15-30%, 3: 30-60%, 4: >60%
# We want to prefer zones that don't create "dead" space.
# For Weibull, small items are frequent. 
# "Dead" space is space that is too small for the next few items.
# Since we don't know the next items, we assume a typical item size.
#
# Let's use a simpler, more robust structural change:
# **Simulated Annealing-like exploration**.
# With a probability that decreases over time, we pick a *random* feasible bin 
# (or a bin from a random fit zone) to explore.
# Otherwise, we use a sophisticated greedy score.
# This allows the algorithm to escape local optima of the greedy heuristic.
#
# However, "random" might be bad.
#
# Let's try **Density-Based Clustering** of the bins.
# We group bins by their current fill level.
# We prefer to place items in the cluster that, on average, 
# leads to the best "consolidation".
#
# Actually, let's look at the "Inspiration 1" which scored 1.0204.
# It just used the seed program.
#
# Let's try a **Dynamic Best Fit** where the "best" is defined by 
# minimizing the *derivative* of the cost function.
#
# Let's implement the **Multiplicative Weights Update (MWU)** for fit zones.
# It is a standard online learning algorithm distinct from static heuristics.

_CAPACITY = 100.0
# Zone boundaries (fraction of capacity remaining)
# Zone 0: remaining < 5
# Zone 1: 5 <= remaining < 15
# Zone 2: 15 <= remaining < 30
# Zone 3: 30 <= remaining < 60
# Zone 4: remaining >= 60

# MWU Weights for each zone
# Start with uniform weights
_w = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
# Learning rate
_eta = 0.1
# Start time for deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0

# Counter to track steps for potential annealing of eta
_step_count = 0

def _get_zone(remaining):
    """Assign a bin to a fit zone based on remaining space."""
    if remaining < 5.0:
        return 0
    elif remaining < 15.0:
        return 1
    elif remaining < 30.0:
        return 2
    elif remaining < 60.0:
        return 3
    else:
        return 4

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _w, _eta, _step_count, _START_TIME, _DEADLINE

    # Deadline guard
    if time.time() > _DEADLINE:
        bins = np.asarray(bins, dtype=float)
        remaining = bins - item
        feasible = remaining >= 0.0
        # Fallback to simple Best Fit
        score = -remaining
        score = np.where(feasible, score, -1e9)
        return score

    bins = np.asarray(bins, dtype=float)
    remaining = bins - item
    feasible = remaining >= 0.0

    if not np.any(feasible):
        # No feasible bin, return infeasible scores
        return np.full_like(bins, -1e9, dtype=float)

    # Calculate zones for all feasible bins
    zones = np.full_like(bins, 5, dtype=int) # 5 is infeasible/invalid
    feasible_remaining = remaining[feasible]
    feasible_bins_idx = np.where(feasible)[0]
    
    # Vectorized zone assignment
    z = np.zeros_like(feasible_remaining, dtype=int)
    z[feasible_remaining < 5.0] = 0
    z[(feasible_remaining >= 5.0) & (feasible_remaining < 15.0)] = 1
    z[(feasible_remaining >= 15.0) & (feasible_remaining < 30.0)] = 2
    z[(feasible_remaining >= 30.0) & (feasible_remaining < 60.0)] = 3
    z[feasible_remaining >= 60.0] = 4
    zones[feasible_bins_idx] = z

    # MWU: Select the zone with the highest weight
    # To allow for some exploration and avoid getting stuck, 
    # we can use a softmax over weights, but for simplicity and speed,
    # we pick the argmax weight. If there are ties, we break by tightness.
    # However, pure argmax is greedy. MWU usually involves randomization.
    # Let's use a deterministic "Softmax" selection for stability in this context,
    # or just pick the best zone and then the best bin in that zone.
    
    # To make it a true "learning" algorithm, we should update weights.
    # But we need a reward. 
    # Let's use a heuristic reward: 
    # The "loss" for a zone is the average remaining space in that zone 
    # *after* the placement, if we were to pick the tightest bin in that zone.
    # We want to minimize this loss.
    
    # Calculate the "best" remaining space for each zone among feasible bins
    # This simulates what would happen if we picked the tightest bin in that zone.
    min_rem_per_zone = np.full(5, np.inf)
    for i in range(5):
        mask = z == i
        if np.any(mask):
            min_rem_per_zone[i] = np.min(feasible_remaining[mask])
    
    # Loss for each zone: 
    # We want to avoid "dead" space. 
    # A simple loss function: 
    # If remaining is very small (<5), loss is low (good, almost full).
    # If remaining is medium (15-30), loss is high (bad, wasted).
    # If remaining is large (>60), loss is medium (ok, might fit a big item, but Weibull has few bigs).
    
    # Let's define loss as: 
    # L(z) = 1.0 if 10 < min_rem < 40 else 0.0
    # This penalizes zones that leave "awkward" amounts of space.
    
    losses = np.zeros(5)
    for i in range(5):
        if not np.isinf(min_rem_per_zone[i]):
            r = min_rem_per_zone[i]
            if 10.0 < r < 40.0:
                losses[i] = 1.0
            else:
                losses[i] = 0.0
        else:
            # If no bin in this zone, high loss (can't pick it)
            losses[i] = 10.0

    # Update MWU weights
    # w_i <- w_i * exp(-eta * loss_i)
    # Normalize
    _w = _w * np.exp(-_eta * losses)
    _w = _w / np.sum(_w)

    # Select the zone with the highest weight
    best_zone = np.argmax(_w)
    
    # Among feasible bins in the best zone, pick the tightest one (Best Fit within zone)
    # If no bin in best zone (shouldn't happen if loss was 10 for empty, but check), 
    # fall back to global Best Fit.
    
    score = np.full_like(bins, -1e9, dtype=float)
    
    # Mask bins in the best zone
    in_best_zone = (zones == best_zone) & feasible
    if np.any(in_best_zone):
        # For bins in the best zone, score them by tightness (negative remaining)
        # This ensures we pick the tightest bin in the preferred zone.
        score[in_best_zone] = -remaining[in_best_zone]
        
        # Add a small bonus to all bins in the best zone to ensure they are preferred
        # over any bin in other zones (which are -1e9)
        # The -remaining is already a good score.
    else:
        # Fallback: Best Fit globally
        score[feasible] = -remaining[feasible]

    _step_count += 1
    
    # Optional: Decay eta or adjust strategy over time
    # For now, keep it simple.
    
    return score
