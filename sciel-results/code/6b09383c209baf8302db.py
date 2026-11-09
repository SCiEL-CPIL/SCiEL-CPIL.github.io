"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for safety
_START_TIME = time.time()
_DEADLINE_SECONDS = 120.0
_SAFETY_MARGIN = 2.0

# UCB Parameters
# We treat each call to select_next_item as a "step" in a sequential decision process.
# However, since the function is stateless (no internal state persistence between calls 
# unless we use globals), we must infer the "step" from the remaining items count.
# As items are removed, n decreases. We can use (100 - n) as a proxy for the step index.
# Or simply use a fixed exploration parameter tuned for 100 items.
# Let's use a standard UCB1 form: Q(s,a) + c * sqrt(ln(t) / N(s,a))
# Since we don't have N(s,a) (times arm pulled), we approximate N(s,a) by assuming 
# all arms are "new" but weighted by their inherent quality.
# Actually, a simpler "UCB-like" heuristic for stateless greedy is:
# Score = ValueDensity + ExplorationBonus
# ExplorationBonus depends on how "rare" or "synergistic" the item is.
# Let's define Synergy = (RemainingCapacity - Weight) / RemainingCapacity.
# High synergy means it leaves lots of room for others.
# We combine Density and Synergy.
# To add "UCB" flavor, we can add a noise term or a bonus for items that are 
# "surprising" (e.g., high value but slightly lower density than top, or low weight).
# Let's stick to a deterministic but sophisticated scoring function that mimics UCB behavior:
# Score = Density + Alpha * (1 - Weight/Capacity) + Beta * (Value / MaxValue)
# This balances density (exploitation) with capacity utilization (exploration of combos).

def _elapsed_ok():
    return (time.time() - _START_TIME) < (_DEADLINE_SECONDS - _SAFETY_MARGIN)

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1

    # Convert to numpy arrays if they aren't already (defensive, though usually they are)
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    cap = float(remaining_capacity)

    # Filter out items that don't fit (though contract says inputs are feasible, 
    # it's safer to check in case of floating point drift or interface misuse)
    feasible_mask = weights <= cap
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    f_weights = weights[feasible_indices]
    f_values = values[feasible_indices]
    
    if len(f_weights) == 0:
        return -1

    # Calculate Value Density
    densities = f_values / f_weights
    
    # Calculate "Synergy" or "Capacity Left" potential
    # Items that are small leave more capacity for high-value items later.
    # Normalize weight relative to capacity to get a 0..1 range.
    # Lower weight -> Higher synergy score.
    synergy = 1.0 - (f_weights / cap)
    
    # Normalize Value to 0..1 to compare with density? 
    # Density is value/weight. Value is absolute.
    # Let's create a composite score.
    # UCB-like: Exploit (Density) + Explore (Synergy)
    # We tune Alpha to balance these. 
    # Typically, density is the dominant factor in knapsack.
    # Let's try Alpha = 0.1 to 0.5. 
    # Let's use a dynamic Alpha based on how many items are left.
    # Early on (many items), we might want to explore more (higher Alpha).
    # Late on (few items), we exploit more (lower Alpha).
    # Step proxy: 100 - n
    step = 100 - n
    # Alpha decays as we go deeper into the search
    alpha = 0.5 * np.exp(-step / 50.0)
    
    # Composite Score
    scores = densities + alpha * synergy
    
    # To mimic UCB exploration of "unknowns", we can add a small bonus to items 
    # that are not the absolute top density but have very high synergy.
    # However, the composite score above already does this.
    
    # One more refinement: "Weight Synergy"
    # Sometimes picking a slightly lower density item that fits perfectly with 
    # others is better. The synergy term covers this.
    
    # Select the item with the highest score
    best_local_idx = np.argmax(scores)
    
    # Map back to original index
    best_global_idx = feasible_indices[best_local_idx]
    
    return int(best_global_idx)

# EVOLVE-BLOCK-END
