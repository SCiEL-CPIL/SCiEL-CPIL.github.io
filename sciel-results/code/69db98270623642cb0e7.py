"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Combine the robust Best-Fit foundation of the current program 
# with the "bimodal residual" insight from inspiration 1.
#
# The current program (score 1.0) uses a linear Best-Fit heuristic with a 
# slight index-based tie-breaker. 
# Inspiration 1 (score 1.0204, which is worse than 1.0 in this "lower is better" 
# context? Wait, the prompt says "1.0 equals the seed program and above 1.0 is an improvement".
# Let's re-read carefully: "The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".
# Wait, if the metric is "mean percent excess" and "lower is better", then:
# If Seed Excess = E_seed.
# If Child Excess = E_child.
# Score = E_seed / E_child.
# If E_child < E_seed (better packing), Score > 1.0.
# If E_child > E_seed (worse packing), Score < 1.0.
#
# Inspiration 1 has score 1.0204. This means it is BETTER than the seed (1.0).
# Inspiration 2 has score 1.0000. Tie.
# Inspiration 3 has score 1.0000. Tie.
#
# So Inspiration 1 is the strongest inspiration.
# Inspiration 1 uses: score = -r * (C - r)
# This function is 0 at r=0 and r=C, and negative in between.
# Maximizing this score means preferring r close to 0 OR r close to C.
# r = bins - item.
# r close to 0 means the item fits perfectly (Best Fit).
# r close to C means the bin is almost empty (Worst Fit).
# This "bimodal" approach aims to either fill bins completely or keep them empty,
# avoiding the "medium" residual state which is often inefficient for future packing.
#
# The current program uses: score = -remaining - index * 1e-12
# This is pure Best-Fit with a tiny preference for earlier bins.
#
# Since Inspiration 1 scored 1.0204 (an improvement), we should adopt its core scoring mechanism.
# However, we should retain the structural robustness of the current program (explicit masking, deadline guard).
# We will also add a very small tie-breaker to ensure determinism and potentially help with fragmentation
# by preferring earlier bins if scores are identical, though the bimodal score is continuous.
#
# Let's refine the bimodal score.
# f(r) = -r * (100 - r)
# At r=0, f=0.
# At r=50, f=-2500.
# At r=99, f=-99.
#
# Is there a way to make this even better?
# The "bimodal" idea is strong for avoiding wasted space in the middle.
# Let's stick to the Inspiration 1 mechanism as it has proven efficacy (1.0204 > 1.0).
# We will integrate it into the current program's structure.

CAPACITY = 100.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    start_time = time.time()
    
    # Compute residuals
    residuals = bins - item
    
    # Valid bins are those with non-negative residuals
    valid = residuals >= 0
    
    # Initialize scores to -inf
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    if not np.any(valid):
        return scores

    # Calculate scores for valid bins using the bimodal heuristic from Inspiration 1
    # This prefers residuals near 0 (tight fit) or near Capacity (empty bin)
    r = residuals[valid]
    # score = -r * (CAPACITY - r)
    # This maximizes when r is close to 0 or 100.
    scores[valid] = -r * (CAPACITY - r)
    
    # Add a tiny tie-breaker for lower indices to ensure stability, similar to current program
    # This is negligible compared to the score magnitude but ensures deterministic behavior
    # in case of floating point ties.
    indices = np.arange(len(bins))
    scores[valid] -= indices[valid] * 1e-12
    
    # Deadline check
    if time.time() - start_time > 115:
        # Fallback to simple best fit if somehow slow (unlikely for this O(N) op)
        scores = np.where(valid, -residuals, -np.inf)
        
    return scores

# EVOLVE-BLOCK-END
