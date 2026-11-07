"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    # Convert to float array for numeric safety
    d = np.asarray(demands, dtype=np.float64)
    n = len(d)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    # Pairwise sums
    pair = d[:, None] + d[None, :]
    cap = float(capacity)

    # Base structural heuristic: prefer pairs that are tight but fit.
    # gap = cap - pair; if pair <= cap, gap >= 0.
    # We want high attractiveness for tight fits (small gap) that still fit.
    # Use a smooth penalty: 1/(1+gap) gives high values for small gaps.
    # For pairs that don't fit, set to 1e-9 (will be clamped later per contract).
    gap = cap - pair
    # Where pair <= cap, use a smooth function of gap; else 1e-9
    # Use 1/(1+gap) normalized: max when gap=0 -> 1, decreases as gap grows.
    # To make it more discriminative, use 1/(1+gap**2) or similar.
    # Let's use a combination: tightness bonus + size complementarity.
    
    # Tightness: how well the pair fills the bin
    # gap in [0, cap] for fitting pairs
    tight = np.where(pair <= cap, 1.0 / (1.0 + gap), 1e-9)
    
    # Size complementarity: prefer pairing large with small (complementary)
    # This is captured by the gap being small.
    # Additional signal: prefer pairs where both items are substantial (not tiny)
    # to reduce fragmentation. Use min(d_i, d_j) as a size weight.
    min_size = np.minimum(d[:, None], d[None, :])
    # Normalize min_size to [0, 1] relative to capacity
    size_weight = min_size / cap  # in (0, 1)
    
    # Combine: tightness * (0.5 + 0.5 * size_weight)
    # This boosts tight pairs of substantial items over tight pairs of tiny items.
    heuristic = tight * (0.5 + 0.5 * size_weight)
    
    # Ensure diagonal (i==j) is not artificially boosted; item can't be placed after itself
    # Actually in ACO, an item is placed once, so i==j won't occur in practice,
    # but let's set diagonal to 1e-9 to be safe.
    np.fill_diagonal(heuristic, 1e-9)
    
    # Clamp: any non-positive becomes 1e-9 (per contract)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
