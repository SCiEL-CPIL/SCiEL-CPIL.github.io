"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = demands.shape[0]
    
    # Compatibility matrix: fits[i, j] is True if item j can follow item i
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Waste matrix: waste[i, j] is the remaining space after placing i then j
    waste = capacity - pair_sum
    
    # Calculate scarcity for each item j:
    # Number of items that can be placed in the same bin as j (regardless of order, or just predecessors?)
    # The directive says "scarcity of its valid partners".
    # Let's count how many items k satisfy demands[j] + demands[k] <= capacity.
    compat_count = (demands[None, :] + demands[:, None] <= capacity).sum(axis=1)
    
    # Avoid division by zero, though with capacity 150 and min demand 20, 
    # every item fits with at least itself? No, 20+20=40 <= 150.
    # Max demand 99. 99+99=198 > 150. So large items might not fit with each other.
    # But they fit with small items.
    # compat_count[j] is the number of items that can co-exist with j.
    scarcity = 1.0 / np.maximum(compat_count, 1)
    
    # Heuristic: 
    # 1. Must fit.
    # 2. High fill (low waste) is better.
    # 3. High scarcity of j is better (place scarce items earlier?).
    
    # Base heuristic: inverse waste
    # If waste is 0, it's perfect. 
    # Use 1 / (waste + 1) to keep it bounded and positive.
    fill_score = 1.0 / (waste + 1.0)
    
    # Combine: fill_score[i, j] * scarcity[j]
    # scarcity[j] is a column vector (n, 1) to broadcast over rows i.
    heuristic = fill_score * scarcity[None, :]
    
    # Apply mask: if not fits, set to 1e-9
    heuristic = np.where(fits, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
