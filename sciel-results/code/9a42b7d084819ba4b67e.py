"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes and weights are shape (5, 100)
    # 1. Normalize weights per constraint to account for relative tightness/scale
    #    Dividing by the mean weight of each constraint makes the constraints comparable.
    eps = 1e-9
    mean_weights = np.mean(weights, axis=1, keepdims=True) + eps
    norm_weights = weights / mean_weights
    
    # 2. Calculate efficiency (prize/weight) for each constraint
    #    eff[c, i] = prizes[c, i] / norm_weights[c, i]
    eff = prizes / (norm_weights + eps)
    
    # 3. Aggregate efficiencies
    #    Using the minimum efficiency captures the "bottleneck" constraint for each item.
    #    An item is only as good as its worst constraint efficiency.
    #    This is a strong heuristic for multi-constraint knapsack.
    scores = np.min(eff, axis=0)
    
    # Ensure positive values for ACO (though eff should be positive if prizes/weights > 0)
    # The problem states non-positive heuristic entries become 1e-9 in the evaluator,
    # but it's safer to return positive values directly.
    scores = np.maximum(scores, eps)
    
    return scores
# EVOLVE-BLOCK-END
