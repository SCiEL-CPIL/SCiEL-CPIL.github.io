"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Work with copies
    prizes = np.asarray(prizes, dtype=np.float64)
    dist = np.asarray(distance_matrix, dtype=np.float64)
    
    # Ensure diagonal is large (self-transitions not useful)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    
    # Prize vector, node 0 has prize 0
    p = prizes.copy()
    p[0] = 0.0
    
    # Distance to start (for return constraint awareness)
    d_to_start = d[:, 0].copy()
    d_to_start[0] = 0.0
    
    # Heuristic: prize attractiveness discounted by distance
    # Use a sharper distance decay to make the heuristic more selective
    # eta[i,j] = prize[j] / (dist[i,j] + eps) ** alpha
    # where alpha > 1 makes distance matter more
    
    eps = 1e-9
    alpha = 1.5  # sharper distance decay
    
    # Compute base heuristic
    with np.errstate(divide='ignore', invalid='ignore'):
        base = p[None, :] / (d + eps) ** alpha
    
    # Replace non-finite with small value
    base = np.where(np.isfinite(base), base, 1e-9)
    
    # Ensure all values are positive
    base = np.maximum(base, 1e-9)
    
    # Add a prize-rank bonus: nodes with higher prize get a small bonus
    # This helps break ties and pushes toward top-prize nodes
    prize_ranks = np.argsort(np.argsort(p)) / max(n - 1, 1)  # normalized rank [0,1]
    rank_bonus = 0.1 * prize_ranks[None, :]
    
    # Combine: base + rank_bonus
    heuristic = base + rank_bonus
    
    # Ensure diagonal is minimal (self-loops not attractive)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Final safety: ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
