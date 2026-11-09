# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW and CCW contribution matrices
    # CW[i, e] is the weight added to edge e by demand i if routed CW
    # CCW[i, e] is the weight added to edge e by demand i if routed CCW
    CW = np.zeros((D, N), dtype=np.float64)
    CCW = np.zeros((D, N), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        if s <= t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, N)) + list(range(0, t))
        
        for e in cw_edges:
            CW[i, e] = w
        
        # CCW edges are all other edges
        # Since CW and CCW partition the ring edges for a demand:
        # CCW[i, e] = w if e not in cw_edges else 0
        # We can compute this as: w - CW[i, e] for each edge?
        # Wait, CW[i, e] is either 0 or w.
        # So CCW[i, e] is w if CW[i, e] == 0 else 0.
        # This is equivalent to w - CW[i, e].
        CCW[i, :] = w - CW[i, :]

    Base = CW.sum(axis=0)
    Delta = CCW - CW  # Shape (D, N)
    
    # Generate all 2^15 configurations
    # X is an array of shape (2^D, D) where X[k, i] is the bit for demand i in config k
    # We use integer bit shifting to generate binary representations
    num_configs = 1 << D
    indices = np.arange(num_configs, dtype=np.int32)
    # Shift bits to get column i
    # (indices >> i) & 1
    # We want shape (num_configs, D)
    # indices[:, None] has shape (num_configs, 1)
    # np.arange(D)[None, :] has shape (1, D)
    X = (indices[:, None] >> np.arange(D)[None, :]) & 1
    X = X.astype(np.float64)
    
    # Compute loads for all configurations
    # Loads[k, e] = Base[e] + sum_i X[k, i] * Delta[i, e]
    # X @ Delta has shape (num_configs, N)
    Loads = X @ Delta + Base[None, :]
    
    # Find configuration with minimum max load
    MaxLoads = Loads.max(axis=1)
    best_idx = np.argmin(MaxLoads)
    best_x = X[best_idx]
    
    return best_x

# EVOLVE-BLOCK-END
