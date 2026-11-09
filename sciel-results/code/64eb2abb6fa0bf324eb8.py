import numpy as np
import time
from scipy.optimize import milp, LinearConstraint, Bounds

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by minimizing the maximum edge load.
    Uses a vectorized exhaustive search over all 2^15 routing combinations.
    """
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    # Pre-compute load contributions
    load_contrib = np.zeros((n_demands, 2, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        cw_edges = []
        current = s
        while current != t:
            cw_edges.append(current)
            current = (current + 1) % n_nodes
        
        load_contrib[i, 0, cw_edges] = w
        
        all_edges = np.arange(n_nodes)
        ccw_edges = np.array([e for e in all_edges if e not in cw_edges])
        load_contrib[i, 1, ccw_edges] = w

    base_load = np.sum(load_contrib[:, 0, :], axis=0)
    delta = load_contrib[:, 1, :] - load_contrib[:, 0, :]  # shape (15, 10)
    
    # Vectorized evaluation of all masks
    masks = np.arange(1 << n_demands, dtype=np.uint32)
    bits = (masks[:, None] >> np.arange(n_demands)[None, :]) & 1
    loads = base_load[None, :] + (bits @ delta)
    
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    best_max_load = max_loads[best_idx]
    
    # Construct result array
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0

    # Try to improve via local search / hill climbing
    if time.time() < deadline - 30:
        # Local search: try flipping individual bits and pairs
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            current_mask = best_mask
            current_max = best_max_load
            
            # Try flipping single bits
            for i in range(n_demands):
                new_mask = current_mask ^ (1 << i)
                if new_mask == current_mask:
                    continue
                # Compute new loads
                new_loads = base_load + np.sum(bits[new_mask][None, :] * delta, axis=0) if False else base_load + (bits[new_mask] @ delta)
                new_max = np.max(new_loads)
                if new_max < current_max - 1e-9:
                    current_mask = new_mask
                    current_max = new_max
                    improved = True
            
            # Try flipping pairs of bits
            if not improved:
                for i in range(n_demands):
                    for j in range(i + 1, n_demands):
                        new_mask = current_mask ^ (1 << i) ^ (1 << j)
                        if new_mask == current_mask:
                            continue
                        new_loads = base_load + (bits[new_mask] @ delta)
                        new_max = np.max(new_loads)
                        if new_max < current_max - 1e-9:
                            current_mask = new_mask
                            current_max = new_max
                            improved = True
            
            if improved:
                best_mask = current_mask
                best_max_load = current_max
                for i in range(n_demands):
                    if (best_mask >> i) & 1:
                        result[i] = 1.0
                    else:
                        result[i] = 0.0

    return result
