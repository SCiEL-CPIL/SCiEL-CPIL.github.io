import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring routing problem by exhaustive search over all 2^15 routing 
    combinations to find the one that minimizes the maximum edge load.
    Uses vectorized NumPy operations for speed.
    Includes an explicit deadline guard.
    """
    start_time = time.time()
    deadline = 630.0 # Safety margin from 600s limit
    
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute edge indices for CW and CCW routes for each demand
    # cw_edges[i] is a list of edge indices used if demand i goes CW
    # ccw_edges[i] is a list of edge indices used if demand i goes CCW
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw_set = set()
        current = s
        while current != t:
            cw_set.add(current)
            current = (current + 1) % n_nodes
        cw_edges.append(cw_set)
        ccw_edges.append(set(range(n_nodes)) - cw_set)
        
    # Build load difference matrix D
    # If demand i goes CCW instead of CW, the load on edge e changes by D[i, e]
    # Load_CCW - Load_CW
    # We can compute base load assuming all CW, then add D[i] for each i routed CCW.
    
    # Let M_CW be the load vector if all demands go CW.
    # Let M_CCW be the load vector if all demands go CCW.
    # Actually, it's easier to define:
    # L(mask) = sum_{i in mask} L_i_CCW + sum_{j not in mask} L_j_CW
    # L(mask) = sum_{j} L_j_CW + sum_{i in mask} (L_i_CCW - L_j_CW)
    # Let Base = sum_{j} L_j_CW
    # Let Delta_i = L_i_CCW - L_i_CW
    # L(mask) = Base + sum_{i in mask} Delta_i
    
    Base = np.zeros(n_nodes, dtype=np.float64)
    Delta = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # Add CW load to Base
        for e in cw_edges[i]:
            Base[e] += w
            
        # Calculate Delta: CCW load - CW load
        # CCW load on edge e is w if e in ccw_edges[i], else 0
        # CW load on edge e is w if e in cw_edges[i], else 0
        # So Delta[e] = w * (1 if e in ccw else 0) - w * (1 if e in cw else 0)
        # Since cw and ccw are disjoint and cover all edges:
        # If e in cw: Delta[e] = 0 - w = -w
        # If e in ccw: Delta[e] = w - 0 = w
        
        for e in cw_edges[i]:
            Delta[i, e] = -w
        for e in ccw_edges[i]:
            Delta[i, e] = w
            
    # Generate all 2^15 masks
    # masks: shape (32768,)
    masks = np.arange(1 << n_demands)
    
    # Create binary indicator matrix
    # bits: shape (32768, 15)
    # bits[k, i] is 1 if bit i of mask k is set (meaning demand i goes CCW)
    bits = (masks[:, np.newaxis] >> np.arange(n_demands)) & 1
    
    # Check deadline
    if time.time() > deadline:
        # This should never happen for this small problem size
        # Fallback to all CW (zeros)
        return np.zeros(n_demands, dtype=np.float64)
        
    # Calculate loads for all masks
    # loads = bits @ Delta + Base
    # bits: (32768, 15)
    # Delta: (15, 10)
    # loads: (32768, 10)
    loads = bits @ Delta + Base
    
    # Find max load for each mask
    max_loads = loads.max(axis=1)
    
    # Find the mask with minimum max load
    best_idx = np.argmin(max_loads)
    
    # Extract the best routing
    # bits[best_idx] is 1 for CCW, 0 for CW
    # The problem states: "An entry >= 0.5 routes the demand counter-clockwise, any other value clockwise."
    # So we return the bits directly as floats.
    best_routing = bits[best_idx].astype(np.float64)
    
    return best_routing
