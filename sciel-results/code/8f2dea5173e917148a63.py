import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Define demands
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute CW and CCW edge usage matrices
    # cw_usage[i, e] is 1 if demand i uses edge e when routed CW
    cw_usage = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # CW from s to t uses edges s, s+1, ..., t-1 (mod 10)
        if s <= t:
            edges_cw = range(s, t)
        else:
            edges_cw = list(range(s, n_nodes)) + list(range(0, t))
            
        for e in edges_cw:
            cw_usage[i, e] = 1.0
            
        # CCW uses all other edges
        # Since the ring is closed, the set of edges for CCW is the complement of CW edges
        # Note: The problem states "counter-clockwise uses all other ring edges".
        # This implies that if CW uses a set of edges E_cw, CCW uses E_ccw = E_all \ E_cw.
        # This is correct for a simple cycle path.
        ccw_usage_i = 1.0 - cw_usage[i, :]
        # We don't need to store ccw_usage separately if we compute loads directly,
        # but let's keep it for clarity in load calculation.
        
    # Weights
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    # Calculate base load (all CW) and delta load (switching to CCW)
    # Load_e = sum_i w_i * (1-x_i)*cw_usage[i,e] + x_i*ccw_usage[i,e]
    # Load_e = sum_i w_i * cw_usage[i,e] + x_i * w_i * (ccw_usage[i,e] - cw_usage[i,e])
    # Let A_cw[i,e] = w_i * cw_usage[i,e]
    # Let A_ccw[i,e] = w_i * ccw_usage[i,e]
    # Load_e(x) = sum_i (1-x_i) A_cw[i,e] + x_i A_ccw[i,e]
    
    A_cw = weights[:, np.newaxis] * cw_usage
    A_ccw = weights[:, np.newaxis] * (1.0 - cw_usage)
    
    # We want to find x in {0,1}^15 minimizing max_e Load_e(x)
    # Total combinations: 2^15 = 32768
    # We can vectorize this using bit operations.
    
    # Generate all bitmasks from 0 to 2^15 - 1
    max_mask = 1 << n_demands
    masks = np.arange(max_mask, dtype=np.int32)
    
    # Create a matrix of shape (2^15, 15) where entry (k, i) is 1 if bit i is set in mask k
    # This represents x_i for each configuration k
    # We can do this by shifting and masking
    # bits = (masks[:, np.newaxis] >> np.arange(n_demands)) & 1
    # This results in a (32768, 15) array of 0s and 1s.
    
    # To save memory and time, we can compute loads in chunks or directly if memory allows.
    # 32768 * 15 * 8 bytes = ~4 MB. This is very small.
    # 32768 * 10 * 8 bytes for loads = ~2.6 MB.
    
    # However, matrix multiplication (32768 x 15) @ (15 x 10) is efficient.
    
    # x_matrix: (32768, 15)
    # x_matrix[k, i] = 1 if demand i is routed CCW (bit i set), else 0 (CW)
    
    # Let's construct x_matrix efficiently
    # shifts = np.arange(n_demands)
    # x_matrix = (masks[:, np.newaxis] >> shifts) & 1
    
    # Load calculation:
    # Load = (1-x) @ A_cw.T + x @ A_ccw.T
    # Load = A_cw.T - x @ A_cw.T + x @ A_ccw.T
    # Load = A_cw.T + x @ (A_ccw.T - A_cw.T)
    
    # Let Delta = A_ccw.T - A_cw.T  (shape 10 x 15)
    # Load_e(k) = base_load[e] + sum_i x[k,i] * Delta[e,i]
    
    base_load = A_cw.sum(axis=0) # shape (10,)
    Delta = A_ccw.T - A_cw.T     # shape (10, 15)
    
    # We need x_matrix to compute Load = base_load + x_matrix @ Delta.T
    # To avoid creating the huge (32768, 15) intermediate if we can help it, 
    # but it's small enough.
    
    # Optimization: We can iterate through chunks of masks to be safe on memory, 
    # but 32768 is small. Let's do it all at once.
    
    # Create x_matrix
    shifts = np.arange(n_demands, dtype=np.int32)
    x_matrix = (masks[:, np.newaxis] >> shifts[np.newaxis, :]) & 1
    
    # Calculate loads for all configurations
    # loads shape: (32768, 10)
    # loads = x_matrix @ Delta.T + base_load[np.newaxis, :]
    
    loads = x_matrix @ Delta.T + base_load[np.newaxis, :]
    
    # Find the max load for each configuration
    max_loads = loads.max(axis=1)
    
    # Find the configuration with the minimum max load
    best_idx = np.argmin(max_loads)
    
    # Extract the routing decision for the best configuration
    best_mask = masks[best_idx]
    best_x = ((best_mask >> shifts) & 1).astype(np.float64)
    
    # The problem asks for an entry >= 0.5 to route CCW.
    # Our x is 1 for CCW, 0 for CW.
    # So returning best_x directly works.
    
    return best_x

# Verification of the logic:
# If bit i is 1, x_i = 1, which means CCW.
# The prompt says: "An entry >= 0.5 routes the demand counter-clockwise".
# So returning 1.0 for CCW and 0.0 for CW is correct.

if __name__ == "__main__":
    start = time.time()
    res = ring_routing()
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")
    print(f"Result: {res}")
    # Verify loads for the best solution
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    n_nodes = 10
    n_demands = len(demands)
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    cw_usage = np.zeros((n_demands, n_nodes), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            edges_cw = range(s, t)
        else:
            edges_cw = list(range(s, n_nodes)) + list(range(0, t))
        for e in edges_cw:
            cw_usage[i, e] = 1.0
    ccw_usage = 1.0 - cw_usage
    
    # Recalculate loads for the specific solution
    loads = np.zeros(n_nodes)
    for i in range(n_demands):
        if res[i] >= 0.5:
            loads += weights[i] * ccw_usage[i, :]
        else:
            loads += weights[i] * cw_usage[i, :]
    print(f"Max Load: {loads.max()}")
    print(f"Score: {1.0 / loads.max()}")
