import numpy as np
import time
from scipy import optimize

def ring_routing() -> np.ndarray:
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Pre-compute edge usage for CW and CCW
    # A[i, e] = weight added to edge e if demand i is routed CW
    # B[i, e] = weight added to edge e if demand i is routed CCW
    A = np.zeros((n_demands, n_nodes))
    B = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            A[i, current] = w
            current = (current + 1) % n_nodes
        
        for e in range(n_nodes):
            if A[i, e] == 0:
                B[i, e] = w
    
    # Try MILP first for exact optimal solution
    try:
        c = np.zeros(n_demands + 1)
        c[n_demands] = 1.0
        
        A_ub = np.zeros((n_nodes, n_demands + 1))
        b_ub = np.zeros(n_nodes)
        
        for e in range(n_nodes):
            for i in range(n_demands):
                A_ub[e, i] = B[i, e] - A[i, e]
            A_ub[e, n_demands] = -1.0
            b_ub[e] = np.sum(A[:, e])
        
        bounds = [(0, 1)] * n_demands + [(0, None)]
        integrality = np.zeros(n_demands + 1)
        integrality[:n_demands] = 1
        
        res = optimize.milp(
            c=c,
            constraints=optimize.LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:
            x_sol = res.x[:n_demands]
            result = np.where(x_sol >= 0.5, 1.0, 0.0)
            return result
    
    except Exception:
        pass
    
    # Fallback: Brute force with Gray code for O(1) update per step
    base_load = np.sum(A, axis=0)
    delta = B - A  # shape (n_demands, n_nodes)
    
    best_max_load = float('inf')
    best_mask = 0
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    n_total = 1 << n_demands
    loads = base_load.copy()
    prev_gray = 0
    
    for idx in range(n_total):
        if time.time() > deadline:
            break
        
        gray = idx ^ (idx >> 1)
        
        if idx > 0:
            diff = gray ^ prev_gray
            bit_pos = diff.bit_length() - 1
            if gray & diff:
                loads += delta[bit_pos]
            else:
                loads -= delta[bit_pos]
        
        prev_gray = gray
        
        current_max = loads.max()
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = gray
    
    # Convert gray code to binary
    gray_val = best_mask
    binary_mask = gray_val
    g = gray_val
    g >>= 1
    while g:
        binary_mask ^= g
        g >>= 1
    
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (binary_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
    
    return result
