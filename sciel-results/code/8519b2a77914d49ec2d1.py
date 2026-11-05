import numpy as np
import time

_DEADLINE = time.time() + 110

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    total_rank = 49 + 44
    
    U_np = np.zeros((n * m, total_rank), dtype=float)
    V_np = np.zeros((m * p, total_rank), dtype=float)
    W_np = np.zeros((p * n, total_rank), dtype=float)
    
    current_rank = 0
    
    # ... (fill U_np, V_np, W_np using indices) ...
    
    # Trim
    U_np = U_np[:, :current_rank]
    V_np = V_np[:, :current_rank]
    W_np = W_np[:, :current_rank]
    
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
