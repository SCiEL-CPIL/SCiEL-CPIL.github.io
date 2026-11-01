import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = 2, 3, 8
    
    rank_per_block = 11
    num_blocks = p // 2  # 4 blocks
    total_rank = rank_per_block * num_blocks  # 44
    
    U = np.zeros((n*m, total_rank))  # (6, 44)
    V = np.zeros((m*p, total_rank))  # (24, 44)
    W = np.zeros((p*n, total_rank))  # (16, 44)
    
    for blk in range(num_blocks):
        base = blk * rank_per_block
        # ... (same construction logic) ...
        
        # Add time check inside loop? Loop is very small (4 iterations).
        
    # Final time check
    if time.time() > deadline - 5:
        # If we are close to deadline, just return what we have.
        # Since it's deterministic, it finishes instantly.
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
