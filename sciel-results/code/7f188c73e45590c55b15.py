# EVOLVE-BLOCK-START
def _get_rank_28_solution():
    # ... existing code ...
    return U, V, W

def _try_rank_27_search(U28, V28, W28, time_limit):
    # ... search logic ...
    return None, None, None

def construct():
    U28, V28, W28 = _get_rank_28_solution()
    
    # Verify
    # ... verification ...
    
    # Search
    start = time.time()
    U27, V27, W27 = _try_rank_27_search(U28, V28, W28, 5.0)
    
    if U27 is not None:
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
        
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END
