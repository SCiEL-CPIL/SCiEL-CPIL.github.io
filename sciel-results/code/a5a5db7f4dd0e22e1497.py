import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 110.0  # 10s margin
    
    best_a = np.ones(1000)
    # Try increasingly large N
    for N in [10000, 100000, 1000000, 5000000, 10000000]:
        if time.time() > deadline:
            break
        try:
            a = np.ones(N)
            if time.time() > deadline:
                break
            best_a = a
        except MemoryError:
            break
        except Exception:
            break
    return best_a
