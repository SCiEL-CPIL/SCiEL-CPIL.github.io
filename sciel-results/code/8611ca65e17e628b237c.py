import time
import random
import numpy as np

N = 18
MIN_DISTANCE = 6
DEADLINE = 85  # seconds, safety margin before 90s kill

# Precompute popcount for all 2^18 values
# This is a one-time cost of ~256k operations, very fast.
_POPCOUNT_TABLE = np.array([i.bit_count() for i in range(2**N)], dtype=np.uint8)

def _greedy_numpy(order, deadline):
    """
    Greedy construction using NumPy vectorized distance checks.
    order: list of candidates to try.
    """
    selected = []
    selected_np = np.array([], dtype=np.int32)
    
    start_time = time.time()
    
    for w in order:
        if time.time() > deadline:
            break
            
        # Check distance to all selected codewords
        if len(selected) == 0:
            selected.append(w)
            selected_np = np.array([w], dtype=np.int32)
            continue
        
        diffs = selected_np ^ w
        dists = _POPCOUNT_TABLE[diffs]
        
        if np.min(dists) >= MIN_DISTANCE:
            selected.append(w)
            # Append to numpy array
            # Resizing numpy arrays is O(n), but n is small (<1000 usually)
            selected_np = np.append(selected_np, w)
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_code = []
    best_len = 0
    
    # Generate orders
    # 1. Natural order (0, 1, ..., 2^N - 1)
    order_natural = list(range(2**N))
    
    # 2. Reverse order
    order_reverse = list(range(2**N - 1, -1, -1))
    
    # 3. Random orders
    random.seed(42)
    order_random1 = order_natural.copy()
    random.shuffle(order_random1)
    
    random.seed(123)
    order_random2 = order_natural.copy()
    random.shuffle(order_random2)
    
    strategies = [
        ("natural", order_natural),
        ("reverse", order_reverse),
        ("random1", order_random1),
        ("random2", order_random2),
    ]
    
    for name, order in strategies:
        if time.time() > deadline - 5:
            break
            
        # Allocate time for each strategy
        # If we have 80s, and 4 strategies, give each 15s?
        # Actually, the first one might finish quickly.
        # Let's just run them sequentially until deadline.
        # But we want to ensure we finish at least one.
        
        # Use a tight deadline for each to ensure we try multiple?
        # Or just let the first one run?
        # The NumPy greedy is very fast. It should finish in < 1s.
        # So we can run many.
        
        code = _greedy_numpy(order, deadline)
        
        if len(code) > best_len:
            best_len = len(code)
            best_code = code
            
        # If we found a very large code, we might stop?
        # No, try to beat it.
        
    return best_code
