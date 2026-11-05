import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110  # Leave 10s buffer for 120s limit
    
    # Precompute all 4-APs
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    A = np.array(ap_list) # Shape (M, 4)
    M = len(A)
    
    # Helper to count mono APs
    def count_mono(colors):
        cols = colors[A]
        # Check if all 4 are equal
        # cols[:, 0] == cols[:, 1] == cols[:, 2] == cols[:, 3]
        return np.sum((cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3]))
    
    def local_search(colors):
        # colors is a numpy array of 0/1
        current_score = count_mono(colors)
        
        while True:
            # Compute deltas for all bits
            deltas = np.zeros(n, dtype=np.int8)
            
            # For each column j in 0..3, compute contribution to delta for that bit
            for j in range(4):
                # Get colors for all APs
                cols = colors[A]
                # The bit at position j
                c_j = cols[:, j]
                # The other 3 bits
                others = np.delete(cols, j, axis=1)
                
                # Check if the other 3 are equal
                # others[:, 0] == others[:, 1] == others[:, 2]
                others_eq = (others[:, 0] == others[:, 1]) & (others[:, 1] == others[:, 2])
                
                # If others are equal:
                # If c_j == others[0], flipping breaks mono (delta -1)
                # If c_j != others[0], flipping creates mono (delta +1)
                # Else (others not equal), delta is 0
                
                is_mono = others_eq & (c_j == others[:, 0])
                becomes_mono = others_eq & (c_j != others[:, 0])
                
                delta_vals = -is_mono.astype(np.int8) + becomes_mono.astype(np.int8)
                
                # Add to deltas at the indices of the bits
                np.add.at(deltas, A[:, j], delta_vals)
            
            # Find best flip
            min_delta = np.min(deltas)
            
            if min_delta >= 0:
                break
            
            # Find index of min_delta
            # If multiple, pick random among them to escape symmetry
            candidates = np.where(deltas == min_delta)[0]
            k = random.choice(candidates)
            
            # Flip k
            colors[k] = 1 - colors[k]
            current_score += min_delta
            
            # Optional: early stop if 0
            if current_score == 0:
                break
        
        return colors, current_score

    best_score = float('inf')
    best_colors = None
    
    # Initializations
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def alternating(n):
        return np.array([i % 2 for i in range(n)], dtype=np.int8)
        
    def blocks(n, size=2):
        return np.array([(i // size) % 2 for i in range(n)], dtype=np.int8)

    init_strategies = [
        lambda: np.random.randint(0, 2, n, dtype=np.int8),
        lambda: thue_morse(n),
        lambda: alternating(n),
        lambda: blocks(n, 2),
        lambda: blocks(n, 4),
        lambda: blocks(n, 8)
    ]
    
    while time.time() < deadline:
        # Pick a random strategy
        init_fn = random.choice(init_strategies)
        colors = init_fn()
        
        # Run local search
        colors, score = local_search(colors)
        
        if score < best_score:
            best_score = score
            best_colors = colors.copy()
            
        # If we found 0, we can stop
        if best_score == 0:
            break

    if best_colors is None:
        return [0] * n
        
    return list(best_colors)
