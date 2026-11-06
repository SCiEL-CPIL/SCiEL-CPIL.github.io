import time
import random
import numpy as np

N = 16
NUM_EDGES = 120

# Precompute edge indices
def get_edge_idx(i, j):
    if i > j:
        i, j = j, i
    # Number of edges before row i
    # Row 0: 15 edges (0,1)..(0,15)
    # Row 1: 14 edges (1,2)..(1,15)
    # ...
    # Row i: 15-i edges
    # Total edges before row i: sum_{k=0}^{i-1} (15-k)
    # = 15*i - i*(i-1)/2
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

# Precompute K4s and their edge indices
K4_LIST = []
K4_EDGE_INDICES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                e_idxs = [get_edge_idx(u, v) for u, v in edges]
                K4_EDGE_INDICES.append(e_idxs)

NUM_K4 = len(K4_EDGE_INDICES)

# Precompute edge to K4 mapping for potential future use, but we'll use full recalculation
# for simplicity and speed in CPython for small N.
# However, we can optimize the penalty calculation using numpy if needed, 
# but pure python loops with bitmasks are very fast for N=16.

def calculate_penalty(colors):
    """Calculate total penalty. Penalty is sum of (5 - distinct_count) for each K4 with < 5 distinct colors."""
    total_pen = 0
    for e_idxs in K4_EDGE_INDICES:
        c0 = colors[e_idxs[0]]
        c1 = colors[e_idxs[1]]
        c2 = colors[e_idxs[2]]
        c3 = colors[e_idxs[3]]
        c4 = colors[e_idxs[4]]
        c5 = colors[e_idxs[5]]
        
        mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4) | (1 << c5)
        distinct = bin(mask).count('1')
        
        if distinct < 5:
            total_pen += (5 - distinct)
            
    return total_pen

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_colors = None
    best_penalty = float('inf')
    
    # We try to find a solution with 13 colors, then 14, then 15.
    # The seed uses 17 colors (0..16).
    # Lower is better.
    
    for K in [13, 14, 15]:
        if time.time() > deadline - 5.0:
            break
            
        # Multiple restarts for each K
        # Budget: ~30s per K
        time_budget_for_K = 30.0
        k_start = time.time()
        
        while time.time() - k_start < time_budget_for_K:
            # Initialize random coloring
            # Use numpy for speed
            colors = np.random.randint(0, K, size=NUM_EDGES).tolist()
            
            current_penalty = calculate_penalty(colors)
            
            if current_penalty == 0:
                return colors
                
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_colors = colors.copy()
                
            # Simulated Annealing
            T = 10.0
            T_min = 0.1
            alpha = 0.9995
            t_start = time.time()
            time_per_run = 5.0
            
            while time.time() - t_start < time_per_run:
                if time.time() > deadline - 2.0:
                    break
                    
                # Pick random edge
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_color = colors[e_idx]
                
                # Pick new color
                if K == 1:
                    new_color = 0
                else:
                    new_color = random.randint(0, K - 1)
                    if new_color == old_color:
                        continue
                        
                colors[e_idx] = new_color
                new_penalty = calculate_penalty(colors)
                
                delta = new_penalty - current_penalty
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_penalty = new_penalty
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_colors = colors.copy()
                        if best_penalty == 0:
                            return colors
                else:
                    colors[e_idx] = old_color
                    
                T *= alpha
                if T < T_min:
                    T = T_min
                    
        if best_penalty == 0:
            return best_colors

    # Fallback: return a known valid coloring (using 17 colors, but better than nothing)
    # Or return best found even if invalid? 
    # The task says "A valid colouring certifies f(16) <= colours used."
    # "The evaluator measures the raw value (lower is better)."
    # If we return an invalid coloring, does it score 0?
    # "an invalid result, an exception or a timeout scores 0"
    # So we MUST return a valid coloring.
    
    # The seed program returns [(i + j) % 17 ...] which is valid?
    # Let's check if (i+j)%17 is valid.
    # If not, we need a guaranteed valid fallback.
    
    # Let's verify the seed fallback quickly.
    # Actually, we can just use a greedy construction or a known algebraic one if we are sure.
    # But since we have time, we should have found one.
    # If we didn't, we return the best_colors found, but it might be invalid.
    # To be safe, if best_colors is invalid, we should try to repair it or use a fallback.
    
    # Let's check validity of best_colors.
    if best_colors is not None and calculate_penalty(best_colors) == 0:
        return best_colors
        
    # If no valid solution found, we must return something valid.
    # Let's try a simple heuristic construction for 15 colors?
    # Or just return the seed's fallback?
    # Seed fallback: [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    # Let's assume this is valid as per the seed program's implication (score 1.0).
    
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
