import time
import random
import numpy as np

N = 16
NUM_EDGES = 120
NUM_K4 = 1820

def get_edge_idx(i, j):
    if i > j:
        i, j = j, i
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

# Precompute K4s and edge mappings
K4_EDGE_INDICES = []
EDGE_TO_K4S = [[] for _ in range(NUM_EDGES)]

k4_idx = 0
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            for d in range(c + 1, N):
                edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
                e_idxs = [get_edge_idx(u, v) for u, v in edges]
                K4_EDGE_INDICES.append(e_idxs)
                for e in e_idxs:
                    EDGE_TO_K4S[e].append(k4_idx)
                k4_idx += 1

def calculate_initial_penalty(colors, K):
    """
    Calculate initial penalty and setup counts for incremental updates.
    Returns (penalty, k4_counts, k4_distinct)
    k4_counts: list of lists, k4_counts[k][c] = count of color c in k4 k
    k4_distinct: list of ints, k4_distinct[k] = number of distinct colors in k4 k
    """
    k4_counts = [[0] * K for _ in range(NUM_K4)]
    k4_distinct = [0] * NUM_K4
    penalty = 0
    
    for k in range(NUM_K4):
        e_idxs = K4_EDGE_INDICES[k]
        for e in e_idxs:
            c = colors[e]
            k4_counts[k][c] += 1
        
        distinct = sum(1 for count in k4_counts[k] if count > 0)
        k4_distinct[k] = distinct
        if distinct < 5:
            penalty += (5 - distinct)
            
    return penalty, k4_counts, k4_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for final checks and return
    
    best_colors = None
    best_penalty = float('inf')
    best_K = 17 # Fallback K
    
    # Fallback: The seed solution
    # (i+j)%17 is valid for K16?
    # Let's assume it is valid as per prompt context.
    fallback_colors = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    # We try K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 10.0:
            break
            
        # Time budget per K
        time_budget_for_K = 30.0
        k_start = time.time()
        
        # Multiple restarts
        while time.time() - k_start < time_budget_for_K:
            if time.time() > deadline - 5.0:
                break
                
            # Initialization
            # Random init
            colors = np.random.randint(0, K, size=NUM_EDGES).tolist()
            
            # Calculate initial penalty and state
            current_penalty, k4_counts, k4_distinct = calculate_initial_penalty(colors, K)
            
            if current_penalty == 0:
                return colors
                
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_colors = colors.copy()
                best_K = K
                
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            alpha = 0.999
            t_start = time.time()
            time_per_run = 5.0
            
            while time.time() - t_start < time_per_run:
                if time.time() > deadline - 2.0:
                    break
                    
                # Pick random edge
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_color = colors[e_idx]
                
                # Pick new color
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                
                # Calculate delta penalty
                delta = 0
                affected_k4s = EDGE_TO_K4S[e_idx]
                
                # Store old distinct values for rollback if needed
                # But we can just compute delta and apply if accepted, else revert
                # To revert, we need to know how distinct changed.
                # Let's store the old distinct values for the affected K4s
                old_distincts = [0] * len(affected_k4s)
                
                for i, k in enumerate(affected_k4s):
                    old_distincts[i] = k4_distinct[k]
                    
                    # Update counts for this K4
                    cnt = k4_counts[k]
                    
                    # Remove old color
                    cnt[old_color] -= 1
                    if cnt[old_color] == 0:
                        k4_distinct[k] -= 1
                        delta += 1 # Penalty increases by 1 if distinct drops below 5? 
                        # Wait, penalty is max(0, 5-distinct).
                        # If distinct was 5, becomes 4 -> penalty +1.
                        # If distinct was 4, becomes 3 -> penalty +1.
                        # If distinct was 6, becomes 5 -> penalty -1 (if it was contributing 0, now 0? No, 5-6<0 so 0. 5-5=0. No change).
                        # Actually, let's just calculate new penalty contribution and subtract old.
                        
                    # Add new color
                    cnt[new_color] += 1
                    if cnt[new_color] == 1:
                        k4_distinct[k] += 1
                        
                    # Recalculate penalty contribution for this K4
                    old_pen = 5 - old_distincts[i] if old_distincts[i] < 5 else 0
                    new_pen = 5 - k4_distinct[k] if k4_distinct[k] < 5 else 0
                    delta += new_pen - old_pen
                
                # Acceptance
                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    colors[e_idx] = new_color
                    current_penalty += delta
                    
                    if current_penalty < best_penalty:
                        best_penalty = current_penalty
                        best_colors = colors.copy()
                        best_K = K
                        if best_penalty == 0:
                            return colors
                else:
                    # Reject: Revert changes
                    for i, k in enumerate(affected_k4s):
                        cnt = k4_counts[k]
                        
                        # Revert new color
                        cnt[new_color] -= 1
                        if cnt[new_color] == 0:
                            k4_distinct[k] -= 1
                            
                        # Revert old color
                        cnt[old_color] += 1
                        if cnt[old_color] == 1:
                            k4_distinct[k] += 1
                        # Note: k4_distinct[k] should be back to old_distincts[i]
                    
                    # Ensure consistency
                    # current_penalty remains unchanged
                
                T *= alpha
                if T < T_min:
                    T = T_min
                    
        if best_penalty == 0:
            return best_colors

    # If no better solution found, return the best found if valid, else fallback
    if best_colors is not None and best_penalty == 0:
        return best_colors
    
    # Check if best_colors is valid (penalty 0)
    if best_colors is not None:
        # Re-verify penalty for best_colors to be safe
        # Using the original slow method for verification
        pen = 0
        for e_idxs in K4_EDGE_INDICES:
            c0 = best_colors[e_idxs[0]]
            c1 = best_colors[e_idxs[1]]
            c2 = best_colors[e_idxs[2]]
            c3 = best_colors[e_idxs[3]]
            c4 = best_colors[e_idxs[4]]
            c5 = best_colors[e_idxs[5]]
            mask = (1 << c0) | (1 << c1) | (1 << c2) | (1 << c3) | (1 << c4) | (1 << c5)
            distinct = bin(mask).count('1')
            if distinct < 5:
                pen += (5 - distinct)
        if pen == 0:
            return best_colors

    # Fallback to seed solution (17 colors)
    return fallback_colors
