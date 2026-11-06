import time
import random
import sys
import numpy as np

N = 16
NUM_EDGES = 120

def get_edge_idx(i, j):
    if i > j:
        i, j = j, i
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

# Precompute edges
EDGES = []
for i in range(N):
    for j in range(i + 1, N):
        EDGES.append((i, j))

# Precompute K4s
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

# Map edge to list of K4 indices
EDGE_TO_K4 = [[] for _ in range(NUM_EDGES)]
for k4_idx, e_idxs in enumerate(K4_EDGE_INDICES):
    for e_idx in e_idxs:
        EDGE_TO_K4[e_idx].append(k4_idx)

# Helper to calculate penalty for a single K4
# Returns squared violation: (5 - distinct_count)^2 if distinct_count < 5 else 0
def calc_k4_penalty(colors, e_idxs):
    # colors is a list/array of edge colors
    # e_idxs is a list of 6 edge indices
    c_set = {colors[e] for e in e_idxs}
    distinct = len(c_set)
    if distinct < 5:
        return (5 - distinct) ** 2
    return 0

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We use 140s to be safe.
    deadline = start_time + 140.0
    
    # We want to minimize the number of colors K.
    # Try K = 13, 14, 15
    # For each K, run SA to find a valid coloring (penalty 0).
    
    for K in [13, 14, 15]:
        time_start_K = time.time()
        # Allocate time for this K. 
        # If we find a solution for 13, we return immediately.
        # If we fail for 13, we move to 14.
        # We should spend most of our time on the smallest K.
        # Let's give 40s to K=13, 40s to K=14, 40s to K=15.
        time_budget_K = 40.0
        
        # Simulated Annealing
        # Initial random coloring
        colors = np.random.randint(0, K, size=NUM_EDGES)
        
        # Calculate initial penalties for all K4s
        k4_penalties = np.zeros(NUM_K4, dtype=int)
        total_penalty = 0
        for k4_idx in range(NUM_K4):
            pen = calc_k4_penalty(colors, K4_EDGE_INDICES[k4_idx])
            k4_penalties[k4_idx] = pen
            total_penalty += pen
        
        if total_penalty == 0:
            return colors.tolist()
            
        T = 100.0
        T_min = 0.1
        cooling_rate = 0.995
        
        best_total_penalty = total_penalty
        best_colors = colors.copy()
        
        while time.time() - time_start_K < time_budget_K:
            if time.time() > deadline:
                break
                
            # Pick a random edge
            e_idx = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e_idx]
            
            # Pick a new color
            # To encourage diversity, we can pick a random color from 0..K-1
            new_color = random.randint(0, K - 1)
            if new_color == old_color:
                continue
                
            # Calculate delta penalty
            # Only K4s containing e_idx are affected
            delta = 0
            affected_k4s = EDGE_TO_K4[e_idx]
            
            # Store old penalties to restore if rejected
            old_k4_pens = []
            
            # Apply new color tentatively
            colors[e_idx] = new_color
            
            for k4_idx in affected_k4s:
                old_pen = k4_penalties[k4_idx]
                new_pen = calc_k4_penalty(colors, K4_EDGE_INDICES[k4_idx])
                old_k4_pens.append((k4_idx, old_pen))
                delta += new_pen - old_pen
                k4_penalties[k4_idx] = new_pen
                
            new_total_penalty = total_penalty + delta
            
            # Acceptance criterion
            accept = False
            if new_total_penalty < total_penalty:
                accept = True
            else:
                # Delta is positive
                if delta == 0:
                    accept = True
                else:
                    # Boltzmann factor
                    # Avoid overflow
                    if T > 0:
                        ratio = np.exp(-delta / T)
                        accept = random.random() < ratio
            
            if accept:
                total_penalty = new_total_penalty
                if total_penalty < best_total_penalty:
                    best_total_penalty = total_penalty
                    best_colors = colors.copy()
                    if total_penalty == 0:
                        # Found a valid coloring with K colors
                        return colors.tolist()
            else:
                # Reject: revert changes
                colors[e_idx] = old_color
                for k4_idx, old_pen in old_k4_pens:
                    k4_penalties[k4_idx] = old_pen
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Optional: Restart with higher T if stuck? 
                # For now, just keep going.
                
        # If we didn't find a valid coloring for this K, continue to next K
        # We could also try to use best_colors as a start for next K, but K changes, so re-init is safer.
        
    # Fallback: If we couldn't find a solution with 13, 14, or 15 colors in the time limit,
    # we must return a valid solution. The seed solution uses 17 colors.
    # It is better to return a valid 17-color solution than an invalid 13-color one.
    # The score is seed/value. If value=17, score < 1. If value=13, score > 1.
    # But invalid scores 0.
    # So we must ensure validity.
    
    # Let's quickly try K=16 or 17 with a very short SA run to ensure we have a valid fallback
    # Or just use the deterministic seed.
    # Seed: (i+j) % 17
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# To ensure we don't exceed recursion limits or other issues, we run construct directly.
# The function is defined above.

if __name__ == "__main__":
    # Test run
    res = construct()
    print(len(res))
    # Verify
    # ...
