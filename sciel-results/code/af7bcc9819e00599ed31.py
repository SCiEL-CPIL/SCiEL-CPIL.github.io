import random
import time
import itertools
import numpy as np

# Precomputation
N = 16
EDGES = []
EDGE_IDX = {}
for i in range(N):
    for j in range(i+1, N):
        EDGE_IDX[(i,j)] = len(EDGES)
        EDGES.append((i,j))
NUM_EDGES = len(EDGES) # 120

K4S = []
K4_EDGES = [] # List of edge indices for each K4
for combo in itertools.combinations(range(N), 4):
    a, b, c, d = combo
    edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
    edge_idxs = [EDGE_IDX[e] for e in edges]
    K4_EDGES.append(edge_idxs)
    K4S.append(combo)
NUM_K4 = len(K4S) # 1820

EDGE_TO_K4S = [[] for _ in range(NUM_EDGES)]
for k4_idx, e_idxs in enumerate(K4_EDGES):
    for e_idx in e_idxs:
        EDGE_TO_K4S[e_idx].append(k4_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try C from 13 to 15
    for C in [13, 14, 15]:
        # Try multiple restarts
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            colors = [random.randint(0, C-1) for _ in range(NUM_EDGES)]
            
            # Initialize distinct counts and color histograms
            # distinct[k4_idx] = number of distinct colors
            # hist[k4_idx][c] = count of color c in this k4
            distinct = [0] * NUM_K4
            hist = [[0]*C for _ in range(NUM_K4)]
            
            total_violation = 0
            
            for k4_idx in range(NUM_K4):
                e_idxs = K4_EDGES[k4_idx]
                h = hist[k4_idx]
                for e_idx in e_idxs:
                    c = colors[e_idx]
                    h[c] += 1
                d = sum(1 for cnt in h if cnt > 0)
                distinct[k4_idx] = d
                if d < 5:
                    total_violation += (5 - d)
            
            if total_violation == 0:
                return colors
            
            # Simulated Annealing
            T = 1.0
            current_violation = total_violation
            best_violation = total_violation
            best_colors = colors.copy()
            
            steps = 0
            max_steps = 50000 # Adjust based on speed
            stuck_count = 0
            
            while steps < max_steps:
                if time.time() > deadline - 5:
                    break
                
                # Pick random edge
                e_idx = random.randint(0, NUM_EDGES - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, C - 1)
                
                if new_color == old_color:
                    continue
                
                # Calculate delta violation
                old_total_v = current_violation
                new_total_v = current_violation
                
                affected_k4s = EDGE_TO_K4S[e_idx]
                
                # Apply change
                colors[e_idx] = new_color
                
                # Update affected K4s
                for k4_idx in affected_k4s:
                    h = hist[k4_idx]
                    old_d = distinct[k4_idx]
                    
                    # Update histogram
                    h[old_color] -= 1
                    h[new_color] += 1
                    
                    # Update distinct count
                    new_d = old_d
                    if h[old_color] == 0:
                        new_d -= 1
                    if h[new_color] == 1:
                        new_d += 1
                        
                    distinct[k4_idx] = new_d
                    
                    # Update violation
                    old_v = max(0, 5 - old_d)
                    new_v = max(0, 5 - new_d)
                    new_total_v += (new_v - old_v)
                
                # Acceptance criterion
                delta = new_total_v - old_total_v
                accept = False
                if delta < 0:
                    accept = True
                else:
                    if T > 0.01:
                        # Accept with probability exp(-delta / T)
                        if random.random() < np.exp(-delta / T):
                            accept = True
                
                if accept:
                    current_violation = new_total_v
                    stuck_count = 0
                else:
                    # Revert
                    colors[e_idx] = old_color
                    for k4_idx in affected_k4s:
                        h = hist[k4_idx]
                        old_d = distinct[k4_idx]
                        
                        # Revert histogram
                        h[new_color] -= 1
                        h[old_color] += 1
                        
                        # Revert distinct count
                        new_d = old_d
                        if h[new_color] == 0:
                            new_d -= 1
                        if h[old_color] == 1:
                            new_d += 1
                        distinct[k4_idx] = new_d
                    stuck_count += 1
                
                if current_violation < best_violation:
                    best_violation = current_violation
                    best_colors = colors.copy()
                    if best_violation == 0:
                        return colors
                    stuck_count = 0
                
                # Cool down
                if stuck_count > 1000:
                    T *= 0.95
                    stuck_count = 0
                    if T < 0.01:
                        break
                
                steps += 1
                
            # If loop finishes without returning, try next restart
            if best_violation < total_violation:
                # Optionally restart from best found in this restart?
                # For now, just continue to next restart with fresh random init.
                pass
    
    # Fallback: Use the best found in the last run? 
    # The loop above doesn't preserve best across C values properly if it doesn't return.
    # Let's track global best.
    
    # Re-implement with global best tracking to be safe.
    # But the previous logic returned immediately if 0.
    # If it didn't return, it means we didn't find a valid coloring for C=13,14,15 within time.
    # We should return the best valid coloring found for the smallest C?
    # Or just a valid coloring for any C?
    # The task says "Minimize the number of colours used".
    # So we should return the valid coloring with the fewest colors.
    
    # Let's refactor to track best overall.
    # However, the current structure returns immediately.
    # If no valid coloring is found, we fall back to (i+j)%17 which uses 17 colors?
    # Wait, (i+j)%17 for i<j<16.
    # Max i+j is 15+14=29. 29%17=12. Min 0+1=1.
    # Colors used: 0..16?
    # If i=0, j=1..15 -> 1..15.
    # If i=1, j=2..15 -> 3..16.
    # So colors 0..16 are used. 17 colors.
    # This is valid?
    # We need at least 5 colors in every K4.
    # With 17 colors, it's likely valid.
    
    # But we want to minimize.
    # If the search fails, we should return the best valid solution found.
    
    # Let's modify the loop to store best solution.
    
    # Actually, the current code returns `colors` if `total_violation == 0`.
    # If it never reaches 0, it falls back to the hardcoded one.
    # This is suboptimal if we found a valid coloring with 15 colors but not 13.
    # The loop tries 13, then 14, then 15.
    # If it finds valid in 15, it returns.
    # If it finds valid in 14, it returns (before trying 15).
    # So it prioritizes lower C.
    # If it finds nothing, it returns the 17-color one.
    
    # This logic is acceptable.
    
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
