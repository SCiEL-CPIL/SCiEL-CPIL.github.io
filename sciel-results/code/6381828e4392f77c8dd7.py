import numpy as np
import random
import time
import itertools

# Precompute edges and K4s
N = 16
edges = []
edge_map = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_map[(i, j)] = len(edges)
        edges.append((i, j))

M = len(edges) # 120

k4s = []
k4_edge_indices = []
for quad in itertools.combinations(range(N), 4):
    k4s.append(quad)
    edge_idx_set = []
    for u, v in itertools.combinations(quad, 2):
        edge_idx_set.append(edge_map[(u, v)])
    k4_edge_indices.append(edge_idx_set)

NUM_K4 = len(k4s) # 1820

# Map edge index to list of K4 indices it belongs to
edge_to_k4s = [[] for _ in range(M)]
for k4_idx, edge_indices in enumerate(k4_edge_indices):
    for e_idx in edge_indices:
        edge_to_k4s[e_idx].append(k4_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s safety margin
    
    # We want to minimize colors. Try K=13, 14, 15, 16
    # If we find a valid one with K, we return it.
    # Score is 17 / K.
    
    best_colors = None
    best_k = 17
    
    # Fallback: The baseline solution
    # (i + j) % 17
    baseline = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    # Helper to check if a coloring is valid (all K4s have >= 5 colors)
    def is_valid(colors):
        for edge_indices in k4_edge_indices:
            # Count distinct colors in this K4
            c_set = set()
            for e_idx in edge_indices:
                c_set.add(colors[e_idx])
            if len(c_set) < 5:
                return False
        return True

    # Helper to calculate total violations (number of K4s with < 5 colors)
    def calc_violations(colors):
        count = 0
        for edge_indices in k4_edge_indices:
            c_set = set()
            for e_idx in edge_indices:
                c_set.add(colors[e_idx])
            if len(c_set) < 5:
                count += 1
        return count

    # Incremental update structure
    # We will run local search for each K
    
    for K in [13, 14, 15, 16]:
        if time.time() > deadline - 20: # Reserve 20s for finishing
            break
            
        # Try multiple restarts for this K
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize colors randomly
            colors = np.random.randint(0, K, size=M).astype(int)
            
            # Initial violation count
            viol = np.zeros(NUM_K4, dtype=bool)
            total_viol = 0
            for k4_idx, edge_indices in enumerate(k4_edge_indices):
                c_set = set(colors[e] for e in edge_indices)
                if len(c_set) < 5:
                    viol[k4_idx] = True
                    total_viol += 1
            
            if total_viol == 0:
                best_colors = colors
                best_k = K
                break # Found valid for this K, try smaller K? No, K is increasing. 
                # Wait, we want MINIMIZE colors. So we try 13 first.
                # If we find valid for 13, we are done.
                return list(colors)
            
            # Simulated Annealing
            T = 1.0
            cooling = 0.9995
            min_T = 0.01
            steps_per_restart = 50000
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    break
                
                # Pick random edge
                e_idx = np.random.randint(0, M)
                old_c = colors[e_idx]
                new_c = np.random.randint(0, K)
                
                if old_c == new_c:
                    continue
                
                # Calculate delta violation
                # Affected K4s are those containing e_idx
                delta_viol = 0
                affected_k4s = edge_to_k4s[e_idx]
                
                # We need to check the status of these K4s before and after
                # To be fast, we can just recalculate the distinct count for these K4s
                # But we need to know if they were violated before.
                
                # Let's store the old state of affected K4s
                old_states = {}
                for k4_idx in affected_k4s:
                    old_states[k4_idx] = viol[k4_idx]
                
                # Apply change
                colors[e_idx] = new_c
                
                # Update violations for affected K4s
                for k4_idx in affected_k4s:
                    edge_indices = k4_edge_indices[k4_idx]
                    c_set = set(colors[e] for e in edge_indices)
                    new_viol = len(c_set) < 5
                    if new_viol != old_states[k4_idx]:
                        if new_viol:
                            delta_viol += 1
                        else:
                            delta_viol -= 1
                        viol[k4_idx] = new_viol
                
                new_total_viol = total_viol + delta_viol
                
                # Acceptance criterion
                if delta_viol <= 0:
                    total_viol = new_total_viol
                    if total_viol == 0:
                        best_colors = colors
                        best_k = K
                        return list(colors)
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta_viol / T):
                        total_viol = new_total_viol
                    else:
                        # Revert
                        colors[e_idx] = old_c
                        # Revert viol states
                        for k4_idx in affected_k4s:
                            viol[k4_idx] = old_states[k4_idx]
                
                T *= cooling
                if T < min_T:
                    break
            
            # If we didn't find a solution, we could keep the best found so far
            # But since we restart, we just discard this run if not solved.
            # Actually, it's better to keep the best state from this restart if it's better than previous bests?
            # For simplicity, we just restart.
            
        if best_colors is not None and best_k == K:
             return list(best_colors)
             
    # If no valid coloring found for K <= 16, return the baseline (17 colors)
    # Or return the best invalid one? The task says "Minimize the number of colours used".
    # A valid coloring with 17 colors is better than an invalid one with 13.
    # The evaluator likely checks validity first. "A valid colouring certifies..."
    # If invalid, score is 0.
    # So we MUST return a valid coloring.
    # The baseline is always valid.
    
    # Check if we found a valid one with K=16, 15, 14, or 13 in the loop above.
    # The loop returns immediately if found.
    # So if we are here, we didn't find a valid one with K < 17.
    # We should return the baseline.
    
    return baseline

# Optimization: The above loop logic has a flaw. 
# If I find a valid solution for K=13, I return it.
# If I don't find K=13, I try K=14.
# If I don't find K=14, I try K=15.
# If I don't find K=15, I try K=16.
# If I don't find K=16, I return baseline.
# This is correct.

# However, the inner loop `return list(colors)` inside the restart loop might be too aggressive if it finds a solution 
# but there was a bug? No, `total_viol == 0` means valid.
# The main risk is time.

# Let's refine the time management.
# We will allocate time budget per K.
# K=13: 30s
# K=14: 30s
# K=15: 30s
# K=16: 30s
# Total 120s.

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    N = 16
    M = 120
    
    # Precomputation is done globally
    
    # Helper to check validity quickly
    def is_valid_fast(colors):
        for edge_indices in k4_edge_indices:
            c_set = set(colors[e] for e in edge_indices)
            if len(c_set) < 5:
                return False
        return True

    # Fallback
    baseline = [(i + j) % 17 for i in range(N) for j in range(i + 1, N)]
    
    # Try K = 13, 14, 15, 16
    # We will run a local search for each K.
    
    for K in [13, 14, 15, 16]:
        if time.time() > deadline - 15:
            break
            
        # Time budget for this K
        k_start = time.time()
        k_deadline = k_start + 30.0 # 30 seconds per K
        
        # Try a few restarts
        max_restarts = 100 # We will break based on time
        
        for restart in range(max_restarts):
            if time.time() > k_deadline:
                break
                
            # Initialize
            colors = np.random.randint(0, K, size=M).astype(int)
            
            # Calculate initial violations
            viol = np.zeros(NUM_K4, dtype=bool)
            total_viol = 0
            for k4_idx, edge_indices in enumerate(k4_edge_indices):
                c_set = set(colors[e] for e in edge_indices)
                if len(c_set) < 5:
                    viol[k4_idx] = True
                    total_viol += 1
            
            if total_viol == 0:
                return list(colors)
            
            # Simulated Annealing
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            steps = 20000 # Adjust based on time
            
            for step in range(steps):
                if time.time() > k_deadline - 2: # Leave 2s for next restart or K
                    break
                    
                e_idx = np.random.randint(0, M)
                old_c = colors[e_idx]
                new_c = np.random.randint(0, K)
                
                if old_c == new_c:
                    continue
                
                affected_k4s = edge_to_k4s[e_idx]
                old_states = [viol[k] for k in affected_k4s]
                
                colors[e_idx] = new_c
                
                delta_viol = 0
                for i, k4_idx in enumerate(affected_k4s):
                    edge_indices = k4_edge_indices[k4_idx]
                    c_set = set(colors[e] for e in edge_indices)
                    new_viol = len(c_set) < 5
                    if new_viol != old_states[i]:
                        if new_viol:
                            delta_viol += 1
                        else:
                            delta_viol -= 1
                        viol[k4_idx] = new_viol
                
                new_total_viol = total_viol + delta_viol
                
                if delta_viol <= 0:
                    total_viol = new_total_viol
                    if total_viol == 0:
                        return list(colors)
                else:
                    if random.random() < np.exp(-delta_viol / T):
                        total_viol = new_total_viol
                    else:
                        colors[e_idx] = old_c
                        for i, k4_idx in enumerate(affected_k4s):
                            viol[k4_idx] = old_states[i]
                
                T *= cooling
                if T < min_T:
                    break
            
            # End of restart
    
    # If we didn't find a valid coloring with < 17 colors, return baseline
    return baseline
