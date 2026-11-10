import random
import time
import math

def get_edge_index(i, j):
    if i > j: i, j = j, i
    return 15*i - i*(i-1)//2 + (j - i - 1)

# Generate all K4s
k4s = []
verts = range(16)
for a in range(16):
    for b in range(a+1, 16):
        for c in range(b+1, 16):
            for d in range(c+1, 16):
                idxs = [
                    get_edge_index(a, b),
                    get_edge_index(a, c),
                    get_edge_index(a, d),
                    get_edge_index(b, c),
                    get_edge_index(b, d),
                    get_edge_index(c, d)
                ]
                k4s.append(tuple(idxs))

# Precompute edge to K4 mapping
edge_to_k4s = [[] for _ in range(120)]
for k4_idx, k4 in enumerate(k4s):
    for edge_idx in k4:
        edge_to_k4s[edge_idx].append(k4_idx)

def is_violation(coloring, k4):
    c_set = set(coloring[idx] for idx in k4)
    return len(c_set) < 5

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    for num_colors in [14, 15, 16]:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Initialize random coloring
        coloring = [random.randint(0, num_colors - 1) for _ in range(120)]
        
        # Initialize violation status
        viol_status = [0] * len(k4s)
        total_violations = 0
        for k4_idx, k4 in enumerate(k4s):
            if is_violation(coloring, k4):
                viol_status[k4_idx] = 1
                total_violations += 1
                
        if total_violations == 0:
            return coloring
            
        best_coloring = coloring[:]
        best_violations = total_violations
        
        T = 1.0
        cooling_rate = 0.995
        min_T = 0.01
        
        while T > min_T and time.time() - start_time < time_limit - 5:
            edge_idx = random.randint(0, 119)
            new_color = random.randint(0, num_colors - 1)
            
            if coloring[edge_idx] == new_color:
                continue
                
            old_color = coloring[edge_idx]
            
            # Calculate delta violations
            delta = 0
            for k4_idx in edge_to_k4s[edge_idx]:
                was_viol = viol_status[k4_idx]
                
                # Temporarily change
                coloring[edge_idx] = new_color
                now_viol = is_violation(coloring, k4s[k4_idx])
                
                if was_viol and not now_viol:
                    delta -= 1
                elif not was_viol and now_viol:
                    delta += 1
                
                # Revert for next check? No, we need to keep the change to check next K4?
                # Wait, is_violation checks the current state of `coloring`.
                # If we change `coloring[edge_idx]` to `new_color`, all subsequent checks in this loop
                # will see the new color. This is correct because we are evaluating the state
                # AFTER the change for all affected K4s.
                # However, we need to update `viol_status` only if we accept.
                # For now, we just calculate delta based on the hypothetical new state.
                
            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Apply change
                coloring[edge_idx] = new_color
                total_violations += delta
                
                # Update viol_status
                for k4_idx in edge_to_k4s[edge_idx]:
                    # We need to re-evaluate or track?
                    # It's safer to re-evaluate for the ones we touched, but we already did in the delta loop?
                    # No, in the delta loop we checked `now_viol` but didn't store it.
                    # Let's just re-evaluate for the affected K4s to update status.
                    # It's cheap.
                    viol_status[k4_idx] = 1 if is_violation(coloring, k4s[k4_idx]) else 0
                    
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_coloring = coloring[:]
                    
                if total_violations == 0:
                    return coloring
            else:
                # Revert
                coloring[edge_idx] = old_color
                
            T *= cooling_rate
            
        if best_violations == 0:
            return best_coloring
            
    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
