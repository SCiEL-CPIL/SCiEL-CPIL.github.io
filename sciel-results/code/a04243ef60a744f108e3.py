# EVOLVE-BLOCK-START
import numpy as np
import random

def construct():
    # Precompute edge indices and K4s
    edge_idx = {}
    idx = 0
    for i in range(16):
        for j in range(i+1, 16):
            edge_idx[(i,j)] = idx
            idx += 1
            
    k4s = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    edges = [
                        edge_idx[(i,j)], edge_idx[(i,k)], edge_idx[(i,l)],
                        edge_idx[(j,k)], edge_idx[(j,l)], edge_idx[(k,l)]
                    ]
                    k4s.append(edges)
                    
    # Map edge -> list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edges in enumerate(k4s):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)
            
    # Function to get distinct colors count for a K4
    def get_distinct_count(colors, edges):
        # edges is a list of 6 indices
        # Use a set
        return len(set(colors[e] for e in edges))

    def check_violations(colors):
        # Returns list of violating K4 indices
        viol = []
        for k4_idx, edges in enumerate(k4s):
            if get_distinct_count(colors, edges) < 5:
                viol.append(k4_idx)
        return viol

    def find_coloring(K, max_iters=5000):
        # Initialize random colors
        colors = np.random.randint(0, K, size=120)
        
        # Initial violations
        current_violations = check_violations(colors)
        num_viol = len(current_violations)
        
        if num_viol == 0:
            return colors.tolist()
            
        best_colors = colors.copy()
        best_num_viol = num_viol
        
        # Simulated Annealing / Hill Climbing
        # Temperature starts high, cools down
        # We accept worse moves with probability exp(-delta/T)
        
        T = 1.0
        cooling_rate = 0.999
        
        for i in range(max_iters):
            if num_viol == 0:
                return colors.tolist()
                
            # Pick a random violating K4
            # If no violations, we are done.
            # If we have violations, pick one.
            # To avoid recalculating the list of violations every time (expensive),
            # we pick a random K4 and check if it's violating.
            # If not, we skip. This is inefficient if violations are sparse.
            # But with incremental updates, we can maintain the list?
            # Maintaining a dynamic list of violations is complex.
            # Let's just pick a random K4.
            
            k4_idx = np.random.randint(0, len(k4s))
            edges = k4s[k4_idx]
            
            # Check if this K4 is currently violating
            curr_distinct = get_distinct_count(colors, edges)
            if curr_distinct >= 5:
                continue # Not a violation, try next iteration
                
            # It is a violation. Try to fix it.
            # Pick a random edge in this K4
            edge_pos = np.random.randint(0, 6)
            e_idx = edges[edge_pos]
            
            old_color = colors[e_idx]
            new_color = np.random.randint(0, K)
            
            if new_color == old_color:
                continue
                
            # Calculate change in violations
            # Only K4s containing e_idx are affected.
            delta_viol = 0
            
            # Store old states to revert if needed
            affected_k4s = edge_to_k4s[e_idx]
            
            # We need to update the violation status for each affected K4
            # But we don't have a global "is_violating" array easily accessible in O(1) for all K4s
            # unless we maintain one.
            
            # Let's maintain an array `is_violating` of size 1820.
            # This requires updating it whenever we change a color.
            
            # Let's restructure to maintain `is_violating`.
            
            # For now, let's just do a full check if we want simplicity, 
            # but that's too slow.
            
            # Let's implement the incremental update properly.
            
            # We need to know the distinct count BEFORE and AFTER for each affected K4.
            
            # 1. Compute distinct count for affected K4s BEFORE change
            before_counts = {}
            for k_idx in affected_k4s:
                before_counts[k_idx] = get_distinct_count(colors, k4s[k_idx])
                
            # 2. Apply change
            colors[e_idx] = new_color
            
            # 3. Compute distinct count for affected K4s AFTER change
            after_counts = {}
            for k_idx in affected_k4s:
                after_counts[k_idx] = get_distinct_count(colors, k4s[k_idx])
                
            # 4. Calculate delta
            for k_idx in affected_k4s:
                was_viol = before_counts[k_idx] < 5
                is_viol = after_counts[k_idx] < 5
                if was_viol and not is_viol:
                    delta_viol -= 1
                elif not was_viol and is_viol:
                    delta_viol += 1
                    
            # 5. Acceptance
            if delta_viol < 0:
                # Better
                num_viol += delta_viol
                if num_viol < best_num_viol:
                    best_num_viol = num_viol
                    best_colors = colors.copy()
            elif delta_viol == 0:
                # Same, accept
                pass
            else:
                # Worse
                # Accept with probability
                if np.random.random() < np.exp(-delta_viol / T):
                    num_viol += delta_viol
                else:
                    # Revert
                    colors[e_idx] = old_color
                    
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
                
        return best_colors.tolist() if best_num_viol == 0 else None

    # Try K=13, 14, 15
    for K in [13, 14, 15]:
        found = None
        for _ in range(10): # 10 restarts
            res = find_coloring(K, max_iters=5000)
            if res is not None:
                found = res
                break
        if found is not None:
            return found

    # Fallback to Z_16 sum coloring (16 colors)
    cols = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(i+1, 16):
            c = (i + j) % 16
            cols[i][j] = c
            cols[j][i] = c
            
    return [cols[i][j] for i in range(16) for j in range(i+1, 16)]

# EVOLVE-BLOCK-END
