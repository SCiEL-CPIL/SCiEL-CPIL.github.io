import numpy as np
import time
import random
from itertools import combinations

# Precompute all K4s
V = 16
K4s = list(combinations(range(V), 4))
# Map edge to list of K4 indices containing that edge
edge_to_k4s = { (i,j): [] for i in range(V) for j in range(i+1, V) }
for idx, k4 in enumerate(K4s):
    for e in combinations(k4, 2):
        edge_to_k4s[e].append(idx)

def count_distinct_colors(k4, colors):
    c1, c2, c3, c4 = k4
    # 6 edges
    cols = {
        colors[c1, c2],
        colors[c1, c3],
        colors[c1, c4],
        colors[c2, c3],
        colors[c2, c4],
        colors[c3, c4]
    }
    return len(cols)

def get_cost(colors):
    cost = 0
    for k4 in K4s:
        if count_distinct_colors(k4, colors) < 5:
            cost += 1
    return cost

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    best_colors = None
    best_cost = float('inf')
    best_num_colors = 17
    
    # Try to find a valid coloring with k colors for k in 13, 14, 15
    for k in [13, 14, 15]:
        # Try multiple restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize
            colors = np.random.randint(0, k, (V, V))
            np.fill_diagonal(colors, -1) # Not used
            # Make symmetric
            for i in range(V):
                for j in range(i+1, V):
                    colors[j, i] = colors[i, j]
            
            current_cost = get_cost(colors)
            
            # Hill Climbing / SA
            temp = 1.0
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Pick random edge
                i = random.randint(0, V-1)
                j = random.randint(i+1, V-1)
                old_color = colors[i, j]
                new_color = random.randint(0, k-1)
                
                if new_color == old_color:
                    continue
                    
                # Calculate delta cost
                # Only K4s containing (i,j) change
                delta = 0
                for k4_idx in edge_to_k4s[(i,j)]:
                    k4 = K4s[k4_idx]
                    old_count = count_distinct_colors(k4, colors)
                    
                    # Temporarily change
                    colors[i, j] = new_color
                    colors[j, i] = new_color
                    new_count = count_distinct_colors(k4, colors)
                    
                    # Revert for now, just to calculate delta? 
                    # Actually, we need to update the global cost if we accept.
                    # For delta, we just compare old vs new for this specific K4.
                    # But wait, other K4s in the list might also contain (i,j).
                    # The cost contribution of a K4 is 1 if count < 5, else 0.
                    # old_contrib = 1 if old_count < 5 else 0
                    # new_contrib = 1 if new_count < 5 else 0
                    # delta += new_contrib - old_contrib
                    
                    # We must be careful: we changed the color, so we must revert before checking next K4?
                    # No, the color change is global for this edge.
                    # So we can just compute the new total cost for these specific K4s?
                    # No, we need the delta of the total cost.
                    # Total Cost = Sum( I(count(k4) < 5) )
                    # Delta = Sum( I(new_count(k4) < 5) - I(old_count(k4) < 5) ) for k4 in affected
                    
                    # Let's just compute the new cost of the affected K4s and subtract old.
                    # We need to store old counts? Or just recompute?
                    # Recomputing is fast.
                    
                    # Let's do it properly:
                    # 1. Save old colors
                    # 2. Apply new color
                    # 3. Calculate new cost for affected K4s
                    # 4. Calculate old cost for affected K4s (we have old_color, so we can restore or just calculate)
                    # 5. Delta = New - Old
                    
                    # Since we are iterating, let's just do:
                    # Save state, apply change, calc delta, revert if not accepting.
                    pass 
                
                # This incremental logic is tricky with the "restore" step inside the loop if we don't accept.
                # Simpler: 
                # 1. Save old_color.
                # 2. Apply new_color.
                # 3. Calculate new_cost for affected K4s.
                # 4. Calculate old_cost for affected K4s (using old_color).
                # 5. delta = new_cost - old_cost.
                # 6. If accept, keep new_color. Else, revert.
                
                # Let's implement this.
                affected_k4s = edge_to_k4s[(i,j)]
                
                # Calculate old cost contribution
                old_contrib = 0
                for k4_idx in affected_k4s:
                    k4 = K4s[k4_idx]
                    if count_distinct_colors(k4, colors) < 5: # Current colors (old)
                        old_contrib += 1
                        
                # Apply change
                colors[i, j] = new_color
                colors[j, i] = new_color
                
                # Calculate new cost contribution
                new_contrib = 0
                for k4_idx in affected_k4s:
                    k4 = K4s[k4_idx]
                    if count_distinct_colors(k4, colors) < 5: # New colors
                        new_contrib += 1
                        
                delta = new_contrib - old_contrib
                
                # Acceptance criterion (SA)
                if delta < 0 or random.random() < np.exp(-delta / temp):
                    current_cost += delta
                    # Keep change
                else:
                    # Revert
                    colors[i, j] = old_color
                    colors[j, i] = old_color
                    
                temp *= 0.999
                
                if current_cost == 0:
                    break
                    
            if current_cost == 0:
                # Found a valid coloring with k colors
                return [int(colors[i, j]) for i in range(V) for j in range(i+1, V)]
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                best_num_colors = k
                
    # If no valid coloring found with < 17 colors, return the seed solution
    # The seed solution is valid and uses 17 colors.
    # We should probably return the seed to guarantee validity if our search fails.
    # Or return the best found if it's valid? No, if not valid, it's invalid.
    # The prompt says "A valid colouring certifies...". An invalid result scores 0.
    # So we MUST return a valid coloring.
    # The seed is always valid.
    
    # Check if we found a valid one in the loop?
    # The loop returns immediately if found.
    # So if we are here, we didn't find a valid one with 13, 14, 15.
    # We should return the seed.
    
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
