import random
import time

# Precompute all K4 edge index combinations
def get_edge_index(i, j):
    if i > j: i, j = j, i
    return 15*i - i*(i-1)//2 + (j - i - 1)

# Generate all K4s and their edge indices
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

# Map edge index to list of K4 indices it belongs to
edge_to_k4s = [[] for _ in range(120)]
for k4_idx, k4_edges in enumerate(k4s):
    for e_idx in k4_edges:
        edge_to_k4s[e_idx].append(k4_idx)

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    # Try 14 first, then 15
    for num_colors in [14, 15]:
        if time.time() - start_time > time_limit - 20:
            break
            
        # Try multiple random restarts
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
                
            coloring = [random.randint(0, num_colors - 1) for _ in range(120)]
            
            # Calculate initial violations
            # violation_count[k4_idx] = 1 if < 5 colors else 0
            violation_count = [0] * len(k4s)
            for k4_idx, edges in enumerate(k4s):
                colors = set(coloring[e] for e in edges)
                if len(colors) < 5:
                    violation_count[k4_idx] = 1
            
            total_violations = sum(violation_count)
            
            # Min-Conflicts Loop
            max_iters = 5000
            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                    
                if total_violations == 0:
                    return coloring
                
                # Find edge with max conflicts
                # conflict[edge_idx] = number of violated K4s containing this edge
                conflicts = [0] * 120
                for k4_idx in range(len(k4s)):
                    if violation_count[k4_idx] == 1:
                        for e_idx in k4s[k4_idx]:
                            conflicts[e_idx] += 1
                
                max_conf = max(conflicts)
                if max_conf == 0:
                    # Should not happen if total_violations > 0
                    break
                
                # Pick a random edge among those with max conflicts
                candidates = [i for i, c in enumerate(conflicts) if c == max_conf]
                e_idx = random.choice(candidates)
                
                # Try all colors for e_idx
                best_color = coloring[e_idx]
                best_cost = total_violations
                
                for c in range(num_colors):
                    if c == coloring[e_idx]:
                        continue
                    
                    # Calculate change in total_violations if we set coloring[e_idx] = c
                    # Only K4s containing e_idx are affected
                    delta = 0
                    old_color = coloring[e_idx]
                    
                    for k4_idx in edge_to_k4s[e_idx]:
                        # Get current colors of the K4
                        # We need to know the set of colors before and after
                        # Efficiently: 
                        # old_set = set of colors in K4
                        # new_set = (old_set - {old_color}) U {c}
                        # But we need to count distinct colors.
                        
                        # Let's just compute the distinct count for the 6 edges
                        # This is small (6 edges), so it's fast enough
                        k4_edges = k4s[k4_idx]
                        
                        old_colors = set()
                        for e in k4_edges:
                            if e == e_idx:
                                old_colors.add(old_color)
                            else:
                                old_colors.add(coloring[e])
                        
                        new_colors = set()
                        for e in k4_edges:
                            if e == e_idx:
                                new_colors.add(c)
                            else:
                                new_colors.add(coloring[e])
                        
                        old_valid = 1 if len(old_colors) >= 5 else 0
                        new_valid = 1 if len(new_colors) >= 5 else 0
                        
                        # violation_count is 1 if invalid
                        # So if old_valid was 0 (invalid), violation was 1.
                        # If new_valid is 1 (valid), violation becomes 0.
                        # Change in violation_count = new_viol - old_viol
                        # old_viol = 1 - old_valid
                        # new_viol = 1 - new_valid
                        # delta += (1 - new_valid) - (1 - old_valid) = old_valid - new_valid
                        
                        delta += (1 if len(old_colors) >= 5 else 0) - (1 if len(new_colors) >= 5 else 0)
                    
                    new_total = total_violations + delta
                    if new_total < best_cost:
                        best_cost = new_total
                        best_color = c
                
                # Update coloring and violation counts
                if best_color != coloring[e_idx]:
                    old_color = coloring[e_idx]
                    coloring[e_idx] = best_color
                    
                    # Update violation_count for affected K4s
                    for k4_idx in edge_to_k4s[e_idx]:
                        k4_edges = k4s[k4_idx]
                        new_colors = set(coloring[e] for e in k4_edges)
                        old_valid = 1 if len(new_colors) >= 5 else 0
                        # Wait, I need the OLD state to update total_violations correctly?
                        # No, I calculated delta.
                        # But I need to update the `violation_count` array for future steps.
                        # So I just recalculate validity for these K4s.
                        violation_count[k4_idx] = 0 if len(new_colors) >= 5 else 1
                    
                    total_violations = best_cost
                    
                else:
                    # No improvement, do a random move to escape local minimum
                    # Pick a random edge and random color
                    e_idx = random.randint(0, 119)
                    old_color = coloring[e_idx]
                    new_color = random.randint(0, num_colors - 1)
                    if new_color != old_color:
                        coloring[e_idx] = new_color
                        for k4_idx in edge_to_k4s[e_idx]:
                            k4_edges = k4s[k4_idx]
                            new_colors = set(coloring[e] for e in k4_edges)
                            violation_count[k4_idx] = 0 if len(new_colors) >= 5 else 1
                        total_violations = sum(violation_count)
            
            # End of min-conflicts loop for this restart

    # Fallback
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
