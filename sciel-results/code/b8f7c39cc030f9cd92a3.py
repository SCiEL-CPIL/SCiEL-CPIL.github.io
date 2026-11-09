import random
import time
import itertools

N = 16
edges = []
edge_to_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        edge_to_idx[(i, j)] = len(edges)
        edges.append((i, j))

k4s = []
for quad in itertools.combinations(range(N), 4):
    k4s.append(quad)

# Map edge index to list of K4 indices containing it
edge_to_k4s = [[] for _ in range(len(edges))]
# Also map K4 index to the 6 edge indices in it
k4_edges = []
for k4_idx, quad in enumerate(k4s):
    e_indices = []
    for i in range(4):
        for j in range(i + 1, 4):
            u = quad[i]
            v = quad[j]
            e_idx = edge_to_idx[(u, v)]
            e_indices.append(e_idx)
            edge_to_k4s[e_idx].append(k4_idx)
    k4_edges.append(e_indices)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_valid_colors = None
    best_valid_k = 17
    
    # Try to find a valid coloring with k colors
    for k in [13, 14, 15]:
        # Try a few restarts
        for restart in range(3):
            if time.time() > deadline:
                break
                
            colors = [random.randint(0, k - 1) for _ in range(len(edges))]
            
            # Initialize K4 color counts
            # k4_color_counts[k4_idx][color] = count
            # Since k is small (<=15), we can use a list of size k
            k4_color_counts = [[0] * k for _ in range(len(k4s))]
            k4_distinct_counts = [0] * len(k4s)
            
            for k4_idx in range(len(k4s)):
                e_indices = k4_edges[k4_idx]
                for e_idx in e_indices:
                    c = colors[e_idx]
                    k4_color_counts[k4_idx][c] += 1
                k4_distinct_counts[k4_idx] = sum(1 for cnt in k4_color_counts[k4_idx] if cnt > 0)
            
            bad_count = sum(1 for d in k4_distinct_counts if d < 5)
            
            if bad_count == 0:
                used = set(colors)
                if best_valid_colors is None or len(used) < best_valid_k:
                    best_valid_colors = colors.copy()
                    best_valid_k = len(used)
                return colors
            
            # Local Search
            last_improvement = time.time()
            
            while time.time() < deadline and (time.time() - last_improvement) < 3.0:
                e_idx = random.randint(0, len(edges) - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, k - 1)
                
                if new_color == old_color:
                    continue
                
                delta_bad = 0
                for k4_idx in edge_to_k4s[e_idx]:
                    # Update distinct count for this K4
                    # Decrement old
                    k4_color_counts[k4_idx][old_color] -= 1
                    if k4_color_counts[k4_idx][old_color] == 0:
                        k4_distinct_counts[k4_idx] -= 1
                    # Increment new
                    if k4_color_counts[k4_idx][new_color] == 0:
                        k4_distinct_counts[k4_idx] += 1
                    k4_color_counts[k4_idx][new_color] += 1
                    
                    old_distinct = k4_distinct_counts[k4_idx] - (1 if k4_color_counts[k4_idx][new_color] == 1 and k4_color_counts[k4_idx][old_color] == 0 else 0)
                    # This logic is getting messy. Let's just save old distinct, apply, check, revert if needed.
                    # But we want to calculate delta before committing.
                    
                    # Let's do it properly:
                    # 1. Save old distinct count
                    # 2. Apply change to counts
                    # 3. Calculate new distinct count
                    # 4. Delta = new - old
                    # 5. If we decide to revert, restore counts.
                    
                    pass # Placeholder for correct logic below

                # Correct Logic:
                # We will simulate the change, calculate delta, and if accepted, keep it.
                # If rejected, revert.
                
                delta_bad = 0
                affected_k4s = edge_to_k4s[e_idx]
                
                # Save old state of affected K4s
                old_states = []
                for k4_idx in affected_k4s:
                    old_states.append((k4_color_counts[k4_idx][old_color], k4_color_counts[k4_idx][new_color], k4_distinct_counts[k4_idx]))
                    
                # Apply change
                for k4_idx in affected_k4s:
                    # Decrement old
                    k4_color_counts[k4_idx][old_color] -= 1
                    if k4_color_counts[k4_idx][old_color] == 0:
                        k4_distinct_counts[k4_idx] -= 1
                    # Increment new
                    if k4_color_counts[k4_idx][new_color] == 0:
                        k4_distinct_counts[k4_idx] += 1
                    k4_color_counts[k4_idx][new_color] += 1
                    
                # Calculate delta_bad
                for k4_idx in affected_k4s:
                    new_distinct = k4_distinct_counts[k4_idx]
                    old_distinct = old_states[affected_k4s.index(k4_idx)][2]
                    
                    if old_distinct < 5 and new_distinct < 5:
                        pass
                    elif old_distinct < 5 and new_distinct >= 5:
                        delta_bad -= 1
                    elif old_distinct >= 5 and new_distinct < 5:
                        delta_bad += 1
                    else:
                        pass
                        
                # Decide accept/reject
                if delta_bad < 0:
                    # Accept
                    colors[e_idx] = new_color
                    bad_count += delta_bad
                    last_improvement = time.time()
                    
                    if bad_count == 0:
                        used = set(colors)
                        if best_valid_colors is None or len(used) < best_valid_k:
                            best_valid_colors = colors.copy()
                            best_valid_k = len(used)
                        return colors
                else:
                    # Revert
                    for i, k4_idx in enumerate(affected_k4s):
                        old_old_c, old_new_c, old_distinct = old_states[i]
                        k4_color_counts[k4_idx][old_color] = old_old_c
                        k4_color_counts[k4_idx][new_color] = old_new_c
                        k4_distinct_counts[k4_idx] = old_distinct

    # Fallback
    if best_valid_colors:
        return best_valid_colors
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
