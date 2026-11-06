import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    # Map (i, j) to edge index
    ij_to_edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute for each K4, the indices of its 6 edges
    k4s = list(combinations(range(n), 4))
    k4_edge_indices = []
    for verts in k4s:
        edges_in_k4 = []
        for i in range(4):
            for j in range(i + 1, 4):
                u = verts[i]
                v = verts[j]
                edges_in_k4.append(ij_to_edge_idx[(u, v)])
        k4_edge_indices.append(edges_in_k4)
    
    # Convert to numpy array for faster operations
    k4_edge_indices_np = np.array(k4_edge_indices, dtype=np.int32)
    
    def is_valid_fast(col):
        """Check if every K4 has at least 5 distinct colors using vectorized numpy."""
        col_arr = np.array(col, dtype=np.int32)
        k4_colors = col_arr[k4_edge_indices_np]  # shape: (1820, 6)
        
        # Count distinct colors in each K4
        sorted_colors = np.sort(k4_colors, axis=1)
        diffs = np.diff(sorted_colors, axis=1)
        distinct_counts = np.sum(diffs != 0, axis=1) + 1
        
        return bool(np.all(distinct_counts >= 5))

    def find_invalid_k4(col):
        """Find one invalid K4 if any, return its index or -1."""
        col_arr = np.array(col, dtype=np.int32)
        k4_colors = col_arr[k4_edge_indices_np]
        sorted_colors = np.sort(k4_colors, axis=1)
        diffs = np.diff(sorted_colors, axis=1)
        distinct_counts = np.sum(diffs != 0, axis=1) + 1
        
        invalid_indices = np.where(distinct_counts < 5)[0]
        if len(invalid_indices) == 0:
            return -1
        return int(invalid_indices[0])

    def count_colors(col):
        return len(set(col))

    best_colors = None
    best_count = 999
    
    # Strategy: Try to find a valid 14-coloring, then try to reduce to 13.
    
    for attempt in range(20): # Limit attempts to ensure time
        if time.time() > deadline:
            break
            
        # 1. Initialize with random 14 colors
        num_colors = 14
        colors = [random.randint(0, num_colors - 1) for _ in range(120)]
        
        # 2. Repair to make valid
        # We might need to add colors if stuck, but let's try with fixed pool first
        # If we can't repair with 14, we might need 15?
        # Actually, if a random 14-coloring is hard to repair, we can increase pool.
        # But we want to minimize.
        
        # Repair loop
        repair_steps = 0
        max_repair_steps = 5000
        while repair_steps < max_repair_steps:
            if time.time() > deadline:
                break
            
            inv_idx = find_invalid_k4(colors)
            if inv_idx == -1:
                break
            
            # Get the edges of this invalid K4
            edge_indices = k4_edge_indices_np[inv_idx]
            
            # Pick a random edge in this K4 to change
            edge_to_change = random.choice(edge_indices)
            
            # Change to a random color in the current pool
            # Sometimes adding a new color helps escape local minima
            if random.random() < 0.1:
                new_color = random.randint(0, 20) # Allow some new colors temporarily?
                # But we want to keep count low.
                # Let's stick to existing colors first.
                new_color = random.randint(0, num_colors - 1)
            else:
                new_color = random.randint(0, num_colors - 1)
            
            old_color = colors[edge_to_change]
            if new_color != old_color:
                colors[edge_to_change] = new_color
            else:
                # If same, try another
                colors[edge_to_change] = (old_color + 1) % num_colors
                
            repair_steps += 1
            
        # Check if valid
        if is_valid_fast(colors):
            current_count = count_colors(colors)
            if current_count < best_count:
                best_count = current_count
                best_colors = colors[:]
                
            # 3. Try to merge colors to reduce count
            # We have a valid coloring with 'current_count' colors.
            # Try to merge any two colors.
            
            used_colors = set(colors)
            # Sort used colors for deterministic iteration?
            # We want to try to eliminate colors.
            
            # If we are at 14, try to get to 13.
            # If we are at 13, try to get to 12 (unlikely).
            
            merge_success = True
            while merge_success and time.time() < deadline:
                merge_success = False
                used_colors_list = sorted(used_colors)
                
                # Try to merge u into v
                for u in used_colors_list:
                    if time.time() > deadline:
                        break
                    for v in used_colors_list:
                        if u == v:
                            continue
                        if u not in used_colors:
                            continue
                        
                        # Create temp
                        temp_colors = colors[:]
                        for k in range(120):
                            if temp_colors[k] == u:
                                temp_colors[k] = v
                            
                        if is_valid_fast(temp_colors):
                            colors = temp_colors
                            used_colors.remove(u)
                            merge_success = True
                            break
                    if merge_success:
                        break
                            
            # If we improved, update best
            if len(used_colors) < best_count:
                best_count = len(used_colors)
                best_colors = colors[:]
                
            # If we reached 13, we can stop early or try for 12?
            # Bounds say 13 is min. So 13 is optimal.
            if best_count <= 13:
                return best_colors

    # If no valid solution found (should not happen), return a fallback
    # Fallback: Random 15 colors?
    if best_colors is None:
        # Construct a known valid 15-coloring?
        # The xor coloring was invalid.
        # Let's just return a random valid one found in repair.
        # If repair failed, we need a guaranteed valid one.
        # A simple valid one: assign each edge a unique color? No, 120 colors.
        # We can use the strategy of "each edge gets color (i+j) % 17" and repair.
        colors = [(i + j) % 17 for i, j in edges]
        # Repair this
        for _ in range(10000):
            inv_idx = find_invalid_k4(colors)
            if inv_idx == -1:
                break
            edge_indices = k4_edge_indices_np[inv_idx]
            edge_to_change = random.choice(edge_indices)
            colors[edge_to_change] = random.randint(0, 16)
            
        if is_valid_fast(colors):
            return colors
        else:
            # Last resort: 120 unique colors
            return list(range(120))

    return best_colors
