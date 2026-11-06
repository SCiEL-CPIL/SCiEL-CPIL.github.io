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
    
    # Strategy: Start with the known valid 17-coloring (i+j)%17, then aggressively merge colors.
    # The 17-coloring is guaranteed valid (or very close), providing a solid starting point.
    
    # 1. Initialize with (i + j) % 17
    colors = [(i + j) % 17 for i, j in edges]
    
    # Verify validity
    if not is_valid_fast(colors):
        # Fallback repair if initial is somehow invalid (shouldn't happen)
        for _ in range(10000):
            if time.time() > deadline: break
            inv_idx = find_invalid_k4(colors)
            if inv_idx == -1: break
            edge_indices = k4_edge_indices_np[inv_idx]
            edge_to_change = random.choice(edge_indices)
            colors[edge_to_change] = random.randint(0, 16)
    
    if is_valid_fast(colors):
        best_count = count_colors(colors)
        best_colors = colors[:]
    
    # 2. Aggressive Merging Loop
    # We want to reduce the number of distinct colors.
    # We iterate through pairs of colors and try to merge one into the other.
    # To avoid local minima, we use a randomized order and potentially restart the merge process.
    
    used_colors = set(colors)
    
    # We'll run this merge phase for a while
    # Each iteration of the outer loop tries to find *any* merge.
    # If no merge is found in a full pass, we might be stuck.
    
    # Optimization: Precompute list of edges for each color?
    # Since colors change, this is hard to maintain efficiently without overhead.
    # 120 edges is small, so linear scan is fast.
    
    while time.time() < deadline:
        # Shuffle the order of trying merges to escape local minima
        # We try to eliminate each color in the current set
        current_used = list(used_colors)
        random.shuffle(current_used)
        
        merged_any = False
        
        for u in current_used:
            if time.time() > deadline:
                break
            if u not in used_colors:
                continue
                
            # Try to merge u into each other color v
            # Shuffle targets too
            targets = [v for v in used_colors if v != u]
            random.shuffle(targets)
            
            for v in targets:
                if time.time() > deadline:
                    break
                if v not in used_colors:
                    continue
                    
                # Create temp state
                temp_colors = colors[:]
                for k in range(120):
                    if temp_colors[k] == u:
                        temp_colors[k] = v
                        
                if is_valid_fast(temp_colors):
                    # Accept merge
                    colors = temp_colors
                    used_colors.remove(u)
                    merged_any = True
                    
                    # Update best
                    current_count = len(used_colors)
                    if current_count < best_count:
                        best_count = current_count
                        best_colors = colors[:]
                        
                    # If we reached the theoretical lower bound (13), we can stop
                    if best_count <= 13:
                        return best_colors
                    break
            
            if merged_any:
                break
        
        if not merged_any:
            # No merge found in this pass.
            # We are stuck in a local minimum for simple merges.
            # Strategy: Try a "kick" - randomly perturb the coloring slightly, 
            # then repair, then try merging again.
            # Or simply break if we've spent enough time.
            
            # Let's try a few random restarts of the "repair + merge" cycle
            # if we are stuck.
            
            # Actually, simple merges from (i+j)%17 might get stuck at 14 or 15.
            # If we are stuck, we can try to *un-merge* some edges? No, that increases colors.
            
            # Alternative: Randomly change a few edge colors to a new random value (within existing palette)
            # to break symmetry, then repair, then merge.
            
            if random.random() < 0.1: # 10% chance to kick
                # Pick 5 random edges and change their color to a random existing color
                for _ in range(5):
                    idx = random.randint(0, 119)
                    colors[idx] = random.choice(list(used_colors))
                
                # Repair the coloring
                for _ in range(1000):
                    if time.time() > deadline: break
                    inv_idx = find_invalid_k4(colors)
                    if inv_idx == -1: break
                    edge_indices = k4_edge_indices_np[inv_idx]
                    edge_to_change = random.choice(edge_indices)
                    # Change to a random existing color
                    colors[edge_to_change] = random.choice(list(used_colors))
                
                if is_valid_fast(colors):
                    # Update used colors
                    used_colors = set(colors)
                    # Continue merging loop
                else:
                    # Revert? We don't have a copy. 
                    # This is risky. Let's revert to best_colors if this fails.
                    colors = best_colors[:]
                    used_colors = set(colors)
            else:
                # If we didn't kick and didn't merge, we might be truly stuck.
                # Given the time budget, we can break.
                # But let's try one more full pass with different random seeds/shuffles.
                # If we still fail, break.
                # For simplicity and safety, let's break after a few failed passes.
                # The current program breaks immediately. Let's try a few more shuffles.
                
                # If we are stuck, it's likely because no single merge is valid.
                # We might need a sequence of merges.
                # But the loop above tries all pairs.
                # So if no single merge works, we are stuck.
                
                # Let's break to save time for potential other strategies if we had any.
                # But we don't. So we return best.
                break

    # Final check
    if best_colors is not None and is_valid_fast(best_colors):
        return best_colors
    
    # Fallback
    if best_colors is None:
        # Return the initial 17-coloring if it was valid
        initial = [(i + j) % 17 for i, j in edges]
        if is_valid_fast(initial):
            return initial
        
        # Last resort: 120 unique colors
        return list(range(120))

    return best_colors
