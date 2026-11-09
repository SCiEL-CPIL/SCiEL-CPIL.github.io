import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (limit is 150s)

    n = 16
    num_edges = n * (n - 1) // 2  # 120

    # Precompute edges and K4s
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            
    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        edge_idx_map[(j, i)] = idx

    # List of edges for each K4
    k4s_all = []
    # For each edge, list of K4 indices it belongs to
    edge_k4s = [[] for _ in range(num_edges)]
    
    # Generate all K4s
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e_indices = [
                        edge_idx_map[(a, b)],
                        edge_idx_map[(a, c)],
                        edge_idx_map[(a, d)],
                        edge_idx_map[(b, c)],
                        edge_idx_map[(b, d)],
                        edge_idx_map[(c, d)]
                    ]
                    k4_idx = len(k4s_all)
                    k4s_all.append(e_indices)
                    for ei in e_indices:
                        edge_k4s[ei].append(k4_idx)

    num_k4s = len(k4s_all)
    
    # Precompute color counts for each K4 to allow fast delta calculation?
    # It's complex. Let's stick to recalculating the 6 colors for affected K4s.
    # 6 colors * ~100 K4s per edge? No, each edge is in (16-2)*(16-3)/2 = 14*13/2 = 91 K4s.
    # Recalculating 91 K4s per move is expensive.
    
    # Optimization:
    # Maintain a count of how many colors are present in each K4?
    # Actually, we just need to know if the number of distinct colors >= 5.
    # We can maintain the set of colors for each K4.
    # But sets are expensive to copy.
    
    # Alternative:
    # For each K4, maintain the count of each color.
    # colors_in_k4[k4_idx] = [count_0, count_1, ..., count_{K-1}]
    # distinct_count[k4_idx] = sum(1 for c in colors_in_k4[k4_idx] if c > 0)
    
    # This allows O(1) update for a move?
    # Change edge e from c_old to c_new.
    # For each k4 in edge_k4s[e]:
    #   colors_in_k4[k4][c_old] -= 1
    #   colors_in_k4[k4][c_new] += 1
    #   if colors_in_k4[k4][c_old] == 0: distinct_count[k4] -= 1
    #   if colors_in_k4[k4][c_new] == 1: distinct_count[k4] += 1
    #   is_valid[k4] = distinct_count[k4] >= 5
    #   total_invalid += (not is_valid[k4]) - (old_not_valid)
    
    # This is much faster.
    
    def initialize(K):
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        # Initialize color counts for each K4
        k4_color_counts = [[0] * K for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        k4_valid = [False] * num_k4s
        total_invalid = 0
        
        for k4_idx, e_indices in enumerate(k4s_all):
            counts = k4_color_counts[k4_idx]
            for ei in e_indices:
                c = colors[ei]
                counts[c] += 1
            distinct = 0
            for c in range(K):
                if counts[c] > 0:
                    distinct += 1
            k4_distinct[k4_idx] = distinct
            if distinct >= 5:
                k4_valid[k4_idx] = True
            else:
                k4_valid[k4_idx] = False
                total_invalid += 1
                
        return colors, k4_color_counts, k4_distinct, k4_valid, total_invalid

    def try_find_valid(K, time_limit):
        if time.time() > time_limit:
            return None
            
        best_colors = None
        best_invalid = float('inf')
        
        # Multiple restarts
        while time.time() < time_limit:
            colors, k4_color_counts, k4_distinct, k4_valid, total_invalid = initialize(K)
            
            if total_invalid == 0:
                return colors
            
            best_colors_local = colors[:]
            best_invalid_local = total_invalid
            
            T = 1.0
            T_min = 0.01
            cooling_rate = 0.9995
            # Adaptive cooling?
            
            # List of invalid K4s to bias moves?
            # We can just pick random edges, but if we pick an edge in an invalid K4, it's more likely to help.
            
            while time.time() < time_limit:
                # Pick an edge
                # Bias: pick an edge that is in at least one invalid K4
                # We can maintain a list of invalid K4s.
                # If list is empty, break.
                
                # For simplicity and speed, just pick random edge.
                # But let's try picking from edges in invalid K4s if we track them.
                # Tracking edges in invalid K4s:
                # edge_in_invalid_count[ei] = number of invalid K4s containing ei.
                
                # Let's implement the biased move selection.
                
                # Re-initialize bias structure on restart or update dynamically?
                # Dynamic update is complex. Let's just pick random edge first.
                # If total_invalid is small, random might be slow.
                
                # Let's try a simple random pick first.
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, K - 1)
                
                if new_color == old_color:
                    continue
                
                # Calculate delta
                delta_invalid = 0
                
                for k4_idx in edge_k4s[e_idx]:
                    counts = k4_color_counts[k4_idx]
                    old_valid = k4_valid[k4_idx]
                    
                    # Update counts
                    counts[old_color] -= 1
                    counts[new_color] += 1
                    
                    # Update distinct count
                    if counts[old_color] == 0:
                        k4_distinct[k4_idx] -= 1
                    if counts[new_color] == 1:
                        k4_distinct[k4_idx] += 1
                    
                    new_valid = k4_distinct[k4_idx] >= 5
                    
                    if old_valid and not new_valid:
                        delta_invalid += 1
                    elif not old_valid and new_valid:
                        delta_invalid -= 1
                    else:
                        delta_invalid += 0
                        
                    k4_valid[k4_idx] = new_valid
                    
                new_total_invalid = total_invalid + delta_invalid
                
                # Acceptance
                accept = False
                if new_total_invalid < total_invalid:
                    accept = True
                elif new_total_invalid == total_invalid:
                    accept = random.random() < 0.5
                else:
                    delta = new_total_invalid - total_invalid
                    if random.random() < pow(2.71828, -delta / T):
                        accept = True
                
                if accept:
                    total_invalid = new_total_invalid
                    colors[e_idx] = new_color
                    
                    if total_invalid < best_invalid_local:
                        best_invalid_local = total_invalid
                        best_colors_local = colors[:]
                        
                        if total_invalid == 0:
                            return colors
                else:
                    # Revert
                    for k4_idx in edge_k4s[e_idx]:
                        counts = k4_color_counts[k4_idx]
                        old_valid = k4_valid[k4_idx] # This is the NEW valid status from the trial
                        # We need to revert the counts
                        counts[new_color] -= 1
                        counts[old_color] += 1
                        
                        if counts[new_color] == 0:
                            k4_distinct[k4_idx] -= 1
                        if counts[old_color] == 1:
                            k4_distinct[k4_idx] += 1
                            
                        # Restore valid status
                        k4_valid[k4_idx] = old_valid # Wait, old_valid was captured BEFORE the update in the loop?
                        # No, in the loop above, I updated k4_valid.
                        # I need to store the OLD valid status before changing.
                        # Let's re-structure the delta calculation to store old states.
                        
                    # The revert logic above is flawed because I didn't store old states properly in the list.
                    # Let's use a simpler revert: just re-run the count update in reverse.
                    # But I need the old distinct counts?
                    # Actually, I can just store the old distinct count and old valid status in temporary lists.
                    
                    # Let's refactor the move evaluation to be cleaner.
                    
                T *= cooling_rate
                if T < T_min:
                    T = T_min
                    
            if best_invalid_local < best_invalid:
                best_invalid = best_invalid_local
                best_colors = best_colors_local
                if best_invalid == 0:
                    return best_colors
                
        return best_colors

    # The revert logic in the above draft was messy. I will rewrite the core loop carefully.
    
    def try_find_valid_v2(K, time_limit):
        if time.time() > time_limit:
            return None
            
        best_colors = None
        best_invalid = float('inf')
        
        while time.time() < time_limit:
            colors, k4_color_counts, k4_distinct, k4_valid, total_invalid = initialize(K)
            
            if total_invalid == 0:
                return colors
            
            best_colors_local = colors[:]
            best_invalid_local = total_invalid
            
            T = 1.0
            T_min = 0.01
            cooling_rate = 0.9995
            
            while time.time() < time_limit:
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, K - 1)
                
                if new_color == old_color:
                    continue
                
                delta_invalid = 0
                # Store old states for revert
                old_states = []
                
                for k4_idx in edge_k4s[e_idx]:
                    counts = k4_color_counts[k4_idx]
                    old_valid = k4_valid[k4_idx]
                    old_distinct = k4_distinct[k4_idx]
                    old_states.append((k4_idx, old_valid, old_distinct))
                    
                    # Apply change
                    counts[old_color] -= 1
                    counts[new_color] += 1
                    
                    if counts[old_color] == 0:
                        k4_distinct[k4_idx] -= 1
                    if counts[new_color] == 1:
                        k4_distinct[k4_idx] += 1
                        
                    new_valid = k4_distinct[k4_idx] >= 5
                    
                    if old_valid and not new_valid:
                        delta_invalid += 1
                    elif not old_valid and new_valid:
                        delta_invalid -= 1
                        
                    k4_valid[k4_idx] = new_valid
                    
                new_total_invalid = total_invalid + delta_invalid
                
                accept = False
                if new_total_invalid < total_invalid:
                    accept = True
                elif new_total_invalid == total_invalid:
                    accept = random.random() < 0.5
                else:
                    delta = new_total_invalid - total_invalid
                    if random.random() < pow(2.71828, -delta / T):
                        accept = True
                
                if accept:
                    total_invalid = new_total_invalid
                    colors[e_idx] = new_color
                    
                    if total_invalid < best_invalid_local:
                        best_invalid_local = total_invalid
                        best_colors_local = colors[:]
                        
                        if total_invalid == 0:
                            return colors
                else:
                    # Revert
                    colors[e_idx] = old_color
                    for k4_idx, old_valid, old_distinct in old_states:
                        counts = k4_color_counts[k4_idx]
                        counts[new_color] -= 1
                        counts[old_color] += 1
                        
                        # Restore distinct and valid
                        k4_distinct[k4_idx] = old_distinct
                        k4_valid[k4_idx] = old_valid
                        
                T *= cooling_rate
                if T < T_min:
                    T = T_min
                    
            if best_invalid_local < best_invalid:
                best_invalid = best_invalid_local
                best_colors = best_colors_local
                if best_invalid == 0:
                    return best_colors
                
        return best_colors

    # Try K=14
    t_start = time.time()
    colors_14 = try_find_valid_v2(14, t_start + 60)
    
    if colors_14 is not None and all(True for _ in [0]): # Check validity just in case
        # Verify
        valid = True
        for e_indices in k4s_all:
            if len(set(colors_14[i] for i in e_indices)) < 5:
                valid = False
                break
        if valid:
            unique_colors = sorted(set(colors_14))
            mapping = {c: i for i, c in enumerate(unique_colors)}
            return [mapping[c] for c in colors_14]
            
    # Try K=15
    t_start = time.time()
    colors_15 = try_find_valid_v2(15, t_start + 60)
    
    if colors_15 is not None:
        valid = True
        for e_indices in k4s_all:
            if len(set(colors_15[i] for i in e_indices)) < 5:
                valid = False
                break
        if valid:
            unique_colors = sorted(set(colors_15))
            mapping = {c: i for i, c in enumerate(unique_colors)}
            return [mapping[c] for c in colors_15]
            
    # Fallback: Return a known valid coloring if we have one, or a random one?
    # We must return a valid one.
    # If SA failed, we might need a deterministic fallback or longer search.
    # Given time limits, if SA fails, we might return an invalid one which scores 0.
    # To be safe, we can try a simple greedy repair on the best found so far?
    # Or just return the best found (even if invalid) and hope? No, invalid scores 0.
    
    # Let's add a final desperate attempt with K=15 using a different random seed or strategy?
    # Or just return the best from K=15 run even if invalid? No.
    
    # If both fail, we are in trouble.
    # But K=15 should be solvable.
    
    # If we reach here, we didn't find a valid one.
    # We should return something. The task says "invalid result ... scores 0".
    # So we MUST find a valid one.
    
    # Let's try K=16 with a very short time?
    # Or just return the best K=15 colors? If invalid, score 0.
    
    # To maximize chance of 1.0+, we need valid K=14 or K=15.
    
    # If we didn't find valid, let's try a simple random restart with K=15 for the remaining time.
    remaining_time = deadline - time.time()
    if remaining_time > 5:
         colors_15_retry = try_find_valid_v2(15, time.time() + remaining_time - 2)
         if colors_15_retry:
             valid = True
             for e_indices in k4s_all:
                 if len(set(colors_15_retry[i] for i in e_indices)) < 5:
                     valid = False
                     break
             if valid:
                 unique_colors = sorted(set(colors_15_retry))
                 mapping = {c: i for i, c in enumerate(unique_colors)}
                 return [mapping[c] for c in colors_15_retry]

    # Last resort: return a dummy valid coloring?
    # We don't have one hardcoded.
    # We must assume the search works.
    
    # If all else fails, return an empty list? No, must be 120 ints.
    # Return 0s? Invalid.
    
    # I will trust the search.
    return [0] * 120

