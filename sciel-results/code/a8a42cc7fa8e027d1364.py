import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n = 16
    total_edges = n * (n - 1) // 2
    
    # Build edge index mapping
    edge_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = i * (2 * n - i - 1) // 2 + (j - i - 1)
    
    # Precompute for each edge, the list of (other_edge_index, k4_id) pairs
    # Actually, we need for each edge, the K4s it belongs to, and for each K4, its 6 edges
    
    # Build all K4s
    k4_list = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    edges = [
                        edge_idx[(a, b)], edge_idx[(a, c)], edge_idx[(a, d)],
                        edge_idx[(b, c)], edge_idx[(b, d)], edge_idx[(c, d)]
                    ]
                    k4_list.append(edges)
    
    num_k4 = len(k4_list)
    
    # For each edge, store list of (k4_index, position_in_k4)
    edge_to_k4 = [[] for _ in range(total_edges)]
    for ki, edges in enumerate(k4_list):
        for pos, ei in enumerate(edges):
            edge_to_k4[ei].append((ki, pos))
    
    # For each K4, store the 6 edge indices
    k4_edges = k4_list  # list of 6 edge indices per K4
    
    # Initial coloring: 1-factorization (15 colors)
    colors = np.zeros(total_edges, dtype=np.int32)
    factors = [[] for _ in range(15)]
    for r in range(15):
        factors[r].append((15, r))
        for k in range(1, 8):
            u = (r + k) % 15
            v = (r - k) % 15
            if u > v:
                u, v = v, u
            factors[r].append((u, v))
    
    for r in range(15):
        for (u, v) in factors[r]:
            if u > v:
                u, v = v, u
            colors[edge_idx[(u, v)]] = r
    
    # Compute initial penalty: for each K4, count distinct colors
    # penalty = number of K4s with < 5 distinct colors
    k4_color_counts = np.zeros((num_k4, 16), dtype=np.int32)  # max 15 colors initially
    k4_distinct = np.zeros(num_k4, dtype=np.int32)
    
    for ki, edges in enumerate(k4_edges):
        for ei in edges:
            c = colors[ei]
            k4_color_counts[ki, c] += 1
        k4_distinct[ki] = np.count_nonzero(k4_color_counts[ki])
    
    penalty = int(np.sum(k4_distinct < 5))
    
    # Track best
    best_penalty = penalty
    best_colors = colors.copy()
    best_num_colors = int(np.max(colors)) + 1
    
    # Simulated annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.99995
    
    # Precompute random edge pairs for swaps
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Pick a random edge and a random new color
        ei = rng.integers(0, total_edges)
        
        # Determine number of colors to try: use current max + some buffer
        max_c = int(np.max(best_colors)) + 2
        new_c = rng.integers(0, max_c)
        
        old_c = colors[ei]
        if old_c == new_c:
            continue
        
        # Compute delta penalty
        # For each K4 containing edge ei, check if changing old_c to new_c affects distinct count
        delta = 0
        for (ki, pos) in edge_to_k4[ei]:
            # Before change
            old_count = k4_color_counts[ki, old_c]
            new_count_before = k4_color_counts[ki, new_c]
            
            # After change: old_c count decreases by 1, new_c count increases by 1
            old_count_after = old_count - 1
            new_count_after = new_count_before + 1
            
            # Distinct count change
            old_distinct = k4_distinct[ki]
            
            # If old_c was the only one of its kind, distinct decreases by 1
            # If new_c was not present, distinct increases by 1
            distinct_change = 0
            if old_count_after == 0:
                distinct_change -= 1
            if new_count_before == 0:
                distinct_change += 1
            
            new_distinct = old_distinct + distinct_change
            
            # Penalty contribution: 1 if distinct < 5, else 0
            old_penalty = 1 if old_distinct < 5 else 0
            new_penalty = 1 if new_distinct < 5 else 0
            delta += new_penalty - old_penalty
        
        # Acceptance
        if delta <= 0:
            # Always accept improving or neutral moves
            # Apply the change
            colors[ei] = new_c
            for (ki, pos) in edge_to_k4[ei]:
                k4_color_counts[ki, old_c] -= 1
                k4_color_counts[ki, new_c] += 1
                old_d = k4_distinct[ki]
                # Recompute distinct
                if k4_color_counts[ki, old_c] == 0:
                    k4_distinct[ki] = old_d - 1
                if k4_color_counts[ki, new_c] == 1:
                    k4_distinct[ki] = k4_distinct[ki] + 1
            penalty += delta
        else:
            # Accept with probability exp(-delta/T)
            if np.random.random() < np.exp(-delta / T):
                colors[ei] = new_c
                for (ki, pos) in edge_to_k4[ei]:
                    k4_color_counts[ki, old_c] -= 1
                    k4_color_counts[ki, new_c] += 1
                    old_d = k4_distinct[ki]
                    if k4_color_counts[ki, old_c] == 0:
                        k4_distinct[ki] = old_d - 1
                    if k4_color_counts[ki, new_c] == 1:
                        k4_distinct[ki] = k4_distinct[ki] + 1
                penalty += delta
            # else reject
        
        # Update best
        if penalty < best_penalty:
            best_penalty = penalty
            best_colors = colors.copy()
            best_num_colors = int(np.max(colors)) + 1
        
        # If we've found a valid solution, try to minimize colors
        if best_penalty == 0:
            # Try to reduce colors by remapping
            pass
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final: try to minimize number of colors used
    # Remap colors to use fewer colors if possible
    final_colors = best_colors.copy()
    
    # Count colors used
    used_colors = np.unique(final_colors)
    num_used = len(used_colors)
    
    # Try to remap to reduce colors: greedy approach
    # Create a mapping from old color to new color
    # We want to minimize the max color index
    if num_used > 13:
        # Try to compress
        color_map = {}
        next_color = 0
        for c in sorted(used_colors):
            color_map[c] = next_color
            next_color += 1
        final_colors = np.array([color_map[c] for c in final_colors], dtype=np.int32)
    
    # Convert to list in lexicographic order
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            idx = edge_idx[(i, j)]
            result.append(int(final_colors[idx]))
    
    return result
