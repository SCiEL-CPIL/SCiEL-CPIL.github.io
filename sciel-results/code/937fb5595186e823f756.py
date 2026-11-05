import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    # Initial 17-coloring
    colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Precompute K4 edge indices
    # Vertices 0..15
    # Edge index for (i, j) with i<j is i*15 - i*(i-1)//2 + (j-i-1) ? 
    # Let's use a dictionary or a formula.
    # Total edges 120.
    # Index of (i, j) in lexicographic order:
    # Number of edges before i: sum_{k=0}^{i-1} (15-k) = 15*i - i*(i-1)/2
    # Index of j in row i: j - i - 1
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return 15*i - i*(i-1)//2 + (j - i - 1)
        
    # Generate all K4s
    k4s = []
    for a in range(16):
        for b in range(a+1, 16):
            for c in range(b+1, 16):
                for d in range(c+1, 16):
                    edges = [
                        edge_idx(a, b), edge_idx(a, c), edge_idx(a, d),
                        edge_idx(b, c), edge_idx(b, d),
                        edge_idx(c, d)
                    ]
                    k4s.append(edges)
                    
    def is_valid(col_list):
        for edges in k4s:
            # Get colors of the 6 edges
            c0 = col_list[edges[0]]
            c1 = col_list[edges[1]]
            c2 = col_list[edges[2]]
            c3 = col_list[edges[3]]
            c4 = col_list[edges[4]]
            c5 = col_list[edges[5]]
            # Count distinct
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                return False
        return True

    # Check initial
    if not is_valid(colors):
        # Should not happen
        pass

    active_colors = set(colors)
    
    # Greedy Merging
    # We try to merge colors until no more merges are possible.
    # To avoid getting stuck in a specific local minimum, we can restart with shuffled color labels or random order of attempts.
    # But since the structure is symmetric, maybe one pass is enough?
    # Let's try multiple passes with random pair selection.
    
    best_colors = colors
    best_count = len(active_colors)
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        current_colors = best_colors.copy()
        current_active = set(current_colors)
        
        # Shuffle the list of colors to try merges in random order
        # We represent colors by their integer values.
        # We want to merge c1, c2.
        # If we merge, we replace all c2 with c1.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Create a list of color pairs to try
            # Only try pairs that are in current_active
            cols_list = list(current_active)
            random.shuffle(cols_list)
            
            for i in range(len(cols_list)):
                for j in range(i+1, len(cols_list)):
                    c1 = cols_list[i]
                    c2 = cols_list[j]
                    if c1 not in current_active or c2 not in current_active:
                        continue
                        
                    # Try merge
                    new_colors = current_colors.copy()
                    for k in range(120):
                        if new_colors[k] == c2:
                            new_colors[k] = c1
                            
                    if is_valid(new_colors):
                        current_colors = new_colors
                        current_active.remove(c2)
                        improved = True
                        if len(current_active) < best_count:
                            best_count = len(current_active)
                            best_colors = current_colors.copy()
                        # Restart the inner loop as colors changed
                        break
                if improved:
                    break
                    
    return best_colors
