import random
import time

N = 16

def construct():
    # Generate all edges in lexicographic order
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    
    best_colors = 0
    best_assignment = None
    
    # Increase restarts to find a better solution
    # The inner loop is fast, so we can afford more restarts
    num_restarts = 2000
    
    for _ in range(num_restarts):
        # Shuffle edge processing order to explore different colorings
        shuffled_edges = edges[:]
        random.shuffle(shuffled_edges)
        
        # colors[i][j] stores the color of edge (i, j). -1 if uncolored.
        colors = [[-1 for _ in range(N)] for _ in range(N)]
        used_colors = set()
        color_freq = {} # Track frequency of colors
        
        success = True
        
        for u, v in shuffled_edges:
            # Find all paths of length 3 between u and v
            # Path: u - w - x - v
            # w != u, v; x != u, v, w
            rainbow_path_colors = []
            
            for w in range(N):
                if w == u or w == v:
                    continue
                if colors[u][w] == -1:
                    continue
                for x in range(N):
                    if x == u or x == v or x == w:
                        continue
                    if colors[w][x] == -1:
                        continue
                    if colors[x][v] == -1:
                        continue
                    
                    c1 = colors[u][w]
                    c2 = colors[w][x]
                    c3 = colors[x][v]
                    
                    # Check if these 3 edges form a rainbow path
                    if len({c1, c2, c3}) == 3:
                        rainbow_path_colors.append((c1, c2, c3))
            
            if not rainbow_path_colors:
                # No rainbow path of length 3 exists.
                # We can safely assign a new color.
                new_color = len(used_colors)
                # Ensure we don't exceed 119 colors
                if new_color >= 120:
                    # Fallback: reuse a random color
                    new_color = random.choice(list(used_colors)) if used_colors else 0
                colors[u][v] = new_color
                colors[v][u] = new_color
                used_colors.add(new_color)
                color_freq[new_color] = color_freq.get(new_color, 0) + 1
            else:
                # Must pick a color that is present in ALL rainbow paths
                # to prevent any of them from becoming a rainbow 4-cycle.
                common_colors = None
                for p in rainbow_path_colors:
                    s = set(p)
                    if common_colors is None:
                        common_colors = s
                    else:
                        common_colors = common_colors.intersection(s)
                
                if common_colors is not None and len(common_colors) > 0:
                    # Heuristic: Pick the most frequent color among the candidates
                    # to encourage clustering and reduce future rainbow paths
                    chosen_color = max(common_colors, key=lambda c: color_freq.get(c, 0))
                    colors[u][v] = chosen_color
                    colors[v][u] = chosen_color
                    color_freq[chosen_color] = color_freq.get(chosen_color, 0) + 1
                else:
                    # Dead end: No color can fix all rainbow paths.
                    # Restart this attempt.
                    success = False
                    break
        
        if success:
            current_count = len(used_colors)
            if current_count > best_colors:
                best_colors = current_count
                # Construct the result array in lexicographic order
                res = []
                for i in range(N):
                    for j in range(i + 1, N):
                        res.append(colors[i][j])
                best_assignment = res
                
                # Early exit if we hit the theoretical max? 
                # Max colors is 120. If we reach 120, we can stop.
                if best_colors == 120:
                    break
    
    if best_assignment is None:
        # Fallback to all zeros if no valid coloring found (unlikely)
        return [0] * (N * (N - 1) // 2)
        
    return best_assignment
