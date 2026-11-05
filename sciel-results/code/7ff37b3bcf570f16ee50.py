import time
import random

_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    if len(colors) != n * (n - 1) // 2:
        return False
    # Build adjacency sets for each color
    # adj[c][u] = set of v such that color(u,v)=c
    adj = [ [set() for _ in range(n)] for _ in range(5) ]
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c < 0 or c > 4:
                return False
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
            
    # Check for C4
    # A C4 exists if there are two vertices u, v with |N_c(u) intersect N_c(v)| >= 2
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                # Only check if they are not connected? No, C4 can be in any subgraph.
                # If u and v are connected by color c, they are adjacent.
                # If they have 2 common neighbors, it's a C4.
                # If they are NOT connected, and have 2 common neighbors, it's a C4.
                # Actually, any two vertices with 2 common neighbors in color c form a C4.
                common = adj[c][u].intersection(adj[c][v])
                if len(common) >= 2:
                    return False
    return True

def construct():
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find a larger n
    # Greedy construction with restarts
    max_n_to_try = 29
    
    while _time_left() > 10:
        # Try to construct a valid coloring for a random n between 12 and max_n_to_try
        # Or increase n gradually
        target_n = random.randint(12, max_n_to_try)
        
        # Greedy build
        colors = []
        # We store colors in a 2D list for easy access during construction
        # But the output needs a flat list.
        # Let's use a list of lists for adjacency during construction
        # adj[i][j] = color
        adj = [[-1] * target_n for _ in range(target_n)]
        
        success = True
        for k in range(1, target_n):
            # Choose colors for edges (0, k), ..., (k-1, k)
            # Shuffle order of i to add randomness
            order = list(range(k))
            random.shuffle(order)
            
            for i in order:
                # Determine valid colors for (i, k)
                valid_colors = []
                for c in range(5):
                    # Check if adding color c to (i, k) creates a C4
                    # Count common neighbors of i and k in color c
                    common = 0
                    for j in range(k):
                        if j == i:
                            continue
                        # We need adj[j][i] == c and adj[j][k] == c
                        # adj[j][k] is already set for j < i? No, we are processing i in shuffled order.
                        # So adj[j][k] might not be set yet if j > i in the original index?
                        # No, we are adding edges to k.
                        # When we are at step i, we have already assigned colors for some j < k.
                        # Specifically, we have assigned colors for (j, k) for all j in order[0..current_index-1].
                        # We have NOT assigned colors for (j, k) for j in order[current_index+1..].
                        # So we cannot check full intersection.
                        # This simple greedy fails because it doesn't know future colors.
                        
                        # Better approach:
                        # For a fixed i, we must choose a color c.
                        # This choice restricts future choices.
                        # A valid color c is one that does NOT immediately create a C4 with EXISTING edges.
                        # But it might create a C4 with FUTURE edges.
                        # To be safe, we should only choose c if the number of common neighbors (among those already connected to k) is < 2.
                        # And we hope that future connections don't complete a C4.
                        
                        pass # This logic is flawed for simple greedy.
                
                # Let's use a different heuristic:
                # For each i, pick a color that minimizes the number of "potential" C4s.
                # Or just pick a random valid color based on current state.
                
                # Current state:
                # For each j < k, if adj[j][k] is set, we know it.
                # For each j < k, adj[j][i] is known (since j < i < k, edge (j,i) was set when vertex i was added).
                
                best_c = -1
                best_score = float('inf')
                for c in range(5):
                    # Count common neighbors j < k such that adj[j][i] == c and adj[j][k] == c
                    # Note: adj[j][k] is only set if j was processed before i in the current loop.
                    # If j was not processed yet, we don't know adj[j][k].
                    # So we can only count confirmed common neighbors.
                    confirmed_common = 0
                    for j in range(k):
                        if j == i:
                            continue
                        if adj[j][i] == c and adj[j][k] == c:
                            confirmed_common += 1
                            
                    # If confirmed_common >= 2, this color is INVALID immediately.
                    if confirmed_common >= 2:
                        continue
                        
                    # Score: lower is better. We want to avoid colors that have many "partial" common neighbors.
                    # A partial common neighbor is j such that adj[j][i] == c but adj[j][k] is not yet set.
                    # If we pick c, and later pick c for (j,k), we might get a C4.
                    # So we should avoid colors with many such j.
                    potential_common = 0
                    for j in range(k):
                        if j == i:
                            continue
                        if adj[j][i] == c and adj[j][k] == -1:
                            potential_common += 1
                            
                    score = confirmed_common * 10 + potential_common
                    if score < best_score:
                        best_score = score
                        best_c = c
                        
                if best_c == -1:
                    success = False
                    break
                adj[i][k] = best_c
                adj[k][i] = best_c
                
        if success:
            # Flatten adj to colors
            flat_colors = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    flat_colors.append(adj[i][j])
                    
            if _validate(target_n, flat_colors):
                if target_n > best_n:
                    best_n = target_n
                    best_colors = flat_colors
                    
    # Return the best found
    return {'n': best_n, 'colors': best_colors}
