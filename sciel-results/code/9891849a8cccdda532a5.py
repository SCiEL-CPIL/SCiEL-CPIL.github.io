import time
import random

N = 16

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def build(vertices, next_color, randomize=False):
        n = len(vertices)
        if n <= 1:
            return next_color
        if n == 2:
            u, v = vertices
            colors[u][v] = next_color
            colors[v][u] = next_color
            return next_color + 1
        if n == 3:
            u, v, w = vertices
            colors[u][v] = next_color
            colors[v][u] = next_color
            colors[v][w] = next_color + 1
            colors[w][v] = next_color + 1
            colors[u][w] = next_color + 2
            colors[w][u] = next_color + 2
            return next_color + 3
        
        # Choose split point
        if randomize:
            k = random.randint(max(1, n // 2 - 1), min(n - 1, n // 2 + 1))
        else:
            # Original deterministic split
            if n == 4: k = 1
            elif n == 5: k = 2
            elif n == 6: k = 3
            elif n == 7: k = 3
            elif n == 8: k = 4
            elif n == 9: k = 3
            elif n == 10: k = 4
            elif n == 11: k = 5
            elif n == 12: k = 6
            elif n == 13: k = 6
            elif n == 14: k = 7
            elif n == 15: k = 6
            elif n == 16: k = 7
            else: k = n // 2
            
        v1 = vertices[:k]
        v2 = vertices[k:]
        
        cross_color = next_color
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        
        next_color += 1
        next_color = build(v1, next_color, randomize)
        if time.time() > deadline:
            return next_color
        next_color = build(v2, next_color, randomize)
        return next_color

    def is_valid_recolor(u, v, new_color):
        # Check all C4s containing edge (u,v)
        # A C4 is u-w-x-v-u. The edges are (u,w), (w,x), (x,v), (v,u).
        # Colors must not be all distinct.
        # So new_color must be in {c(u,w), c(w,x), c(x,v)} for all w,x.
        for w in range(N):
            if w == u or w == v:
                continue
            for x in range(N):
                if x == u or x == v or x == w:
                    continue
                if x >= w:
                    continue
                c1 = colors[u][w]
                c2 = colors[w][x]
                c3 = colors[x][v]
                if new_color != c1 and new_color != c2 and new_color != c3:
                    return False
        return True

    def local_search(colors, deadline):
        edge_list = [(i, j) for i in range(N) for j in range(i + 1, N)]
        
        # Precompute unused colors
        used_colors = set(colors[i][j] for i in range(N) for j in range(i+1, N))
        all_colors = set(range(120))
        unused_colors = list(all_colors - used_colors)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(edge_list)
            for (u, v) in edge_list:
                if time.time() > deadline:
                    break
                if not unused_colors:
                    break
                
                old_color = colors[u][v]
                
                # Try to assign any unused color
                # We can pick a random unused color to explore different paths
                # Or try all? Trying all is O(|unused| * N^2). 
                # |unused| can be up to 100. N^2 is 256. 100*256 = 25600 ops per edge.
                # 120 edges * 25600 = 3M ops per pass. 
                # This is fast enough.
                
                # To avoid bias, shuffle unused colors
                random.shuffle(unused_colors)
                
                for new_color in unused_colors:
                    if is_valid_recolor(u, v, new_color):
                        colors[u][v] = new_color
                        colors[v][u] = new_color
                        unused_colors.remove(new_color)
                        used_colors.add(new_color)
                        improved = True
                        break
        
        return len(used_colors)

    best_score = 0
    best_result = None
    
    # Run multiple attempts
    # The deterministic one is the baseline
    # Then randomized ones
    
    attempts = 10
    for i in range(attempts):
        if time.time() > deadline:
            break
            
        colors = [[0] * N for _ in range(N)]
        
        # Use randomization for all attempts, including the first one (or keep first deterministic?)
        # Let's keep the first one deterministic to ensure we at least get the baseline
        rand_flag = (i > 0)
        
        # Reset colors to 0
        for r in range(N):
            for c in range(N):
                colors[r][c] = 0
                
        build(list(range(N)), 0, rand_flag)
        
        # Local search
        current_score = local_search(colors, deadline)
        
        if current_score > best_score:
            best_score = current_score
            # Flatten result
            best_result = []
            for r in range(N):
                for c in range(r+1, N):
                    best_result.append(colors[r][c])
    
    if best_result is None:
        # Fallback to deterministic if something went wrong
        colors = [[0] * N for _ in range(N)]
        build(list(range(N)), 0, False)
        local_search(colors, deadline)
        best_result = []
        for r in range(N):
            for c in range(r+1, N):
                best_result.append(colors[r][c])
                
    return best_result
