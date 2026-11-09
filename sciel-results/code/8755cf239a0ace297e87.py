import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Generate the base n=32 coloring (Greenwood-Gleason / Clebsch doubling)
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    
    def generate_seed_coloring(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    # Internal to block
                    # The seed code says: log[(i % 16) ^ (j % 16)] % 3
                    # This is a 3-coloring of K16.
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    # Between blocks
                    colors.append(3)
        return colors

    # We will work with a 2D adjacency matrix for easier manipulation
    def colors_to_matrix(colors, n):
        mat = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                mat[i][j] = c
                mat[j][i] = c
                idx += 1
        return mat

    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(mat[i][j])
        return colors

    # Initialize with n=32
    current_n = 32
    base_colors = generate_seed_coloring(current_n)
    mat = colors_to_matrix(base_colors, current_n)
    
    best_n = current_n
    best_colors = base_colors
    
    # Try to expand to larger n
    # We will try to add vertices one by one up to n=40 or until time runs out
    max_target_n = 40
    
    for target_n in range(current_n + 1, max_target_n + 1):
        if time.time() > deadline:
            break
            
        # Add new vertex
        new_mat = [[0]*target_n for _ in range(target_n)]
        for i in range(target_n):
            for j in range(target_n):
                if i == j:
                    new_mat[i][j] = -1
                elif i < target_n and j < target_n:
                    # Copy old
                    if i < current_n and j < current_n:
                        new_mat[i][j] = mat[i][j]
                    else:
                        # New edges
                        # Initialize randomly
                        new_mat[i][j] = random.randint(0, 3)
                        new_mat[j][i] = new_mat[i][j]
        
        mat = new_mat
        current_n = target_n
        
        # Local Search to remove triangles
        # Objective: Minimize number of monochromatic triangles
        
        def count_triangles(m):
            count = 0
            for i in range(current_n):
                for j in range(i+1, current_n):
                    c = m[i][j]
                    if c < 0: continue
                    for k in range(j+1, current_n):
                        if m[i][k] == c and m[j][k] == c:
                            count += 1
            return count
        
        # Optimization: Track triangle counts per vertex?
        # For n<=40, O(N^3) is fast enough.
        # 40^3 = 64,000. 1000 iterations = 64M ops. Python might be slow.
        # Let's use a faster update or limit iterations.
        
        # Better: Calculate "badness" for each vertex.
        # A triangle (i,j,k) contributes 1 to badness of i, j, k.
        
        def get_badness(m):
            badness = [0]*current_n
            for i in range(current_n):
                for j in range(i+1, current_n):
                    c = m[i][j]
                    for k in range(j+1, current_n):
                        if m[i][k] == c and m[j][k] == c:
                            badness[i] += 1
                            badness[j] += 1
                            badness[k] += 1
            return badness, sum(badness)//3 # Total triangles
            
        # Search loop
        max_iters = 2000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            badness, total_tris = get_badness(mat)
            
            if total_tris == 0:
                # Found valid coloring
                best_n = current_n
                best_colors = matrix_to_colors(mat, current_n)
                break
                
            # Find vertex with max badness
            worst_v = max(range(current_n), key=lambda x: badness[x])
            
            # Try changing colors of edges (worst_v, u)
            best_gain = 0
            best_change = None
            
            # For each neighbor u, try all 4 colors
            for u in range(current_n):
                if u == worst_v: continue
                old_c = mat[worst_v][u]
                for new_c in range(4):
                    if new_c == old_c: continue
                    
                    # Calculate change in triangles
                    # Triangles involving (worst_v, u) and some w
                    # Old triangles: count w such that mat[worst_v][w]==old_c and mat[u][w]==old_c
                    # New triangles: count w such that mat[worst_v][w]==new_c and mat[u][w]==new_c
                    # Note: This is an approximation if we only change one edge at a time.
                    # But it's a valid heuristic.
                    
                    old_count = 0
                    new_count = 0
                    for w in range(current_n):
                        if w == worst_v or w == u: continue
                        if mat[worst_v][w] == old_c and mat[u][w] == old_c:
                            old_count += 1
                        if mat[worst_v][w] == new_c and mat[u][w] == new_c:
                            new_count += 1
                            
                    gain = old_count - new_count
                    if gain > best_gain:
                        best_gain = gain
                        best_change = (u, new_c)
                        
            if best_change:
                u, new_c = best_change
                mat[worst_v][u] = new_c
                mat[u][worst_v] = new_c
            else:
                # Random kick
                u = random.randint(0, current_n-1)
                if u != worst_v:
                    mat[worst_v][u] = random.randint(0, 3)
                    mat[u][worst_v] = mat[worst_v][u]
                else:
                    # Pick another
                    u2 = (worst_v + 1) % current_n
                    mat[worst_v][u2] = random.randint(0, 3)
                    mat[u2][worst_v] = mat[worst_v][u2]
            
            if time.time() > deadline:
                break

        # If we didn't find 0 triangles, we keep the best_n found so far
        # But we must ensure we don't return an invalid coloring for the new n.
        # The loop above updates best_n only if total_tris == 0.
        # If it didn't break, we haven't updated best_n for this target_n.
        # So we just continue to next target_n? 
        # No, if we fail to fix n=33, we should probably not try n=34 with the broken mat.
        # We should revert to the last valid state.
        
        # Check if we actually succeeded
        badness, total_tris = get_badness(mat)
        if total_tris != 0:
            # Revert to previous valid state
            # We need to restore mat to the previous valid one.
            # This logic is getting complex. 
            # Simpler: Only proceed to next n if current n is valid.
            # If not valid, break the expansion loop.
            break
            
    return {'n': best_n, 'colors': best_colors}
