import time
import random

def construct():
    deadline = time.time() + 145
    random.seed(42)
    
    # Seed: n=32 Greenwood-Gleason
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
    
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    
    def colors_to_adj(colors, n):
        adj = [[-1]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                adj[i][j] = adj[j][i] = colors[idx]
                idx += 1
        return adj
    
    def adj_to_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(adj[i][j])
        return colors
    
    def count_mono_triangles(adj, n):
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = adj[i][j]
                for k in range(j+1, n):
                    if adj[i][k] == c and adj[j][k] == c:
                        count += 1
        return count
    
    def try_add_vertex(adj, n, time_limit=5.0):
        """Try to add vertex n with efficient backtracking. Returns (success, new_adj)."""
        start_time = time.time()
        new_n = n + 1
        
        # For each vertex i, compute which colors are forbidden
        # A color c is forbidden for vertex i if there exist j, k already assigned
        # such that adj[i][j] == c and adj[j][k] == c (triangle i-j-k in color c)
        # But since we're adding one vertex at a time, we need:
        # For new vertex v, color c is forbidden if there exist i, j with
        # adj[i][j] == c and both i and j would get color c from v
        
        # Better approach: for each candidate color c for vertex v,
        # check if it creates a monochromatic triangle
        # Triangle (v, i, j) is monochromatic in color c if:
        #   adj[i][j] == c AND color(v,i) == c AND color(v,j) == c
        
        # So for a fixed color assignment to v, we need to ensure no such triangle exists.
        # This is a constraint satisfaction problem.
        
        # Use backtracking with forward checking
        assigned = [-1] * n  # assigned[i] = color of edge (v, i)
        
        # Order vertices: start with most constrained
        # A vertex i is more constrained if it has many edges of the same color
        vertex_order = list(range(n))
        
        def get_constraint_score(i):
            counts = [0, 0, 0, 0]
            for j in range(n):
                if j != i:
                    counts[adj[i][j]] += 1
            return max(counts)
        
        vertex_order.sort(key=get_constraint_score, reverse=True)
        
        # Precompute for each pair (i, j) the color adj[i][j]
        # For forward checking: when assigning color c to vertex i,
        # for each unassigned vertex j, color c becomes forbidden for j
        # if adj[i][j] == c (because then triangle (v, i, j) would be monochromatic in c
        # if j also gets color c)
        
        # Actually, the constraint is: if assigned[i] = c and assigned[j] = c and adj[i][j] = c,
        # then we have a monochromatic triangle.
        # So for each pair (i, j) with adj[i][j] = c, we cannot have both assigned[i] = c and assigned[j] = c.
        
        # This is like a graph coloring problem on the "conflict graph" where
        # vertices i, j are connected if adj[i][j] = c, and we're coloring with c.
        # But we have 4 colors, so it's more complex.
        
        nodes = [0]
        max_nodes = 100000
        
        def bt(pos_idx):
            if time.time() - start_time > time_limit:
                return False
            if nodes[0] > max_nodes:
                return False
            if pos_idx == n:
                return True
            nodes[0] += 1
            i = vertex_order[pos_idx]
            
            # Try colors in order of least constraining
            for c in [0, 1, 2, 3]:
                conflict = False
                for j_idx in range(pos_idx):
                    j = vertex_order[j_idx]
                    if assigned[j] == c and adj[i][j] == c:
                        conflict = True
                        break
                if not conflict:
                    assigned[i] = c
                    if bt(pos_idx + 1):
                        return True
                    assigned[i] = -1
            return False
        
        if bt(0):
            new_adj = [row[:] for row in adj]
            for _ in range(new_n):
                new_adj.append([-1]*new_n)
            for i in range(n):
                for j in range(n):
                    new_adj[i][j] = adj[i][j]
            for i in range(n):
                new_adj[n][i] = new_adj[i][n] = assigned[i]
            return True, new_adj
        return False, None
    
    # Start
    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    while time.time() < deadline - 5:
        n = len(adj)
        
        success, new_adj = try_add_vertex(adj, n, time_limit=3.0)
        if success:
            adj = new_adj
            best_n = n + 1
            best_colors = adj_to_colors(adj, best_n)
            continue
        
        # Local search: make random valid moves to enable adding a vertex
        improved = False
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j:
                continue
            if j < i:
                i, j = j, i
            
            old_c = adj[i][j]
            new_c = random.choice([0, 1, 2, 3])
            if new_c == old_c:
                continue
            
            delta = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if adj[i][k] == old_c and adj[j][k] == old_c:
                    delta -= 1
                if adj[i][k] == new_c and adj[j][k] == new_c:
                    delta += 1
            
            if delta == 0:
                adj[i][j] = adj[j][i] = new_c
                improved = True
        
        if not improved:
            # Try a different random restart
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                if i == j:
                    continue
                if j < i:
                    i, j = j, i
                old_c = adj[i][j]
                new_c = random.choice([0, 1, 2, 3])
                if new_c == old_c:
                    continue
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if adj[i][k] == old_c and adj[j][k] == old_c:
                        delta -= 1
                    if adj[i][k] == new_c and adj[j][k] == new_c:
                        delta += 1
                if delta == 0:
                    adj[i][j] = adj[j][i] = new_c
                    improved = True
                    break
        
        if not improved:
            break
        
        success, new_adj = try_add_vertex(adj, n, time_limit=3.0)
        if success:
            adj = new_adj
            best_n = n + 1
            best_colors = adj_to_colors(adj, best_n)
        else:
            added = False
            for _ in range(30):
                if time.time() > deadline - 5:
                    break
                for _ in range(30):
                    i = random.randint(0, n-1)
                    j = random.randint(0, n-1)
                    if i == j:
                        continue
                    if j < i:
                        i, j = j, i
                    old_c = adj[i][j]
                    new_c = random.choice([0, 1, 2, 3])
                    if new_c == old_c:
                        continue
                    delta = 0
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if adj[i][k] == old_c and adj[j][k] == old_c:
                            delta -= 1
                        if adj[i][k] == new_c and adj[j][k] == new_c:
                            delta += 1
                    if delta == 0:
                        adj[i][j] = adj[j][i] = new_c
                
                success, new_adj = try_add_vertex(adj, n, time_limit=2.0)
                if success:
                    adj = new_adj
                    best_n = n + 1
                    best_colors = adj_to_colors(adj, best_n)
                    added = True
                    break
            if not added:
                break
    
    adj_final = colors_to_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        return {'n': 32, 'colors': colors_32}
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
