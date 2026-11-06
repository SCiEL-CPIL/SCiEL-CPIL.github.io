import time
import random

def construct():
    deadline = time.time() + 140
    random.seed(42)
    
    # Greenwood-Gleason n=32 seed
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
    
    def try_add_vertex(adj, n, time_limit=2.0):
        """Try to add vertex n with backtracking. Returns (success, assigned_colors)."""
        start_time = time.time()
        assigned = [-1] * n
        
        # Order vertices by how constrained they are
        vertex_order = list(range(n))
        def constraint_score(i):
            counts = [0,0,0,0]
            for j in range(n):
                if i != j:
                    counts[adj[i][j]] += 1
            return max(counts)
        vertex_order.sort(key=constraint_score, reverse=True)
        
        nodes = [0]
        max_nodes = 50000
        
        def bt(pos_idx):
            if time.time() - start_time > time_limit:
                return False
            if nodes[0] > max_nodes:
                return False
            if pos_idx == n:
                return True
            nodes[0] += 1
            i = vertex_order[pos_idx]
            # Try colors in order of least conflict
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
            return True, assigned
        return False, None
    
    # Start with n=32
    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    # Try to add vertices immediately
    while time.time() < deadline - 10:
        n = len(adj)
        success, assigned = try_add_vertex(adj, n, time_limit=3.0)
        if success:
            new_n = n + 1
            new_adj = [row[:] for row in adj]
            for _ in range(new_n):
                new_adj.append([-1]*new_n)
            for i in range(n):
                for j in range(n):
                    new_adj[i][j] = adj[i][j]
            for i in range(n):
                new_adj[n][i] = new_adj[i][n] = assigned[i]
            adj = new_adj
            best_n = new_n
            best_colors = adj_to_colors(adj, best_n)
            continue
        
        # Local search to enable adding a vertex
        # Use simulated annealing: accept worsening moves with some probability
        improved = False
        temp = 1.0
        for step in range(500):
            if time.time() > deadline - 10:
                break
            if temp < 0.1:
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
            
            # Accept if delta <= 0, or with probability exp(-delta/temp) if delta > 0
            if delta <= 0 or random.random() < (0.5 ** delta):
                adj[i][j] = adj[j][i] = new_c
                improved = True
            temp *= 0.995
        
        if not improved:
            # Try more aggressive random restarts
            for _ in range(10):
                if time.time() > deadline - 10:
                    break
                for _ in range(50):
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
                    if delta <= 1:  # Allow slight worsening
                        adj[i][j] = adj[j][i] = new_c
                
                success, assigned = try_add_vertex(adj, n, time_limit=2.0)
                if success:
                    new_n = n + 1
                    new_adj = [row[:] for row in adj]
                    for _ in range(new_n):
                        new_adj.append([-1]*new_n)
                    for i in range(n):
                        for j in range(n):
                            new_adj[i][j] = adj[i][j]
                    for i in range(n):
                        new_adj[n][i] = new_adj[i][n] = assigned[i]
                    adj = new_adj
                    best_n = new_n
                    best_colors = adj_to_colors(adj, best_n)
                    break
            else:
                break
            break
    
    # Final validation
    adj_final = colors_to_adj(best_colors, best_n)
    if count_mono_triangles(adj_final, best_n) > 0:
        return {'n': 32, 'colors': colors_32}
    
    return {'n': best_n, 'colors': best_colors}
