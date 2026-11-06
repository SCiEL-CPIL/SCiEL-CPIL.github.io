import time
import random

def construct():
    deadline = time.time() + 135
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
    
    def try_add_vertex(adj, n):
        """Try to add vertex n. Returns (success, new_adj)."""
        new_n = n + 1
        new_adj = [row[:] for row in adj]
        for _ in range(new_n):
            new_adj.append([-1]*new_n)
        for i in range(n):
            for j in range(n):
                new_adj[i][j] = adj[i][j]
        
        assigned = [-1] * n
        
        # Order vertices by constraint
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
            if nodes[0] > max_nodes:
                return False
            if pos_idx == n:
                return True
            nodes[0] += 1
            i = vertex_order[pos_idx]
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
            for i in range(n):
                new_adj[n][i] = new_adj[i][n] = assigned[i]
            return True, new_adj
        return False, None
    
    # Start
    adj = colors_to_adj(colors_32, 32)
    best_n = 32
    best_colors = colors_32
    
    while time.time() < deadline - 10:
        n = len(adj)
        
        success, new_adj = try_add_vertex(adj, n)
        if success:
            adj = new_adj
            best_n = n + 1
            best_colors = adj_to_colors(adj, best_n)
            continue
        
        # Local search: make random valid moves
        improved = False
        for _ in range(100):
            if time.time() > deadline - 10:
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
            break
        
        success, new_adj = try_add_vertex(adj, n)
        if success:
            adj = new_adj
            best_n = n + 1
            best_colors = adj_to_colors(adj, best_n)
        else:
            added = False
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                for _ in range(20):
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
                
                success, new_adj = try_add_vertex(adj, n)
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
