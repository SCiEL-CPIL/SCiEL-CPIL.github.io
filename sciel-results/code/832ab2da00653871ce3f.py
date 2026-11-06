import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n = 18
    
    def is_k4_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                for w in range(v + 1, n):
                    if not (adj[u] >> w) & 1 or not (adj[v] >> w) & 1:
                        continue
                    common = adj[u] & adj[v] & adj[w]
                    common &= ~(1 << u) & ~(1 << v) & ~(1 << w)
                    if common:
                        return False
        return True
    
    def max_triangle_free_size(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                for c in range(b + 1, n):
                    if (adj[a] >> c) & 1 and (adj[b] >> c) & 1:
                        triangles.append((a, b, c))
        
        tri_count = [0] * n
        for (a, b, c) in triangles:
            tri_count[a] += 1
            tri_count[b] += 1
            tri_count[c] += 1
        
        order = sorted(range(n), key=lambda x: -tri_count[x])
        
        pair_tri = [[0] * n for _ in range(n)]
        for (a, b, c) in triangles:
            pair_tri[a][b] |= (1 << c)
            pair_tri[b][a] |= (1 << c)
            pair_tri[a][c] |= (1 << b)
            pair_tri[c][a] |= (1 << b)
            pair_tri[b][c] |= (1 << a)
            pair_tri[c][b] |= (1 << a)
        
        def can_add(v, current_mask):
            for u_bit in range(n):
                if not (current_mask >> u_bit) & 1:
                    continue
                common = pair_tri[v][u_bit] & current_mask
                if common:
                    return False
            return True
        
        best = [0]
        
        def search(pos, current_mask, current_size):
            if current_size > best[0]:
                best[0] = current_size
            if pos >= n:
                return
            if current_size + (n - pos) <= best[0]:
                return
            
            v = order[pos]
            search(pos + 1, current_mask, current_size)
            if can_add(v, current_mask):
                search(pos + 1, current_mask | (1 << v), current_size + 1)
        
        search(0, 0, 0)
        return best[0]
    
    candidates = []
    
    # Candidate 1: Turan graph K6,6,6
    adj_turan = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj_turan[u] |= (1 << v)
                    adj_turan[v] |= (1 << u)
    if is_k4_free(adj_turan):
        candidates.append(adj_turan[:])
    
    # Candidate 2: Paley(17) + isolated vertex
    adj_paley = [0] * n
    residues = {1, 4, 9, 16, 8, 2, 15, 13}
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (u - v) % 17
            if diff in residues:
                adj_paley[u] |= (1 << v)
                adj_paley[v] |= (1 << u)
    if is_k4_free(adj_paley):
        candidates.append(adj_paley[:])
    
    best_adj = None
    best_score = float('inf')
    for adj in candidates:
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    if best_adj is None:
        return []
    
    adj = best_adj[:]
    current_score = best_score
    
    def adj_to_edge_set(adj):
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.add((u, v))
        return edges
    
    current_edges = adj_to_edge_set(adj)
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        u, v = random.choice(all_edges)
        
        if (u, v) in current_edges:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            current_edges.discard((u, v))
            flipped = 'remove'
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            current_edges.add((u, v))
            flipped = 'add'
        
        if not is_k4_free(adj):
            if flipped == 'remove':
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges.discard((u, v))
            continue
        
        new_score = max_triangle_free_size(adj)
        
        if new_score < current_score:
            current_score = new_score
            best_score = new_score
            best_adj = adj[:]
        else:
            if flipped == 'remove':
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                current_edges.discard((u, v))
    
    result_adj = best_adj
    result_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (result_adj[u] >> v) & 1:
                result_edges.append([u, v])
    
    return result_edges
