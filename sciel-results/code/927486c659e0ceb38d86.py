import time
import random

N = 18

def is_k4_free(adj):
    for a in range(N):
        for b in range(a+1, N):
            if not adj[a][b]:
                continue
            for c in range(b+1, N):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, N):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def largest_triangle_free(adj, deadline):
    best = [0]
    
    # For each vertex, precompute neighbors
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    def bnb(candidates, current, current_size):
        if time.time() > deadline:
            return
        if current_size + len(candidates) <= best[0]:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Choose vertex with most conflicts for better pruning
        v = candidates[0]
        remaining = candidates[1:]
        
        # Check if v can be added (no triangle formed)
        can_add = True
        for u in current:
            if u in neighbors[v]:
                # Check if any w in current is adjacent to both u and v
                for w in current:
                    if w != u and w in neighbors[u] and w in neighbors[v]:
                        can_add = False
                        break
            if not can_add:
                break
        
        if can_add:
            new_current = current | {v}
            bnb(remaining, new_current, current_size + 1)
        
        bnb(remaining, current, current_size)
    
    bnb(list(range(N)), set(), 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 135
    random.seed(42)
    
    def make_k4_free(adj):
        changed = True
        while changed:
            changed = False
            for a in range(N):
                for b in range(a+1, N):
                    if not adj[a][b]:
                        continue
                    for c in range(b+1, N):
                        if not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c+1, N):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                adj[a][d] = adj[d][a] = False
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        return adj
    
    # Try multiple random starts
    best_adj = None
    best_score = 999
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        adj = [[False]*N for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.35:
                    adj[i][j] = adj[j][i] = True
        adj = make_k4_free(adj)
        
        current_score = largest_triangle_free(adj, deadline)
        
        # Local search
        for _ in range(200):
            if time.time() > deadline:
                break
            i = random.randint(0, N-1)
            j = random.randint(0, N-1)
            if i == j:
                continue
            i, j = min(i,j), max(i,j)
            
            new_adj = [row[:] for row in adj]
            new_adj[i][j] = not new_adj[i][j]
            new_adj[j][i] = new_adj[i][j]
            
            if not is_k4_free(new_adj):
                continue
            
            score = largest_triangle_free(new_adj, deadline)
            if score < current_score:
                adj = new_adj
                current_score = score
            else:
                # Revert
                adj[i][j] = not adj[i][j]
                adj[j][i] = adj[i][j]
        
        if current_score < best_score:
            best_score = current_score
            best_adj = [row[:] for row in adj]
    
    if best_adj is None:
        # Fallback to seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    adj = best_adj
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges
