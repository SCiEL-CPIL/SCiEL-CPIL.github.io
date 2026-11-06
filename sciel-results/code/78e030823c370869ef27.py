import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 18
    random.seed(42)
    
    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append([i, j])
        return edges
    
    def is_k4_free(adj):
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]:
                    continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def max_tf(adj):
        """Max triangle-free induced subgraph size via B&B."""
        tri = [[set() for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b]:
                    for c in range(n):
                        if c != a and c != b and adj[a][c] and adj[b][c]:
                            tri[a][b].add(c)
                            tri[b][a].add(c)
        
        best = 0
        def search(avail, chosen):
            nonlocal best
            if len(chosen) + len(avail) <= best:
                return
            if not avail:
                best = max(best, len(chosen))
                return
            for idx in range(len(avail)):
                v = avail[idx]
                compatible = True
                for a in chosen:
                    if adj[v][a]:
                        for b in chosen:
                            if b != a and adj[a][b] and v in tri[a][b]:
                                compatible = False
                                break
                    if not compatible:
                        break
                if not compatible:
                    continue
                new_avail = []
                for w in avail:
                    if w == v:
                        continue
                    ok = True
                    for a in chosen:
                        if adj[w][a]:
                            for b in chosen:
                                if b != a and adj[a][b] and w in tri[a][b]:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if ok and adj[w][v]:
                        for a in chosen:
                            if adj[v][a] and w in tri[v][a]:
                                ok = False
                                break
                    if ok:
                        new_avail.append(w)
                search(new_avail, chosen + [v])
        search(list(range(n)), [])
        return best
    
    # Initial: K6,6,6
    adj = [[False]*n for _ in range(n)]
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = True
                    adj[b][a] = True
    
    best_adj = [row[:] for row in adj]
    best_score = max_tf(adj)  # 12
    
    all_pairs = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    # Local search with simulated annealing
    T = 3.0
    cooling = 0.999
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline:
        if no_improve > max_no_improve:
            adj = [row[:] for row in best_adj]
            for _ in range(random.randint(5, 15)):
                u, v = random.choice(all_pairs)
                new_adj = [row[:] for row in adj]
                new_adj[u][v] = not adj[u][v]
                new_adj[v][u] = not adj[v][u]
                if is_k4_free(new_adj):
                    adj = new_adj
            no_improve = 0
            T = 3.0
            continue
        
        u, v = random.choice(all_pairs)
        new_adj = [row[:] for row in adj]
        new_adj[u][v] = not adj[u][v]
        new_adj[v][u] = not adj[v][u]
        
        if not is_k4_free(new_adj):
            no_improve += 1
            continue
        
        new_score = max_tf(new_adj)
        old_score = max_tf(adj)
        
        if new_score < old_score:
            adj = new_adj
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
            no_improve = 0
        else:
            delta = new_score - old_score
            if delta == 0 or random.random() < math.exp(-delta / max(T, 0.01)):
                adj = new_adj
            no_improve += 1
        
        T *= cooling
    
    return adj_to_edges(best_adj)
