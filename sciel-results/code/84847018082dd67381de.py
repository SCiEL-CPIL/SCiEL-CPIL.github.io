import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n = 18
    
    def edges_from_adj(adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i][j]]
    
    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not adj[a][b]:
                    continue
                for c in range(b + 1, n):
                    if not adj[a][c] or not adj[b][c]:
                        continue
                    for d in range(c + 1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return True
        return False
    
    def greedy_max_tf(adj):
        best = 0
        for _ in range(30):
            order = list(range(n))
            random.shuffle(order)
            current = []
            for v in order:
                can_add = True
                for i in range(len(current)):
                    for j in range(i + 1, len(current)):
                        u, w = current[i], current[j]
                        if adj[u][v] and adj[w][v] and adj[u][w]:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    current.append(v)
            if len(current) > best:
                best = len(current)
        return best
    
    def exact_max_tf(adj):
        for size in range(n, 0, -1):
            for subset in itertools.combinations(range(n), size):
                s = len(subset)
                is_tf = True
                for i in range(s):
                    for j in range(i + 1, s):
                        for k in range(j + 1, s):
                            u, v, w = subset[i], subset[j], subset[k]
                            if adj[u][v] and adj[v][w] and adj[u][w]:
                                is_tf = False
                                break
                        if not is_tf:
                            break
                    if not is_tf:
                        break
                if is_tf:
                    return size
        return 0
    
    def random_k4_free_adj():
        adj = [[False] * n for _ in range(n)]
        p = 0.4
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = adj[j][i] = True
        changed = True
        while changed:
            changed = False
            for a in range(n):
                if changed:
                    break
                for b in range(a + 1, n):
                    if changed:
                        break
                    if not adj[a][b]:
                        continue
                    for c in range(b + 1, n):
                        if changed:
                            break
                        if not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c + 1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                adj[c][d] = adj[d][c] = False
                                changed = True
                                break
        return adj
    
    def local_search(adj, max_iters=5000):
        best_score = greedy_max_tf(adj)
        best_adj = [row[:] for row in adj]
        current_adj = [row[:] for row in adj]
        no_improve = 0
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            old_val = current_adj[i][j]
            current_adj[i][j] = current_adj[j][i] = not old_val
            if has_k4(current_adj):
                current_adj[i][j] = current_adj[j][i] = old_val
                continue
            new_score = greedy_max_tf(current_adj)
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in current_adj]
                no_improve = 0
            else:
                current_adj[i][j] = current_adj[j][i] = old_val
                no_improve += 1
            if no_improve > 300:
                current_adj = random_k4_free_adj()
                no_improve = 0
        return best_adj, best_score
    
    best_adj = None
    best_exact = 12
    
    # Baseline
    adj = [[False] * n for _ in range(n)]
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = adj[b][a] = True
    s = exact_max_tf(adj)
    if s < best_exact:
        best_exact = s
        best_adj = [row[:] for row in adj]
    
    candidates = []
    for attempt in range(60):
        if time.time() > deadline - 30:
            break
        adj = random_k4_free_adj()
        adj, gs = local_search(adj, max_iters=4000)
        candidates.append((gs, adj))
        if gs <= 7:
            break
    
    candidates.sort(key=lambda x: x[0])
    for gs, adj in candidates[:15]:
        if time.time() > deadline - 5:
            break
        s = exact_max_tf(adj)
        if s < best_exact:
            best_exact = s
            best_adj = [row[:] for row in adj]
    
    if best_adj is None:
        adj = [[False] * n for _ in range(n)]
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a][b] = adj[b][a] = True
        best_adj = adj
    
    return edges_from_adj(best_adj)
