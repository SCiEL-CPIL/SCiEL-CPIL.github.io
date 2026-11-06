import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def max_triangle_free_size(adj):
        """Exact branch and bound for largest triangle-free induced subgraph."""
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        def backtrack(idx, current_mask, current_size):
            if time.time() > deadline - 5:
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - idx) <= best[0]:
                return
            v = order[idx]
            # Try excluding v
            backtrack(idx + 1, current_mask, current_size)
            # Try including v if no triangle formed
            neighbors_in_S = current_mask & adj[v]
            has_edge = False
            tmp = neighbors_in_S
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    has_edge = True
                    break
                tmp = rest
            if not has_edge:
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)

        backtrack(0, 0, 0)
        return best[0]

    # Build K_{6,6,6}: complete tripartite with parts of size 6
    adj = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
    
    assert not has_k4(adj), "K_{6,6,6} should be K4-free"
    score = max_triangle_free_size(adj)
    
    # Try to improve with local search: flip edges to reduce max triangle-free size
    best_adj = adj[:]
    best_score = score
    
    for iteration in range(2000):
        if time.time() > deadline - 10:
            break
        if best_score <= 8:
            break
        
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        old_adj = best_adj[:]
        if (best_adj[i] >> j) & 1:
            best_adj[i] &= ~(1 << j)
            best_adj[j] &= ~(1 << i)
        else:
            best_adj[i] |= (1 << j)
            best_adj[j] |= (1 << i)
        
        if has_k4(best_adj):
            best_adj = old_adj
            continue
        
        score = max_triangle_free_size(best_adj)
        if score < best_score:
            best_score = score
        else:
            best_adj = old_adj
    
    return edges_from_adj(best_adj)
