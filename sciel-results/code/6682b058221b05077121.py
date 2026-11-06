import time
import random

def construct():
    start_time = time.time()
    hard_deadline = start_time + 110.0
    rng = random.Random(42)

    def verify(n, colors):
        """Verify that no color class contains a C6."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        for c in range(4):
            adj_list = [[] for _ in range(n)]
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    adj_list[i].append(j)
                    mask -= lsb
            
            for u in range(n):
                for v in adj_list[u]:
                    if v <= u: continue
                    for w in adj_list[v]:
                        if w == u: continue
                        for x in adj_list[w]:
                            if x == u or x == v: continue
                            for y in adj_list[x]:
                                if y == u or y == v or y == w: continue
                                for z in adj_list[y]:
                                    if z == u or z == v or z == w or z == x: continue
                                    if (adj[c][z] >> u) & 1:
                                        return False
        return True

    def edge_idx(n, i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def build_adj(n, colors):
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    def has_path_len_5(u, v, adj_c, n):
        """Check if there's a simple path of length 5 from u to v in graph adj_c."""
        visited = 1 << u
        frontier = [u]
        for depth in range(4):
            next_frontier = []
            for node in frontier:
                mask = adj_c[node] & ~visited
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    if nxt == v and depth == 3:
                        return True
                    if nxt != v:
                        next_frontier.append(nxt)
                        visited |= lsb
                    mask -= lsb
            frontier = next_frontier
            if not frontier:
                break
        for node in frontier:
            if (adj_c[node] >> v) & 1:
                return True
        return False

    def greedy_construct(n, seed=0):
        """Greedy construction: assign edges in random order, pick a color that doesn't create C6."""
        r = random.Random(seed)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        r.shuffle(edges)
        colors = [-1] * len(edges)
        adj = [[0] * n for _ in range(4)]
        
        for ei, (u, v) in enumerate(edges):
            placed = False
            color_order = list(range(4))
            r.shuffle(color_order)
            for c in color_order:
                if has_path_len_5(u, v, adj[c], n):
                    continue
                colors[ei] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
                placed = True
                break
            
            if not placed:
                return None
        
        result = [0] * len(edges)
        for ei, (u, v) in enumerate(edges):
            li = edge_idx(n, u, v)
            result[li] = colors[ei]
        return result

    def local_search(n, colors, max_iters=100000):
        """Local search: try flipping edge colors, using incremental C6 detection."""
        num_edges = n * (n - 1) // 2
        colors = colors[:]
        adj = build_adj(n, colors)
        edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        def count_total_c6s():
            total = 0
            for c in range(4):
                adj_c = adj[c]
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj_c[i] >> j) & 1:
                            adj_c[i] &= ~(1 << j)
                            adj_c[j] &= ~(1 << i)
                            if has_path_len_5(i, j, adj_c, n):
                                total += 1
                            adj_c[i] |= (1 << j)
                            adj_c[j] |= (1 << i)
            return total
        
        current_total = count_total_c6s()
        best_total = current_total
        best_colors = colors[:]
        
        for it in range(max_iters):
            if time.time() > hard_deadline:
                break
            
            ei = rng.randint(0, num_edges - 1)
            u, v = edge_list[ei]
            old_c = colors[ei]
            new_c = rng.randint(0, 3)
            if new_c == old_c:
                continue
            
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            removed = 1 if has_path_len_5(u, v, adj[old_c], n) else 0
            adj[old_c][u] |= (1 << v)
            adj[old_c][v] |= (1 << u)
            
            added = 1 if has_path_len_5(u, v, adj[new_c], n) else 0
            
            delta = added - removed
            new_total = current_total + delta
            
            if new_total <= best_total:
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                colors[ei] = new_c
                current_total = new_total
                best_total = new_total
                best_colors = colors[:]
            else:
                if delta > 2 or rng.random() > 0.03:
                    pass
                else:
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    colors[ei] = new_c
                    current_total = new_total
            
            if best_total == 0:
                return best_colors
        
        return best_colors

    def try_n(n, time_budget):
        """Try to find a valid coloring for n vertices within time_budget."""
        deadline = time.time() + time_budget
        
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            colors = greedy_construct(n, seed=attempt * 7 + 13)
            if colors is None:
                continue
            
            if verify(n, colors):
                return colors
            
            colors = local_search(n, colors, max_iters=30000)
            
            if verify(n, colors):
                return colors
        
        return None

    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > hard_deadline - 10:
            break
        
        remaining = hard_deadline - time.time()
        if n == 18:
            budget = min(remaining - 5, 12.0)
        elif n == 17:
            budget = min(remaining - 5, 35.0)
        else:
            budget = min(remaining - 5, 10.0)
        if budget < 2:
            break
        
        result = try_n(n, budget)
        if result is not None:
            if verify(n, result):
                return {'n': n, 'colors': result}
    
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i + j) % 4)
    if verify(n, colors):
        return {'n': n, 'colors': colors}
    
    return {'n': 1, 'colors': []}

if __name__ == '__main__':
    construct()
