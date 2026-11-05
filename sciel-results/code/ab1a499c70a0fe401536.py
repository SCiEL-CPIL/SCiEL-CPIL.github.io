import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    def solve(n, time_limit):
        if time.time() > deadline:
            return None
            
        # adj[c][u] is an integer bitmask of neighbors of u in color c
        adj = [[0] * n for _ in range(4)]
        
        # Initialize random
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                
        def count_paths_len5(u, v, c):
            if u == v: return 0
            count = 0
            def dfs(curr, depth, visited_mask):
                nonlocal count
                if depth == 4:
                    if (adj[c][curr] >> v) & 1:
                        count += 1
                    return
                mask = adj[c][curr]
                while mask:
                    lsb = mask & -mask
                    nxt = lsb.bit_length() - 1
                    mask -= lsb
                    if nxt == u: continue
                    if nxt == v: continue
                    if (visited_mask >> nxt) & 1: continue
                    dfs(nxt, depth + 1, visited_mask | (1 << nxt))
            dfs(u, 0, 1 << u)
            return count

        def total_c6():
            total = 0
            for c in range(4):
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[c][i] >> j) & 1:
                            total += count_paths_len5(i, j, c)
            return total

        current_cost = total_c6()
        
        if current_cost == 0:
            return extract_colors(adj, n)

        T = 10.0
        T_min = 0.01
        cooling_rate = 0.99999
        
        while time.time() < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            if i > j: i, j = j, i
            
            c_old = -1
            for c in range(4):
                if (adj[c][i] >> j) & 1:
                    c_old = c
                    break
            
            c_new = random.choice([c for c in range(4) if c != c_old])
            
            # Calculate delta
            # Remove edge from c_old, add to c_new
            # Delta = (new C6s in c_new using i,j) - (old C6s in c_old using i,j)
            
            # To count C6s using edge (i,j) in color c, we count paths of length 5 from i to j in G_c \ {(i,j)}
            # We can just call count_paths_len5(i, j, c) but we must ensure the path doesn't use the direct edge.
            # My dfs above skips `nxt == v` at intermediate steps, but does it skip the direct edge at the end?
            # At depth 4, it checks if `v` is in `adj[curr]`. If `curr` is a neighbor of `i` via a path of length 4,
            # and `curr` is connected to `v`, that's a path of length 5.
            # Does this path use the direct edge `(i,v)`?
            # The path is $i \to x_1 \to x_2 \to x_3 \to x_4 \to v$.
            # If $x_1 = v$, it's skipped.
            # If $x_4 = i$, it's skipped (nxt == u).
            # If the path uses the direct edge $(i,v)$ as one of the intermediate edges, it would mean $x_k = v$ for $k < 4$, which is skipped.
            # Or $x_k = i$ for $k > 0$, which is skipped.
            # So `count_paths_len5` correctly counts paths that do NOT use the direct edge $(i,v)$?
            # Wait. If the direct edge $(i,v)$ exists, it is in `adj`.
            # But the path is simple.
            # If the path was $i \to v \to \dots$, it would be caught by `nxt == v` at depth 0? No, depth 0 is `i`.
            # Neighbors of `i` include `v`.
            # If we pick `v` as `x_1`, then `nxt == v`.
            # My code: `if nxt == v: continue`.
            # So yes, it excludes paths that go $i \to v$ immediately.
            # What about paths that come back to $v$ at the end? That's the goal.
            # So `count_paths_len5(i, j, c)` gives the number of C6s in color $c$ that use edge $(i,j)$.
            
            # However, we need to be careful.
            # When we change color, the graph changes.
            # `delta = count_paths_len5(i, j, c_new) - count_paths_len5(i, j, c_old)`
            # But wait.
            # In `c_old`, the edge $(i,j)$ is present.
            # `count_paths_len5` counts paths of length 5.
            # The C6 formed is $Path + Edge$.
            # So yes, the number of C6s destroyed is `count_paths_len5(i, j, c_old)`.
            # The number of C6s created is `count_paths_len5(i, j, c_new)`.
            
            delta = count_paths_len5(i, j, c_new) - count_paths_len5(i, j, c_old)
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                # Update adj
                adj[c_old][i] &= ~(1 << j)
                adj[c_old][j] &= ~(1 << i)
                adj[c_new][i] |= (1 << j)
                adj[c_new][j] |= (1 << i)
                current_cost += delta
                if current_cost == 0:
                    return extract_colors(adj, n)
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return None

    def extract_colors(adj, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                for c in range(4):
                    if (adj[c][i] >> j) & 1:
                        colors.append(c)
                        break
        return {'n': n, 'colors': colors}

    # Try 18
    res = solve(18, time.time() + 60)
    if res:
        return res
        
    # Try 17
    res = solve(17, time.time() + 80)
    if res:
        return res
        
    # Try 16
    res = solve(16, time.time() + 10)
    if res:
        return res
        
    # Fallback
    return {'n': 1, 'colors': []}
