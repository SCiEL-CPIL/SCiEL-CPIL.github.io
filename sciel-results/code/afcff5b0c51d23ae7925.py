import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_strong_conflict_fast(edges, adj, n):
        n_edges = len(edges)
        vert_to_edges = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
            
        strong_adj = [0] * n_edges
        
        for i, (u, v) in enumerate(edges):
            mask = 0
            # Type 1: Share vertex
            for j in vert_to_edges[u]:
                mask |= (1 << j)
            for j in vert_to_edges[v]:
                mask |= (1 << j)
                
            # Type 2: Connected by edge
            for x in adj[u]:
                for j in vert_to_edges[x]:
                    mask |= (1 << j)
            for x in adj[v]:
                for j in vert_to_edges[x]:
                    mask |= (1 << j)
                    
            strong_adj[i] = mask
        return strong_adj

    def solve_coloring(n_edges, strong_adj, max_colors, time_limit):
        # Order vertices by degree descending
        degrees = [bin(strong_adj[i]).count('1') for i in range(n_edges)]
        order = sorted(range(n_edges), key=lambda i: degrees[i], reverse=True)
        
        color = [-1] * n_edges
        available = [(1 << max_colors) - 1] * n_edges
        
        # Forward checking with backtracking
        def backtrack(idx):
            if time.time() > time_limit:
                return False
            if idx == n_edges:
                return True
            
            i = order[idx]
            avail = available[i]
            if avail == 0:
                return False
            
            while avail:
                c_bit = avail & (-avail)
                c = c_bit.bit_length() - 1
                avail -= c_bit
                
                color[i] = c
                neighbors_mask = strong_adj[i]
                valid = True
                affected = []
                n_mask = neighbors_mask
                while n_mask:
                    nb_bit = n_mask & (-n_mask)
                    nb = nb_bit.bit_length() - 1
                    n_mask -= nb_bit
                    
                    if color[nb] == -1:
                        available[nb] &= ~c_bit
                        if available[nb] == 0:
                            valid = False
                            break
                        affected.append(nb)
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                for nb in affected:
                    available[nb] |= c_bit
                color[i] = -1
                
            return False
        
        if backtrack(0):
            return color
        return None

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        n_edges = len(edges)
        if n_edges == 0:
            out.append([])
            continue
            
        strong_adj = build_strong_conflict_fast(edges, adj, n)
        
        max_k = int(5 * d * d / 4)
        
        best_colors = None
        # Try from max_k down
        for k in range(max_k, 0, -1):
            if time.time() > deadline - 10:
                break
            colors = solve_coloring(n_edges, strong_adj, k, deadline - 5)
            if colors is not None:
                best_colors = colors
                break
        
        if best_colors is None:
            # Fallback: Greedy
            # This shouldn't happen if conjecture holds
            best_colors = [0] * n_edges
            for i in range(n_edges):
                # Find first available color
                used = set()
                mask = strong_adj[i]
                while mask:
                    nb_bit = mask & (-mask)
                    nb = nb_bit.bit_length() - 1
                    mask -= nb_bit
                    if best_colors[nb] != -1:
                        used.add(best_colors[nb])
                c = 0
                while c in used:
                    c += 1
                best_colors[i] = c
        
        out.append(best_colors)

    return out
