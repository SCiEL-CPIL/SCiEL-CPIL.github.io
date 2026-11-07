"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    eu = [e[0] for e in edges]
    ev = [e[1] for e in edges]
    
    # We want to assign colors 0, 1, 2 to edges such that:
    # 1. Each vertex has exactly 2 edges of each color.
    # 2. Each color class forms a single cycle of length n.
    
    # State:
    # color: list of length m, -1 if unassigned
    # deg: list of n lists of length 3
    # parent, size: for Union-Find per color
    
    # To make it fast, we use a simple backtracking with copy-on-write.
    
    def solve():
        init_color = [-1] * m
        init_deg = [[0]*3 for _ in range(n)]
        init_parent = [[v for v in range(n)] for _ in range(3)]
        init_size = [[1]*n for _ in range(3)]
        
        # Symmetry breaking:
        # Fix edge 0 (0,1) to color 0
        # Fix edge 1 (0,2) to color 1
        # Fix edge 2 (0,5) to color 2
        # This ensures vertex 0 is connected to 1, 2, 5 with distinct colors.
        
        def assign(idx, c, color, deg, parent, size):
            u, v = eu[idx], ev[idx]
            color[idx] = c
            deg[u][c] += 1
            deg[v][c] += 1
            ru = u
            while parent[c][ru] != ru:
                ru = parent[c][ru]
            rv = v
            while parent[c][rv] != rv:
                rv = parent[c][rv]
            if ru != rv:
                if size[c][ru] < size[c][rv]:
                    ru, rv = rv, ru
                parent[c][rv] = ru
                size[c][ru] += size[c][rv]
            return ru, rv

        # Apply initial fixes
        assign(0, 0, init_color, init_deg, init_parent, init_size)
        assign(1, 1, init_color, init_deg, init_parent, init_size)
        assign(2, 2, init_color, init_deg, init_parent, init_size)
        
        def backtrack(idx, color, deg, parent, size):
            if idx == m:
                # Verify
                for v in range(n):
                    for c in range(3):
                        if deg[v][c] != 2:
                            return None
                for c in range(3):
                    roots = set()
                    for v in range(n):
                        r = v
                        while parent[c][r] != r:
                            r = parent[c][r]
                        roots.add(r)
                    if len(roots) != 1 or size[c][list(roots)[0]] != n:
                        return None
                return color
                
            u, v = eu[idx], ev[idx]
            
            # Try colors
            for c in range(3):
                if deg[u][c] >= 2 or deg[v][c] >= 2:
                    continue
                
                # Check cycle formation
                ru = u
                while parent[c][ru] != ru:
                    ru = parent[c][ru]
                rv = v
                while parent[c][rv] != rv:
                    rv = parent[c][rv]
                    
                if ru == rv and size[c][ru] != n:
                    continue
                
                # Copy state
                new_color = color[:]
                new_deg = [row[:] for row in deg]
                new_parent = [row[:] for row in parent]
                new_size = [row[:] for row in size]
                
                # Apply assignment
                assign(idx, c, new_color, new_deg, new_parent, new_size)
                
                res = backtrack(idx + 1, new_color, new_deg, new_parent, new_size)
                if res is not None:
                    return res
                    
            return None
            
        return backtrack(3, init_color, init_deg, init_parent, init_size)

    colors = solve()
    
    if colors is None:
        # Fallback: try without symmetry breaking?
        # Or just return a single cycle.
        def find_one_cycle():
            adj_set = {v: set() for v in range(n)}
            for u, v in edges:
                adj_set[u].add(v)
                adj_set[v].add(u)
                
            def dfs(path, visited):
                if len(path) == n:
                    if path[-1] in adj_set[path[0]]:
                        return path
                    return None
                last = path[-1]
                for nb in sorted(adj_set[last]):
                    if nb not in visited:
                        res = dfs(path + [nb], visited | {nb})
                        if res:
                            return res
                return None
                
            return dfs([0], {0})
            
        cycle = find_one_cycle()
        if cycle:
            return [cycle]
        return []

    # Extract cycles
    final_cycles = []
    for c in range(3):
        edges_c = [(eu[i], ev[i]) for i in range(m) if colors[i] == c]
        if not edges_c:
            continue
            
        adj_c = {v: [] for v in range(n)}
        for u, v in edges_c:
            adj_c[u].append(v)
            adj_c[v].append(u)
            
        # Reconstruct cycle
        path = [0]
        prev = -1
        curr = 0
        for _ in range(n - 1):
            nexts = [x for x in adj_c[curr] if x != prev]
            if not nexts:
                break
            curr = nexts[0]
            path.append(curr)
            prev = path[-2]
            
        if len(path) == n and (0 in adj_c[path[-1]]):
            final_cycles.append(path)
            
    return final_cycles

# EVOLVE-BLOCK-END
