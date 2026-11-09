"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Union-Find for bicolored cycle detection
    class DSU:
        def __init__(self, size):
            self.parent = list(range(size))
            self.rank = [0] * size
        
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            return True
    
    def has_bicolored_cycle(coloring, c1, c2):
        """Check if subgraph induced by colors c1 and c2 has a cycle."""
        # Build subgraph with only vertices of color c1 or c2
        vertices = [v for v in range(n) if coloring[v] in (c1, c2)]
        if len(vertices) < 3:
            return False
        
        # Map to local indices
        vset = set(vertices)
        local_map = {v: i for i, v in enumerate(vertices)}
        local_n = len(vertices)
        
        # Build local adjacency
        local_adj = [[] for _ in range(local_n)]
        for v in vertices:
            for w in adj[v]:
                if w in vset:
                    local_adj[local_map[v]].append(local_map[w])
        
        # DFS to detect cycle
        visited = [False] * local_n
        parent = [-1] * local_n
        
        def dfs(node, par):
            visited[node] = True
            for neighbor in local_adj[node]:
                if not visited[neighbor]:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != par:
                    return True
            return False
        
        for i in range(local_n):
            if not visited[i]:
                if dfs(i, -1):
                    return True
        return False
    
    # Backtracking with constraint propagation
    def solve():
        # Try small number of colors first
        for num_colors in range(2, 10):
            coloring = [-1] * n
            
            def is_valid_assignment(v, c):
                # Check adjacent vertices
                for w in adj[v]:
                    if coloring[w] == c:
                        return False
                return True
            
            def check_bicolored(v, c):
                # After assigning color c to v, check all bicolored subgraphs
                for other_c in range(num_colors):
                    if other_c == c:
                        continue
                    # Check if adding v with color c creates a bicolored cycle with color other_c
                    # Only need to check if v has neighbors with color other_c
                    has_other = False
                    for w in adj[v]:
                        if coloring[w] == other_c:
                            has_other = True
                            break
                    if not has_other:
                        continue
                    # Check for cycle in subgraph of colors c and other_c
                    # Build the subgraph including v
                    vset = {v}
                    for w in range(n):
                        if coloring[w] in (c, other_c):
                            vset.add(w)
                    
                    if len(vset) < 3:
                        continue
                    
                    # Check if there's a cycle
                    # Use union-find on the subgraph
                    vlist = list(vset)
                    lmap = {v: i for i, v in enumerate(vlist)}
                    dsu = DSU(len(vlist))
                    
                    for v_ in vlist:
                        for w_ in adj[v_]:
                            if w_ in lmap and lmap[v_] < lmap[w_]:
                                if not dsu.union(lmap[v_], lmap[w_]):
                                    return False  # Cycle found
                    # No cycle
                return True
            
            def backtrack(idx):
                if idx == n:
                    return True
                
                # Try colors in order
                for c in range(num_colors):
                    if not is_valid_assignment(idx, c):
                        continue
                    coloring[idx] = c
                    if check_bicolored(idx, c):
                        if backtrack(idx + 1):
                            return True
                    coloring[idx] = -1
                
                return False
            
            if backtrack(0):
                return [int(c) for c in coloring]
        
        # Fallback: distinct colors
        return [int(i) for i in range(n)]
    
    result = solve()
    return result
# EVOLVE-BLOCK-END
