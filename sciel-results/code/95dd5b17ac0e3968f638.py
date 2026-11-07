"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 85.0
    
    def solve_k(k):
        """Backtracking with forward checking and incremental bicolored cycle detection."""
        color = [-1] * n
        
        # For each color pair (a,b), maintain a DSU with rollback
        # pair_info[(a,b)] = (parent, rank, undo_stack)
        # We'll use a simple approach: for each pair, keep a list of union operations for rollback
        
        class RollbackDSU:
            def __init__(self):
                self.parent = {}
                self.rank = {}
                self.undo_log = []  # list of (child, parent_node, rank_change)
            
            def find(self, x):
                if x not in self.parent:
                    self.parent[x] = x
                    self.rank[x] = 0
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]
            
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return False  # cycle
                if self.rank[rx] < self.rank[ry]:
                    rx, ry = ry, rx
                self.undo_log.append((ry, rx, self.rank[rx]))
                self.parent[ry] = rx
                if self.rank[rx] == self.rank[ry]:
                    self.rank[rx] += 1
                return True
            
            def rollback(self, count):
                for _ in range(count):
                    child, par, old_rank = self.undo_log.pop()
                    self.parent[child] = child
                    self.rank[par] = old_rank
        
        pair_dsu = {}
        
        def get_dsu(a, b):
            key = (min(a, b), max(a, b))
            if key not in pair_dsu:
                pair_dsu[key] = RollbackDSU()
            return pair_dsu[key]
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            if idx == n:
                return color[:]
            
            for c in range(k):
                # Check adjacency
                valid = True
                for w in adj[idx]:
                    if color[w] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Try assigning color c to idx
                color[idx] = c
                unions_done = []  # list of (dsu_key, count) for rollback
                
                valid = True
                for w in adj[idx]:
                    cw = color[w]
                    if cw == -1:
                        continue
                    if cw == c:
                        valid = False
                        break
                    a, b = min(c, cw), max(c, cw)
                    dsu = get_dsu(a, b)
                    result = dsu.union(idx, w)
                    unions_done.append((dsu, 1))
                    if not result:
                        valid = False
                        break
                
                if not valid:
                    # Rollback
                    for dsu, cnt in reversed(unions_done):
                        dsu.rollback(cnt)
                    color[idx] = -1
                    continue
                
                result = backtrack(idx + 1)
                if result is not None:
                    return result
                
                # Rollback
                for dsu, cnt in reversed(unions_done):
                    dsu.rollback(cnt)
                color[idx] = -1
            
            return None
        
        return backtrack(0)
    
    for k in [3, 4, 5, 6, 7, 8]:
        result = solve_k(k)
        if result is not None:
            # Final verification
            ok = True
            for u, v in edges:
                if result[u] == result[v]:
                    ok = False
                    break
            if ok:
                for a in range(k):
                    for b in range(a + 1, k):
                        # Check subgraph induced by colors a and b is a forest
                        visited = set()
                        has_cycle = False
                        for start in range(n):
                            if result[start] in (a, b) and start not in visited:
                                stack = [(start, -1)]
                                while stack:
                                    node, parent = stack.pop()
                                    if node in visited:
                                        continue
                                    visited.add(node)
                                    for nb in adj[node]:
                                        if result[nb] not in (a, b):
                                            continue
                                        if nb == parent:
                                            continue
                                        if nb in visited:
                                            has_cycle = True
                                            break
                                        stack.append((nb, node))
                                    if has_cycle:
                                        break
                            if has_cycle:
                                break
                        if has_cycle:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                return [int(x) for x in result]
    
    # Fallback
    return [int(i) for i in range(n)]

# EVOLVE-BLOCK-END
