"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Union-Find for cycle detection
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True

    def solve_with_result(K):
        coloring = [-1] * n
        dsus = {}
        for i in range(K):
            for j in range(i+1, K):
                dsus[(i, j)] = DSU(n)
        
        def backtrack(idx):
            if idx == n:
                return True
            
            for c in range(K):
                valid = True
                for neighbor in adj[idx]:
                    if coloring[neighbor] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                saved_dsu_states = {}
                for key, dsu in dsus.items():
                    saved_dsu_states[key] = (dsu.parent[:], dsu.rank[:])
                
                coloring[idx] = c
                
                conflict = False
                for c_prime in range(K):
                    if c_prime == c:
                        continue
                    key = (min(c, c_prime), max(c, c_prime))
                    dsu = dsus[key]
                    for neighbor in adj[idx]:
                        if coloring[neighbor] == c_prime:
                            if not dsu.union(idx, neighbor):
                                conflict = True
                                break
                    if conflict:
                        break
                
                if conflict:
                    coloring[idx] = -1
                    for key, (p, r) in saved_dsu_states.items():
                        dsus[key].parent = p
                        dsus[key].rank = r
                    continue
                
                if backtrack(idx + 1):
                    return True
                
                coloring[idx] = -1
                for key, (p, r) in saved_dsu_states.items():
                    dsus[key].parent = p
                    dsus[key].rank = r
            
            return False
        
        if backtrack(0):
            return [int(c) for c in coloring]
        return None

    for K in range(3, 6):
        res = solve_with_result(K)
        if res is not None:
            return res
            
    # Fallback
    return [int(i % 50) for i in range(50)]
# EVOLVE-BLOCK-END
