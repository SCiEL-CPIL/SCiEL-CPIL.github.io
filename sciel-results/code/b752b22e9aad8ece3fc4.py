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
    
    # Union-Find class
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
        
        def copy(self):
            new_dsu = DSU(len(self.parent))
            new_dsu.parent = self.parent[:]
            new_dsu.rank = self.rank[:]
            return new_dsu

    def solve():
        # Try 3 colors
        coloring = [-1] * n
        
        # We will maintain DSUs for each pair of colors.
        # Since we only have 3 colors, pairs are (0,1), (0,2), (1,2).
        # However, the DSU only contains vertices that are actually colored with one of the two.
        # We can just use a single DSU per pair, and add vertices as they are colored.
        # But we need to undo. So we copy the DSU before each recursive step.
        
        def backtrack(idx, dsus):
            if idx == n:
                return True
            
            # Try colors 0, 1, 2
            for c in range(3):
                # Check if c is valid for idx (neighbors don't have c)
                valid = True
                for w in adj[idx]:
                    if coloring[w] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check bicolored cycles
                # We need to check pairs (c, c') for c' != c
                # We simulate adding idx to the subgraphs of (c, c')
                
                # We'll create temporary copies of the DSUs for the pairs involving c
                # Pairs: (0,1), (0,2), (1,2)
                # If c=0, pairs are (0,1) and (0,2).
                # If c=1, pairs are (0,1) and (1,2).
                # If c=2, pairs are (0,2) and (1,2).
                
                new_dsus = {}
                for key in dsus:
                    new_dsus[key] = dsus[key].copy()
                
                ok = True
                for c_prime in range(3):
                    if c_prime == c:
                        continue
                    key = (min(c, c_prime), max(c, c_prime))
                    dsu = new_dsus[key]
                    
                    # Add idx to this DSU
                    # Union idx with all neighbors that have color c or c_prime
                    for w in adj[idx]:
                        if coloring[w] == c or coloring[w] == c_prime:
                            if not dsu.union(idx, w):
                                ok = False
                                break
                    if not ok:
                        break
                
                if not ok:
                    continue
                
                # If ok, recurse
                coloring[idx] = c
                if backtrack(idx + 1, new_dsus):
                    return True
                coloring[idx] = -1
                
            return False
        
        # Initialize DSUs
        # Keys: (0,1), (0,2), (1,2)
        init_dsus = {}
        for c1 in range(3):
            for c2 in range(c1 + 1, 3):
                init_dsus[(c1, c2)] = DSU(n)
        
        if backtrack(0, init_dsus):
            return [int(c) for c in coloring]
        
        # Fallback to 4 colors if 3 fails (unlikely)
        # Just return a valid 4-coloring or distinct
        return [int(i) for i in range(n)]

    result = solve()
    return result
# EVOLVE-BLOCK-END
