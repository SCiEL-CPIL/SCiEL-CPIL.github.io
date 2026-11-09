import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute a good vertex ordering (degree descending)
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def solve(k):
        """
        Backtracking search for a valid coloring with k colors.
        Uses union-find with undo stack for O(alpha) cycle detection.
        """
        coloring = [-1] * n
        
        # Union-Find structures per color pair
        # uf_data[key] = (parent_dict, rank_dict)
        uf_data = {}
        # Undo stack: list of (key, u, v, parent_before, rank_before)
        undo_stack = []
        
        def get_uf(c1, c2):
            """Get or create union-find for color pair (c1, c2)."""
            key = (c1, c2) if c1 < c2 else (c2, c1)
            if key not in uf_data:
                uf_data[key] = ({}, {})
            return key, uf_data[key]
        
        def uf_find(parent, x):
            """Find root with path compression (but we don't compress since we need undo)."""
            if x not in parent:
                return x
            # Path compression not used for simplicity with undo
            while parent[x] != x:
                x = parent[x]
            return x
        
        def uf_union(parent, rank, u, v):
            """
            Union u and v. Returns True if they were in different components (no cycle).
            Returns False if they were already connected (cycle would form).
            """
            ru = uf_find(parent, u)
            rv = uf_find(parent, v)
            if ru == rv:
                return False  # Cycle would form
            
            # Union by rank
            if rank.get(ru, 0) < rank.get(rv, 0):
                ru, rv = rv, ru
            
            # Record state for undo
            undo_stack.append((u, v, ru, rv, parent.get(ru, ru), parent.get(rv, rv), rank.get(ru, 0), rank.get(rv, 0)))
            
            parent[rv] = ru
            if rank.get(ru, 0) == rank.get(rv, 0):
                rank[ru] = rank.get(ru, 0) + 1
            return True
        
        def uf_undo():
            """Undo the last union operation."""
            if not undo_stack:
                return False
            u, v, ru, rv, parent_ru, parent_rv, rank_ru, rank_rv = undo_stack.pop()
            
            # Restore parent pointers
            if parent_ru != ru:
                parent[ru] = parent_ru
            if parent_rv != rv:
                parent[rv] = parent_rv
            
            # Restore ranks
            if rank_ru != 0:
                rank[ru] = rank_ru
            else:
                rank.pop(ru, None)
            if rank_rv != 0:
                rank[rv] = rank_rv
            else:
                rank.pop(rv, None)
            return True
        
        def can_assign(v, c):
            """
            Check if vertex v can be assigned color c.
            1. No neighbor has color c.
            2. For every neighbor u with color d != c, the pair (c, d) forest must not form a cycle.
            """
            # Check properness
            for u in adj[v]:
                if coloring[u] == c:
                    return False
            
            # Check bicolored cycles
            # Group neighbors by their color
            neighbors_by_color = defaultdict(list)
            for u in adj[v]:
                if coloring[u] != -1:
                    neighbors_by_color[coloring[u]].append(u)
            
            for d, neighbors in neighbors_by_color.items():
                if d == c:
                    continue
                
                key, (parent, rank) = get_uf(c, d)
                
                # If there are 2 or more neighbors with color d, check if any pair is connected
                if len(neighbors) >= 2:
                    # Check all pairs of neighbors in this group
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            if uf_find(parent, neighbors[i]) == uf_find(parent, neighbors[j]):
                                return False
            return True
        
        def assign(v, c):
            """Assign color c to v and update forests."""
            coloring[v] = c
            # Add edges to forests for pairs (c, d) where d is a neighbor's color
            for u in adj[v]:
                if coloring[u] != -1 and coloring[u] != c:
                    d = coloring[u]
                    key, (parent, rank) = get_uf(c, d)
                    # This union should always succeed if can_assign passed
                    uf_union(parent, rank, v, u)
        
        def unassign(v, c):
            """Revert assignment of color c to v."""
            coloring[v] = -1
            # Remove edges from forests by undoing unions
            for u in adj[v]:
                if coloring[u] != -1 and coloring[u] != c:
                    d = coloring[u]
                    key, (parent, rank) = get_uf(c, d)
                    uf_undo()
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                return coloring[:]
            
            v = order[idx]
            
            for c in range(k):
                if can_assign(v, c):
                    assign(v, c)
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    unassign(v, c)
            return None
        
        return backtrack(0)

    # Try increasing k
    # The graph has triangles? 
    # Let's check if 3 colors are enough.
    # If not, try 4, etc.
    
    # Heuristic: Start from 3.
    for k in range(3, 11):
        if time.time() > deadline:
            break
        res = solve(k)
        if res is not None:
            # Ensure types are correct
            return [int(x) for x in res]
            
    # Fallback: return trivial solution (50 colors)
    return list(range(n))

# EVOLVE-BLOCK-END
