"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 50  # Safety margin before 90s kill
    edges = [(min(u,v), max(u,v)) for u,v in P['edges']]
    edge_set = set(edges)
    n = P['n']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_max_biclique(residual_edges, weights=None):
        """Find a large biclique in the residual graph.
        Returns (left, right) or (None, None) if no edges."""
        if not residual_edges:
            return None, None
        
        # Build residual adjacency
        res_adj = [set() for _ in range(n)]
        for u, v in residual_edges:
            res_adj[u].add(v)
            res_adj[v].add(u)
        
        # Try to find a large biclique using a greedy + local search approach
        best_left, best_right = None, None
        best_size = 0
        
        # Candidate left sets: start from individual vertices
        candidates_left = []
        for u in range(n):
            if res_adj[u]:
                candidates_left.append(frozenset([u]))
        
        # Also try pairs of vertices that are both connected to many common neighbors
        # For efficiency, only try a limited number
        high_degree = [u for u in range(n) if len(res_adj[u]) > 0]
        high_degree.sort(key=lambda x: -len(res_adj[x]))
        
        # Try top vertices and their common neighbors
        for u in high_degree[:10]:
            # Find common neighbors of u with other high-degree vertices
            for v in high_degree[:10]:
                if u == v:
                    continue
                common = res_adj[u] & res_adj[v]
                if len(common) >= 2:
                    candidates_left.append(frozenset([u, v]))
        
        # For each candidate left set, find the best right set
        for left_set in candidates_left:
            if not left_set:
                continue
            # Right set = intersection of neighbors of all left vertices
            right_set = set(res_adj[list(left_set)[0]])
            for u in left_set:
                right_set &= res_adj[u]
            
            if not right_set:
                continue
            
            # Check that all cross edges exist in residual
            valid = True
            for u in left_set:
                for v in right_set:
                    if (min(u,v), max(u,v)) not in set(residual_edges):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            size = len(left_set) * len(right_set)
            if size > best_size:
                best_size = size
                best_left = list(left_set)
                best_right = list(right_set)
        
        # If we found something, try to expand
        if best_left and best_right:
            # Try adding vertices to left
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    if u not in best_left:
                        # Check if u is connected to all right vertices
                        if all(v in res_adj[u] for v in best_right):
                            # Check all cross edges exist
                            if all((min(u,v), max(u,v)) in set(residual_edges) for v in best_right):
                                best_left.append(u)
                                changed = True
            
            # Try adding vertices to right
            changed = True
            while changed:
                changed = False
                for v in range(n):
                    if v not in best_right:
                        # Check if v is connected to all left vertices
                        if all(u in res_adj[v] for u in best_left):
                            # Check all cross edges exist
                            if all((min(u,v), max(u,v)) in set(residual_edges) for u in best_left):
                                best_right.append(v)
                                changed = True
        
        return best_left, best_right
    
    # Column generation / greedy decomposition
    residual = set(edges)
    bicliques = []
    
    while residual and time.time() < deadline - 5:
        left, right = find_max_biclique(list(residual))
        
        if left is None or right is None:
            # Fallback: take a single edge
            u, v = next(iter(residual))
            left = [u]
            right = [v]
        
        bicliques.append([left, right])
        
        # Remove all cross edges from residual
        for u in left:
            for v in right:
                e = (min(u,v), max(u,v))
                if e in residual:
                    residual.remove(e)
    
    # Handle any remaining edges
    if residual:
        for u, v in list(residual):
            bicliques.append([[u], [v]])
            residual.discard((u, v))
    
    # Convert to proper Python types
    result = []
    for b in bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
