"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    
    # Build adjacency as sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
        edge_set.add((min(a,b), max(a,b)))
    
    # Track remaining edges
    remaining = set(edge_set)
    
    bicliques = []
    
    def try_grow_biclique(left, right, remaining_edges):
        """Try to grow a biclique by adding vertices with high overlap."""
        left = set(left)
        right = set(right)
        
        # Compute common neighbors
        common = set.intersection(*[adj[v] for v in left]) if left else set()
        common = common & set.intersection(*[adj[v] for v in right]) if right else set()
        
        # Candidates to add to left: vertices in common that are not in right
        # Candidates to add to right: vertices in common that are not in left
        # But we need all cross edges to exist
        
        # Better approach: iteratively add vertices that connect to all on the other side
        improved = True
        while improved:
            improved = False
            # Try adding to left
            candidates_left = set()
            for v in common:
                if v not in left and v not in right:
                    # Check if v connects to all in right
                    if right.issubset(adj[v]):
                        candidates_left.add(v)
            
            # Try adding to right
            candidates_right = set()
            for v in common:
                if v not in left and v not in right:
                    # Check if v connects to all in left
                    if left.issubset(adj[v]):
                        candidates_right.add(v)
            
            # Pick the side that gives more edges
            # Adding a vertex to left covers |right| edges
            # Adding a vertex to right covers |left| edges
            best_gain = 0
            best_action = None
            best_vertex = None
            
            if candidates_left:
                # Pick vertex that maximizes edges covered
                for v in candidates_left:
                    gain = len(right)
                    if gain > best_gain:
                        best_gain = gain
                        best_action = 'left'
                        best_vertex = v
            
            if candidates_right:
                for v in candidates_right:
                    gain = len(left)
                    if gain > best_gain:
                        best_gain = gain
                        best_action = 'right'
                        best_vertex = v
            
            if best_action and best_vertex is not None:
                if best_action == 'left':
                    left.add(best_vertex)
                else:
                    right.add(best_vertex)
                improved = True
                # Update common
                common = set.intersection(*[adj[v] for v in left]) if left else set()
                common = common & set.intersection(*[adj[v] for v in right]) if right else set()
        
        return left, right
    
    # Greedy: repeatedly find the largest biclique we can form from remaining edges
    while remaining:
        if time.time() > deadline:
            break
        
        # Find the best edge to start from (one that's in the densest area)
        # Score each remaining edge by potential biclique size
        best_edge = None
        best_score = -1
        best_left = None
        best_right = None
        
        for edge in remaining:
            a, b = edge
            # Common neighbors of a and b
            common = adj[a] & adj[b]
            # Potential left side: {a} + some from common
            # Potential right side: {b} + some from common
            # Simple score: |common|
            score = len(common)
            if score > best_score:
                best_score = score
                best_edge = edge
                best_left = {a}
                best_right = {b}
        
        if best_edge is None:
            break
        
        a, b = best_edge
        left, right = try_grow_biclique({a}, {b}, remaining)
        
        # Compute which edges are covered
        covered = set()
        for u in left:
            for v in right:
                e = (min(u,v), max(u,v))
                if e in remaining:
                    covered.add(e)
        
        if not covered:
            # Fallback: just take this edge as a biclique
            covered = {best_edge}
            left = {a}
            right = {b}
        
        remaining -= covered
        bicliques.append([sorted(left), sorted(right)])
    
    # If we ran out of time, handle remaining edges as single-edge bicliques
    if remaining:
        for a, b in remaining:
            bicliques.append([[a], [b]])
    
    # Convert to proper Python types
    result = []
    for left, right in bicliques:
        left = [int(x) for x in left]
        right = [int(x) for x in right]
        result.append([left, right])
    
    return result
# EVOLVE-BLOCK-END
