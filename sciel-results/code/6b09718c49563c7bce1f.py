import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix for original graph (for validation if needed, but we use uncovered set)
    # We just need to track uncovered edges.
    # Use a set of tuples (u, v) with u < v for simplicity?
    # Or a 2D list of booleans.
    # 2D list is faster for intersection.
    
    # Initialize uncovered as a list of sets of neighbors
    # uncovered[u] is a set of v such that (u,v) is an uncovered edge.
    uncovered = [set() for _ in range(n)]
    for u, v in edges:
        uncovered[u].add(v)
        uncovered[v].add(u)
        
    bicliques = []
    
    while any(uncovered[u] for u in range(n)):
        if time.time() > deadline:
            break
            
        best_L = None
        best_R = None
        max_covered = 0
        
        # Try L sizes 1 to 5
        # We can iterate over combinations of vertices
        # But we only care about vertices that have uncovered edges.
        # Actually, if a vertex has no uncovered edges, it can't be in L or R effectively.
        # But R can contain vertices with no uncovered edges? No, because if v in R, 
        # then (u,v) must be uncovered for all u in L. 
        # So R must be subset of common uncovered neighbors of L.
        
        # To speed up, we can just iterate over all subsets of vertices of size k
        # But 30 choose 5 is 142506. That's a lot.
        # Maybe we should be smarter.
        
        # Alternative: 
        # 1. Find the vertex u with the most uncovered edges.
        # 2. Try L={u}.
        # 3. Try L={u, v} for v in uncovered[u].
        # 4. Try L={u, v, w} for w in uncovered[u] intersect uncovered[v].
        
        # Let's find the vertex with max degree in uncovered graph.

        u_star = max(range(n), key=lambda x: len(uncovered[x]))
        
        # Candidate 1: L = {u_star}
        R = set(uncovered[u_star])
        covered = len(R)
        if covered > max_covered:
            max_covered = covered
            best_L = [u_star]
            best_R = list(R)
            
        # Candidate 2: L = {u_star, v} for v in uncovered[u_star]
        for v in list(uncovered[u_star]):
            # R is common neighbors of u_star and v in uncovered graph, excluding u_star and v
            common = uncovered[u_star].intersection(uncovered[v])
            common.discard(u_star)
            common.discard(v)
            if not common:
                continue
            R = list(common)
            covered = 2 * len(R)
            if covered > max_covered:
                max_covered = covered
                best_L = [u_star, v]
                best_R = R
                
        # Candidate 3: L = {u_star, v, w}
        # We can limit this to v in top neighbors of u_star
        # And w in common neighbors of u_star and v
        # To keep it fast, maybe just try a few.
        
        # Let's try a more generic search for small L sizes
        # But the above heuristic might be enough.
        
        # If no biclique found (should not happen), break
        if best_L is None:
             # Fallback: pick any edge
             for u in range(n):
                 if uncovered[u]:
                     v = next(iter(uncovered[u]))
                     best_L = [u]
                     best_R = [v]
                     break
             if best_L is None:
                 break
                 
        bicliques.append([best_L, best_R])
        
        # Remove edges from uncovered
        for u in best_L:
            for v in best_R:
                if v in uncovered[u]:
                    uncovered[u].discard(v)
                if u in uncovered[v]:
                    uncovered[v].discard(u)
                    
    # Convert to required format: list of [left, right]
    # Ensure types are int and list
    result = []
    for L, R in bicliques:
        result.append([[int(x) for x in L], [int(y) for y in R]])
        
    return result
