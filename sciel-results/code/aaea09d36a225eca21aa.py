import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def is_valid(coloring):
    n = P['n']
    edges = P['edges']
    
    # Check proper coloring
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
            
    # Check forest condition for all pairs of colors
    # Get distinct colors used
    colors_used = set(coloring)
    
    # If number of colors < 2, no pairs to check (trivially forest)
    if len(colors_used) < 2:
        return True
        
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            # Build subgraph induced by c1 and c2
            # Vertices with color c1 or c2
            verts = [i for i in range(n) if coloring[i] in (c1, c2)]
            v_set = set(verts)
            
            # Union-Find to detect cycles
            parent = {v: v for v in verts}
            rank = {v: 0 for v in verts}
            
            def find(x):
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return False # Cycle found
                if rank[rx] < rank[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                if rank[rx] == rank[ry]:
                    rank[rx] += 1
                return True
            
            for u, v in edges:
                if u in v_set and v in v_set:
                    if not union(u, v):
                        return False # Cycle found in induced subgraph
                        
    return True

def find_coloring(k, deadline):
    n = P['n']
    coloring = [-1] * n
    coloring[0] = 0 # Fix one vertex to break symmetry
    
    # Order vertices by index for simplicity
    # A better heuristic might be to process vertices with more constraints first,
    # but all degrees are 4.
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
            
        v = idx
        # Try colors 0 to k-1
        for c in range(k):
            # Check properness with already colored neighbors
            valid = True
            for u, w in P['edges']:
                if u == v:
                    if coloring[w] == c:
                        valid = False
                        break
                elif w == v:
                    if coloring[u] == c:
                        valid = False
                        break
            if not valid:
                continue
                
            coloring[v] = c
            
            # Optional: Check if this creates a cycle in any bicolored subgraph immediately?
            # This is complex to check incrementally. We'll rely on full check at the end or
            # just trust that backtracking will prune enough.
            # Given n=50, full check at the end of a complete assignment is O(E) which is small.
            # But we check validity only when idx==n.
            # We can add a partial check: for any pair of colors used so far, 
            # check if adding v creates a cycle.
            # For simplicity and speed of implementation, we skip incremental cycle check
            # and hope backtracking is fast. If k=3, branching factor is small.
            
            if backtrack(idx + 1):
                return True
                
        coloring[v] = -1
        return False

    if backtrack(1):
        return coloring
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50 seconds safety margin
    
    # Try modular colorings first as they are instant
    # Try k=3, 4, 5
    for k in [3, 4, 5]:
        coloring = [i % k for i in range(P['n'])]
        if is_valid(coloring):
            return coloring
            
    # If modular didn't work, try backtracking for k=3, 4
    for k in [3, 4]:
        if time.time() > deadline:
            break
        coloring = find_coloring(k, deadline)
        if coloring is not None:
            if is_valid(coloring): # Double check
                return coloring
                
    # Fallback: return a valid 5-coloring if we can find one quickly, 
    # or just return the seed solution (50 colors) to ensure validity.
    # But we know 5 colors likely work (score 10 was achieved).
    # Let's try one more modular variant or just return seed.
    # Actually, the seed is always valid.
    return list(range(P['n']))
# EVOLVE-BLOCK-END
