import random
import itertools

def construct():
    # Vertices 0..17
    n = 18
    
    # Helper to check K4-freeness and calculate score
    def evaluate(edges):
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check K4
        # A K4 exists if there is a triangle u-v-w and a node x connected to all three
        # Or simply check all 4-subsets
        for subset in itertools.combinations(range(n), 4):
            u, v, w, x = subset
            if (adj[u][v] and adj[u][w] and adj[u][x] and 
                adj[v][w] and adj[v][x] and 
                adj[w][x]):
                return None # Invalid
                
        # Calculate max triangle-free induced subgraph
        max_tf = 0
        # Check all subsets? 2^18 is 262k. 262k * (cost of checking triangle-free)
        # Checking triangle-free for a subset S:
        # For each pair in S, check if any 3rd vertex in S connects to both.
        # This is O(|S|^3) or O(|S|^2 * deg).
        # 262k * 18^3 is too slow for Python in 120s if done naively for every candidate.
        
        # Optimization:
        # The score is the size of the largest induced subgraph with no triangles.
        # We can do a DFS/Backtracking to find the max triangle-free set.
        # Or, since n=18 is small, we can iterate subsets in descending size.
        
        # Precompute adjacency bitmask for speed
        adj_mask = [0]*n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            adj_mask[i] = mask
            
        def is_triangle_free(mask):
            # mask represents the subset of vertices
            # Extract vertices
            verts = []
            for i in range(n):
                if mask & (1 << i):
                    verts.append(i)
            # Check all triples
            for u in verts:
                for v in verts:
                    if u < v:
                        for w in verts:
                            if v < w:
                                if adj[u][v] and adj[v][w] and adj[w][u]:
                                    return False
            return True
            
        # Search for max size
        # Start from size n down to 1
        # But we know 12 is the current best. We can stop if we find 11?
        # Actually, we just need the max.
        
        # Fast check:
        # Any subset of size 13 must contain a triangle?
        # In T(18,3), max TF is 12.
        
        for size in range(n, 0, -1):
            found = False
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_triangle_free(mask):
                    return size
            if not found:
                continue
        return 0

    # Initial: Turan T(18,3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = []
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    edges.append([u, v])
    
    best_score = evaluate(edges)
    best_edges = edges[:]
    
    # Local Search
    # Try removing random edges
    # Removing edges makes it easier to be triangle-free, so score likely increases.
    # We want to DECREASE score.
    # This implies we need to ADD edges.
    # But we are at T(18,3), max edges.
    # So we must CHANGE structure.
    
    # Try: Swap edges. Remove one, add another.
    random.seed(42)
    current_edges = best_edges[:]
    current_score = best_score
    
    # Adjacency matrix for current_edges
    adj = [[False]*n for _ in range(n)]
    for u, v in current_edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Set of edges for fast lookup
    edge_set = set()
    for u, v in current_edges:
        edge_set.add((min(u,v), max(u,v)))
        
    for _ in range(500): # Iterations
        # Pick a random edge to remove
        if not current_edges: break
        idx = random.randint(0, len(current_edges)-1)
        u, v = current_edges[idx]
        
        # Pick a random non-edge to add
        # Ensure it's not a self-loop or existing edge
        attempts = 0
        while True:
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            if a == b: continue
            if (min(a,b), max(a,b)) in edge_set: continue
            break
            
        # Check if adding (a,b) creates K4
        # A K4 is created if a,b and two other vertices form a clique.
        # Specifically, if there exist x,y such that:
        # x,y connected to a AND b? No.
        # Adding edge (a,b) creates a K4 if there is a triangle involving a and b?
        # No.
        # It creates a K4 if there is a pair (x,y) such that:
        # x,y are connected to a, x,y are connected to b, and x-y are connected.
        
        valid = True
        for x in range(n):
            if x == a or x == b: continue
            if not adj[a][x]: continue
            for y in range(n):
                if y == a or y == b or y == x: continue
                if not adj[b][y]: continue
                if not adj[x][y]: continue
                # Check if a-b-x-y is K4?
                # We added a-b.
                # We have a-x, a-y (if adj[a][y]), b-x (if adj[b][x]), b-y.
                # We need to check if {a,b,x,y} is a K4.
                # Edges needed: ab (yes), ax (yes), ay (?), bx (?), by (yes), xy (yes).
                # We need adj[a][y] and adj[b][x] to be True.
                if adj[a][y] and adj[b][x]:
                    valid = False
                    break
            if not valid: break
            
        if not valid:
            continue
            
        # If valid, apply change
        # Remove u,v
        current_edges.pop(idx)
        edge_set.discard((min(u,v), max(u,v)))
        adj[u][v] = False
        adj[v][u] = False
        
        # Add a,b
        current_edges.append([a, b])
        edge_set.add((min(a,b), max(a,b)))
        adj[a][b] = True
        adj[b][a] = True
        
        score = evaluate(current_edges)
        
        if score is None:
            # Revert
            current_edges.remove([a, b])
            edge_set.discard((min(a,b), max(a,b)))
            adj[a][b] = False
            adj[b][a] = False
            current_edges.append([u, v])
            edge_set.add((min(u,v), max(u,v)))
            adj[u][v] = True
            adj[v][u] = True
            continue
            
        if score < current_score:
            current_score = score
            best_score = score
            best_edges = current_edges[:]
            
        else:
            # Revert
            current_edges.remove([a, b])
            edge_set.discard((min(a,b), max(a,b)))
            adj[a][b] = False
            adj[b][a] = False
            current_edges.append([u, v])
            edge_set.add((min(u,v), max(u,v)))
            adj[u][v] = True
            adj[v][u] = True
            
    return best_edges
