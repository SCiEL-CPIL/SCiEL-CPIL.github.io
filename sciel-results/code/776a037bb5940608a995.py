import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    def check_safe(G, u, v):
        # Returns True if adding (u,v) to G does not create a C4
        # G is a list of sets of neighbors for each node? 
        # Or a set of edges?
        # Using adjacency sets is faster for intersection.
        # But we need to check distance 3.
        # dist(u, v) == 3 iff N(u) and N(v) are disjoint, but N(N(u)) and N(N(v)) intersect?
        # Actually, if N(u) and N(v) are disjoint, dist is 3 if there is an edge between N(u) and N(v).
        # Let's implement this.
        
        nu = G[u]
        nv = G[v]
        
        # If they share a neighbor, dist is 2. Safe.
        if nu & nv:
            return True
            
        # Check if there is an edge between any node in nu and any node in nv
        # This means dist is 3.
        for x in nu:
            if G[x] & nv:
                return False
        return True

    def try_construct(n):
        # G[c] is a list of sets of neighbors for vertices 0..n-1
        G = [ [set() for _ in range(n)] for _ in range(5) ]
        
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            available = []
            for c in range(5):
                if check_safe(G[c], u, v):
                    available.append(c)
            
            if not available:
                return None
            
            # Heuristic: pick color that minimizes degree sum or something?
            # For now, random
            c = random.choice(available)
            G[c][u].add(v)
            G[c][v].add(u)
            
        # Convert to color list
        colors = [0] * (n * (n-1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                # Find which color class contains (i,j)
                # We stored it in G, but G is a list of sets.
                # We need to recover the color.
                # This is inefficient. Better to store colors directly.
                pass
        # The above reconstruction is O(n^2 * 5).
        # Let's re-implement try_construct to store colors directly.
        return None # Placeholder

    # Better implementation
    def try_construct_fast(n):
        # colors[i, j] where i < j
        # Using a 2D array or list of lists
        # We need to maintain G[c] as adjacency sets for fast checks.
        G = [ [set() for _ in range(n)] for _ in range(5) ]
        color_matrix = [[-1]*n for _ in range(n)]
        
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        
        for u, v in edges:
            available = []
            for c in range(5):
                # Check safety
                nu = G[c][u]
                nv = G[c][v]
                
                # If intersect, dist 2, safe
                if nu & nv:
                    available.append(c)
                    continue
                    
                # Check dist 3
                safe = True
                # Optimization: iterate smaller set
                if len(nu) < len(nv):
                    for x in nu:
                        if G[c][x] & nv:
                            safe = False
                            break
                else:
                    for x in nv:
                        if G[c][x] & nu:
                            safe = False
                            break
                            
                if safe:
                    available.append(c)
            
            if not available:
                return None
            
            c = random.choice(available)
            G[c][u].add(v)
            G[c][v].add(u)
            color_matrix[u][v] = c
            color_matrix[v][u] = c
            
        # Flatten
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(color_matrix[i][j])
        
        return {'n': n, 'colors': colors}

    # Try n=26 first
    # Use multiple random seeds/restarts
    max_restarts = 1000
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        res = try_construct_fast(26)
        if res:
            return res
            
    # If 26 fails, try 25
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        res = try_construct_fast(25)
        if res:
            return res
            
    # Fallback to seed result
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

