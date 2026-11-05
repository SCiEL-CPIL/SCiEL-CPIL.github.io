import itertools
import time

def construct():
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # Generate hexagon of radius 3 (37 points)
    R = 3
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a + b) <= R:
                points.append((a, b))
    
    n_points = len(points)
    
    # Compute edges (distance squared == 1)
    # dist_sq(a1, b1, a2, b2) = (a1-a2)^2 + (b1-b2)^2 + (a1-a2)(b1-b2)
    edges = []
    adj = [set() for _ in range(n_points)]
    deg = [0] * n_points
    
    for i in range(n_points):
        a1, b1 = points[i]
        for j in range(i + 1, n_points):
            a2, b2 = points[j]
            da = a1 - a2
            db = b1 - b2
            d2 = da*da + db*db + da*db
            if d2 == 1:
                edges.append((i, j))
                adj[i].add(j)
                adj[j].add(i)
                deg[i] += 1
                deg[j] += 1
                
    total_edges = len(edges)
    
    # We need to remove 5 points to get 32.
    # Minimize lost edges = sum(deg[i] for i in S) - edges_within(S)
    # S is a subset of 5 indices.
    
    # Precompute edge lookup for fast internal edge counting
    is_edge = [[False] * n_points for _ in range(n_points)]
    for i, j in edges:
        is_edge[i][j] = True
        is_edge[j][i] = True
        
    min_cost = float('inf')
    best_S = None
    
    # Iterate over all combinations of 5 points
    # 37 choose 5 is ~4.3M. This is fast.
    for S in itertools.combinations(range(n_points), 5):
        sum_deg = 0
        for idx in S:
            sum_deg += deg[idx]
            
        edges_int = 0
        # Check pairs in S
        for k in range(5):
            i = S[k]
            for l in range(k+1, 5):
                j = S[l]
                if is_edge[i][j]:
                    edges_int += 1
                    
        cost = sum_deg - edges_int
        if cost < min_cost:
            min_cost = cost
            best_S = S
            
        if time.time() > deadline:
            break
            
    # If best_S is None, something went wrong, return first 32
    if best_S is None:
        return [list(p) for p in points[:32]]
        
    # Construct result
    remove_set = set(best_S)
    result = []
    for i in range(n_points):
        if i not in remove_set:
            result.append(list(points[i]))
            
    return result
