import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 40
    target_edges = 401
    
    # Adjacency matrix for fast lookup
    adj = np.zeros((n, n), dtype=bool)
    
    # common_neighbors[u][v] = number of common neighbors of u and v
    # This is the "book size" for edge (u,v) if the edge exists.
    # If the edge doesn't exist, it's the number of triangles that would be formed if we added it.
    # We maintain this matrix to allow O(1) delta calculation.
    common = np.zeros((n, n), dtype=int)
    
    # Initialize graph: K_20,20 + 1 edge (0,1)
    # This is a strong starting point near the extremal construction.
    for u in range(20):
        for v in range(20, 40):
            adj[u, v] = True
            adj[v, u] = True
            common[u, v] += 0 # Initially 0 for bipartite edges? No, common neighbors of u in A and v in B are 0.
    # Wait, common neighbors of u (in A) and v (in B) in K_20,20 is 0.
    # But we need to initialize `common` correctly.
    # Let's re-initialize common from scratch for correctness.
    
    adj = np.zeros((n, n), dtype=bool)
    for u in range(20):
        for v in range(20, 40):
            adj[u, v] = True
            adj[v, u] = True
    adj[0, 1] = True
    adj[1, 0] = True
    
    # Compute initial common neighbors
    # For all pairs u, v, common[u][v] = sum(adj[u] & adj[v])
    # This is O(n^3) but n=40 is small. 40^3 = 64,000 operations. Very fast.
    for u in range(n):
        for v in range(u + 1, n):
            cnt = np.sum(adj[u] & adj[v])
            common[u, v] = cnt
            common[v, u] = cnt
            
    # List of present edges
    present_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                present_edges.add((u, v))
                
    # List of absent edges for quick sampling? 
    # Total edges = 780. Present = 401. Absent = 379.
    # We can just sample randomly and check.
    
    # Current max book size
    def get_max_book():
        max_b = 0
        for u, v in present_edges:
            if common[u, v] > max_b:
                max_b = common[u, v]
        return max_b
        
    current_score = get_max_book()
    best_score = current_score
    best_adj = adj.copy()
    best_common = common.copy()
    
    # Pre-compute list of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Optimization: Steepest Descent Local Search
    # Instead of random swaps, we can try to find the best swap.
    # But checking all O(E^2) swaps is too slow (401^2 ~ 160k checks per step).
    # We will use a hybrid: Random swaps but with fast delta calculation.
    
    T = 1.0
    T_min = 0.01
    alpha = 0.9999
    
    while time.time() < deadline:
        # Pick random edge to remove (u, v)
        # Pick random edge to add (x, y)
        # To ensure x,y not in present and u,v in present
        
        # Sample remove edge
        while True:
            u, v = random.sample(range(n), 2)
            if u > v: u, v = v, u
            if (u, v) in present_edges:
                break
                
        # Sample add edge
        while True:
            x, y = random.sample(range(n), 2)
            if x > y: x, y = y, x
            if (x, y) not in present_edges:
                break
                
        # Check if removing (u,v) and adding (x,y) is valid (x,y not same as u,v, etc.)
        # Also x,y must not be u,v or v,u obviously.
        
        # Calculate Delta in max book size.
        # This is tricky because max_book is a global statistic.
        # We track the count of edges for each book size.
        
        # Let's maintain a histogram of book sizes for present edges.
        # hist[k] = number of edges with book size k.
        
        # Re-initialize histogram
        hist = np.zeros(50, dtype=int) # Max book size won't be huge
        for u, v in present_edges:
            hist[common[u, v]] += 1
            
        def current_max():
            for k in range(len(hist)-1, -1, -1):
                if hist[k] > 0:
                    return k
            return 0
            
        cur_max = current_max()
        
        # Perform swap
        # Remove (u,v)
        adj[u, v] = False
        adj[v, u] = False
        present_edges.discard((u, v))
        
        # Update common neighbors for all pairs involving u or v
        # This is O(n) per edge removal/addition.
        # When we remove (u,v):
        # For any w != u, v:
        #   common[u, w] changes if v was a neighbor of w? No, common[u,w] = |N(u) \cap N(w)|.
        #   Removing edge (u,v) means v is no longer in N(u).
        #   So for any w, if v is in N(w), then common[u, w] decreases by 1.
        #   Similarly common[v, w] decreases by 1 if u is in N(w).
        #   Also common[u, v] itself becomes irrelevant as edge is removed, but value changes.
        
        # This incremental update is complex to get right quickly.
        # Given n=40, recalculating common[u] and common[v] rows is O(n^2) = 1600 ops.
        # This is very fast.
        
        # Recalculate row u and v of common matrix
        for w in range(n):
            if w == u or w == v:
                continue
            # common[u, w] = sum(adj[u] & adj[w])
            # We just recompute it.
            common[u, w] = np.sum(adj[u] & adj[w])
            common[w, u] = common[u, w]
            common[v, w] = np.sum(adj[v] & adj[w])
            common[w, v] = common[v, w]
        common[u, v] = np.sum(adj[u] & adj[v])
        common[v, u] = common[u, v]
        
        # Add (x,y)
        adj[x, y] = True
        adj[y, x] = True
        present_edges.add((x, y))
        
        # Recalculate row x and y of common matrix
        for w in range(n):
            if w == x or w == y:
                continue
            common[x, w] = np.sum(adj[x] & adj[w])
            common[w, x] = common[x, w]
            common[y, w] = np.sum(adj[y] & adj[w])
            common[w, y] = common[y, w]
        common[x, y] = np.sum(adj[x] & adj[y])
        common[y, x] = common[x, y]
        
        # Update Histogram
        # We need to know the old book sizes of edges that changed.
        # This approach of rebuilding histogram every time is O(E) = 401 ops.
        # Total cost per step: 2 * O(n^2) for common update + O(E) for hist update + O(E) for max check.
        # ~ 3000 ops per step. 100 seconds / 3000 ops = 33,000 steps.
        # This is much better than the previous ~100 steps.
        
        new_hist = np.zeros(50, dtype=int)
        new_max = 0
        for u, v in present_edges:
            b = common[u, v]
            new_hist[b] += 1
            if b > new_max:
                new_max = b
                
        delta = new_max - cur_max
        
        # Acceptance criteria: Simulated Annealing
        if delta < 0:
            accept = True
        elif delta == 0:
            accept = random.random() < 0.5
        else:
            accept = random.random() < np.exp(-delta / T)
            
        if accept:
            cur_max = new_max
            hist = new_hist
            if cur_max < best_score:
                best_score = cur_max
                best_adj = adj.copy()
                best_common = common.copy()
        else:
            # Revert
            adj[u, v] = True
            adj[v, u] = True
            present_edges.add((u, v))
            
            adj[x, y] = False
            adj[y, x] = False
            present_edges.discard((x, y))
            
            # Recalculate common rows again (Revert)
            for w in range(n):
                if w == u or w == v:
                    continue
                common[u, w] = np.sum(adj[u] & adj[w])
                common[w, u] = common[u, w]
                common[v, w] = np.sum(adj[v] & adj[w])
                common[w, v] = common[v, w]
            common[u, v] = np.sum(adj[u] & adj[v])
            common[v, u] = common[u, v]
            
            for w in range(n):
                if w == x or w == y:
                    continue
                common[x, w] = np.sum(adj[x] & adj[w])
                common[w, x] = common[x, w]
                common[y, w] = np.sum(adj[y] & adj[w])
                common[w, y] = common[y, w]
            common[x, y] = np.sum(adj[x] & adj[y])
            common[y, x] = common[x, y]
            
            # Revert Hist
            # We can just recalculate hist for best_score check later, 
            # but for loop consistency, we keep the old hist.
            # Actually, we didn't update the global hist variable on revert, 
            # so it's still the old one. Correct.
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Return best found
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                result.append([u, v])
                
    return result
