import time
import random

def _creates_c4(adj, u, v):
    """
    Check if adding edge (u, v) creates a C4.
    adj is a list of sets.
    A C4 is created if there exists a path u-x-y-v where x in N(u), y in N(v), and (x,y) is an edge.
    """
    nu = adj[u]
    nv = adj[v]
    # Check if any neighbor of u is adjacent to any neighbor of v.
    # To optimize, iterate over the smaller neighbor set.
    if len(nu) > len(nv):
        nu, nv = nv, nu
        
    for x in nu:
        if adj[x] & nv:
            return True
    return False

def _try_construct(n, min_deg, time_limit):
    start_time = time.time()
    
    # We will try to build the graph.
    # Use a list of sets for adjacency.
    
    for attempt in range(5000):
        if time.time() - start_time > time_limit:
            break
            
        adj = [set() for _ in range(n)]
        
        # Strategy:
        # 1. Start with a random permutation of vertices.
        # 2. For each vertex u in the permutation, try to connect it to previous vertices.
        # 3. If stuck or min degree not met, use local repair or restart.
        
        # To maximize min degree, we should ensure no vertex is left behind.
        # A simple greedy approach:
        # Iterate through all pairs (u, v) in a random order?
        # Or, better:
        # While min degree < min_deg:
        #   Pick u with min degree.
        #   Pick v not adjacent to u.
        #   If adding (u, v) is safe, add it.
        #   Else, try removing an edge incident to u or v to free up space? 
        #   Or just restart.
        
        # Let's try a hybrid: 
        # 1. Greedy add edges to satisfy min degree.
        # 2. If stuck, try a few local repairs (remove random edge, retry).
        # 3. If still stuck, restart.
        
        success = False
        
        # To speed up, we can pre-shuffle the order of vertex processing
        # but the "min degree" selection is dynamic.
        
        # Let's use a loop that runs for a max number of steps
        max_steps = n * n * 5
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
                
            # Check if done
            min_deg_curr = min(len(adj[i]) for i in range(n))
            if min_deg_curr >= min_deg:
                success = True
                break
                
            # Find vertex with min degree
            u = min(range(n), key=lambda i: len(adj[i]))
            
            # Find valid neighbors for u
            candidates = [v for v in range(n) if v != u and v not in adj[u]]
            random.shuffle(candidates)
            
            added = False
            for v in candidates:
                if not _creates_c4(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
                    
            if not added:
                # Stuck at u.
                # Try a local repair: remove a random edge incident to u or its neighbors
                # to potentially free up a connection.
                # But this is complex. Restarting is often more effective for this problem size.
                # However, let's try one simple repair: remove a random edge from u's neighbors
                # if u has any, or just break and restart.
                
                # Given the high density required (delta ~ 8 for n=47),
                # getting stuck is likely due to C4 constraints.
                # Restarting with a new random seed/order is a good strategy.
                break
                
        if success:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append([u, v])
            return {'n': n, 'edges': edges}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for overhead
    
    # We aim for n=47 first, then 46, then 45...
    # For n=47, min degree must be >= 8.
    # For n=46, min degree must be >= 7.
    # For n=45, min degree must be >= 6.
    
    # Time budget allocation:
    # n=47: 40s
    # n=46: 40s
    # n=45: 40s
    # n=44: 15s
    # n=43: 10s
    
    targets = [
        (47, 8, 40.0),
        (46, 7, 40.0),
        (45, 6, 40.0),
        (44, 5, 15.0),
        (43, 4, 10.0),
        (42, 3, 5.0)
    ]
    
    for n_target, min_deg_req, t_budget in targets:
        if time.time() > deadline:
            break
            
        # Calculate remaining time for this specific target
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < t_budget:
            t_budget = remaining - 1.0 # leave 1s margin
        if t_budget < 1.0:
            continue
            
        res = _try_construct(n_target, min_deg_req, t_budget)
        if res:
            return res
            
    # Fallback: return a trivial valid graph
    # n=1, min degree 0 (1-39 is negative, so min degree >= 0 is always true)
    # Wait, min degree >= n-39. For n=1, n-39 = -38. Min degree 0 is >= -38.
    # So n=1 is always valid.
    return {'n': 1, 'edges': []}

# Test harness for local verification (not part of the submitted logic, but useful for debugging)
if __name__ == "__main__":
    # Quick test
    res = construct()
    print(res)
