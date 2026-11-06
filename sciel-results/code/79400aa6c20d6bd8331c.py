import random
import time
import itertools

def construct():
    max_n = 22
    start_time = time.time()
    time_limit = 115.0

    def check_new_vertex(adj, n, new_v):
        """
        Check if adding vertex new_v (index n-1) to the graph adj[0..n-2]
        creates any regular induced subgraph of size >= 6 that includes new_v.
        """
        # Subsets including new_v are defined by choosing k-1 vertices from 0..n-2
        # Total size k is from 6 to n
        for k in range(6, n + 1):
            # Choose k-1 vertices from the first n-1 vertices
            if k - 1 > n - 1:
                continue
            if k - 1 < 0:
                continue
            
            # Iterate combinations of k-1 vertices from range(n-1)
            for combo in itertools.combinations(range(n - 1), k - 1):
                # Build mask
                mask = 1 << (n - 1) # new_v is n-1
                for v in combo:
                    mask |= (1 << v)
                
                # Check if this induced subgraph is regular
                # Vertices in mask: new_v and combo
                # deg(new_v) = (adj[n-1] & mask).bit_count()
                d0 = (adj[n - 1] & mask).bit_count()
                
                # Check all v in combo
                is_reg = True
                for v in combo:
                    dv = (adj[v] & mask).bit_count()
                    if dv != d0:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
        
        return True

    def solve(n_target):
        """
        Try to build a valid graph of size n_target using backtracking.
        """
        # adj[i] is bitmask for vertex i
        # We use a stack-based DFS to avoid recursion depth issues and allow time checks
        
        # State: (idx, adj_tuple)
        # We try to find a valid configuration for vertex idx given adj_tuple[0..idx-1]
        
        def dfs(idx, adj_tuple):
            if time.time() - start_time > time_limit:
                return None
            
            if idx == n_target:
                return adj_tuple
            
            # Try random configurations for vertex idx
            # We try up to `max_retries` random configurations
            # If we run out, we return None (backtrack)
            
            # To be efficient, we try a mix of random and heuristic?
            # Just random for now, with enough retries.
            retries = 100 if idx < 15 else 50
            
            for _ in range(retries):
                if time.time() - start_time > time_limit:
                    return None
                
                # Generate random mask for edges to 0..idx-1
                mask = 0
                for i in range(idx):
                    if random.random() < 0.5:
                        mask |= (1 << i)
                
                # Create new adj list
                new_adj = list(adj_tuple)
                new_adj[idx] = mask
                for i in range(idx):
                    if mask & (1 << i):
                        new_adj[i] |= (1 << idx)
                    # else bit remains 0 (since idx > i, it wasn't set before)
                
                # Check if this extension is valid (no new regular subgraphs)
                if check_new_vertex(new_adj, idx + 1, idx):
                    result = dfs(idx + 1, tuple(new_adj))
                    if result:
                        return result
                
                # If recursion failed or check failed, try next mask
            
            return None

        # Start DFS
        initial_adj = tuple([0] * n_target)
        return dfs(0, initial_adj)

    # Try to solve for n=22 down to 20
    for n in range(max_n, 19, -1):
        if time.time() - start_time > time_limit:
            break
        res = solve(n)
        if res:
            edges = [[i, j] for i in range(n) for j in range(i+1, n) if (res[i] >> j) & 1]
            return {'n': n, 'edges': edges}
    
    # Fallback: Return a known valid small graph if we fail
    # A graph with no edges is valid (no regular subgraph of size >= 6? 
    # Wait, empty graph on k vertices is regular (degree 0).
    # So we need a graph with NO regular induced subgraph of size >= 6.
    # If we return n=1, it's trivially valid.
    # But we should try to return the best we found.
    # Since we didn't store intermediate results in this simple version, we return n=1.
    # We can improve by storing the best found during the search, but for now, 
    # if we fail 22, 21, 20, we return 1.
    
    # Actually, let's try a simple greedy fallback to get at least n=10 or so.
    best_n = 1
    best_edges = []
    
    for n in range(10, 19, -1):
        if time.time() - start_time > time_limit:
            break
        # Quick greedy
        adj = [0] * n
        success = True
        for i in range(1, n):
            found = False
            for _ in range(50):
                mask = 0
                for j in range(i):
                    if random.random() < 0.5:
                        mask |= (1 << j)
                adj[i] = mask
                for j in range(i):
                    if mask & (1 << j):
                        adj[j] |= (1 << i)
                
                # Quick check: is the whole graph valid? 
                # This is expensive. 
                # Let's just assume it works for the fallback if we get here.
                # Actually, we can't assume.
                # Let's just return n=1 if we fail the main search.
                found = True
                break
            if not found:
                success = False
                break
        if success:
            # Verify?
            # For fallback, we skip verification to save time, risking invalidity.
            # But the task says invalid scores 0.
            # So we should verify.
            # But verification is slow.
            # Let's just return n=1 to be safe if main search fails.
            pass
            
    return {'n': 1, 'edges': []}
