import numpy as np
import random

def _has_c6_path5(adj, u, v):
    """
    Check if adding edge (u, v) to the graph represented by adj 
    would create a C6. This is equivalent to checking if there is 
    a path of length 5 between u and v in the current graph 
    (excluding the edge (u,v) which is not present yet).
    Path: u - a - b - c - d - v
    """
    n = adj.shape[0]
    # Neighbors of u (excluding v)
    nu = np.where(adj[u])[0]
    nu = [x for x in nu if x != v]
    
    # Neighbors of v (excluding u)
    nv = np.where(adj[v])[0]
    nv = [x for x in nv if x != u]
    
    for a in nu:
        for d in nv:
            if a == d:
                continue
            # We need a path a - b - c - d of length 3
            # b is neighbor of a, c is neighbor of d
            # b != c, b != d, c != a
            # b, c not in {u, v}
            
            na = np.where(adj[a])[0]
            nd = np.where(adj[d])[0]
            
            # Filter candidates for b and c
            # b must be in na, not u, not v, not d, not a (simple graph)
            # c must be in nd, not u, not v, not a, not d
            
            # Optimization: iterate over smaller set
            if len(na) < len(nd):
                for b in na:
                    if b == u or b == v or b == d or b == a:
                        continue
                    # Check if b is connected to any c in nd
                    # c must be neighbor of d, and edge (b,c) exists
                    # c != a, c != d, c != u, c != v
                    for c in nd:
                        if c == a or c == d or c == u or c == v:
                            continue
                        if b == c:
                            continue
                        if adj[b, c]:
                            return True
            else:
                for c in nd:
                    if c == u or c == v or c == a or c == d:
                        continue
                    for b in na:
                        if b == u or b == v or b == d or b == a:
                            continue
                        if b == c:
                            continue
                        if adj[b, c]:
                            return True
    return False

def _is_c6_free(adj):
    """
    Check if the graph is C6-free.
    An edge (u, v) is in a C6 if there is a path of length 5 between u and v 
    in G \ {(u,v)}.
    """
    n = adj.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                # Temporarily remove edge for check? 
                # _has_c6_path5 assumes (u,v) is NOT in the graph.
                # So we just call it. It checks for path u-a-b-c-d-v.
                # Since adj[u,v] is True, we need to be careful.
                # The function _has_c6_path5 excludes v from N(u) and u from N(v).
                # It does NOT check if adj[u,v] is true/false.
                # So it correctly finds paths of length 5 not using (u,v).
                if _has_c6_path5(adj, u, v):
                    return False
    return True

def _generate_random_regular(n, k, seed):
    """
    Generate a random k-regular graph on n vertices.
    Returns None if not possible (e.g., n*k odd).
    """
    if n * k % 2 != 0:
        return None
    
    random.seed(seed)
    adj = np.zeros((n, n), dtype=bool)
    
    # Use the configuration model or a simple swapping algorithm
    # For simplicity and robustness, we can use a random permutation approach
    # or just try to build it.
    
    # A simpler method: Start with a ring graph (2-regular) and add edges?
    # Or use a library? No, stdlib only.
    
    # Let's use a simple heuristic: 
    # 1. Create a list of stubs.
    # 2. Pair them randomly.
    # 3. Check for self-loops and multi-edges.
    # 4. If invalid, swap endpoints to fix.
    
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Initial pairing
    edges = set()
    valid = True
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Self loop, need to fix
            valid = False
            break
        if (min(u, v), max(u, v)) in edges:
            # Multi-edge
            valid = False
            break
        edges.add((min(u, v), max(u, v)))
        
    if not valid:
        # Try to fix by swapping
        # This is complex. Let's just retry with a new seed if this fails.
        # Or use a different generation method.
        pass
        
    # Better method: Start with a known k-regular graph and swap edges.
    # For k=5, n=40.
    # We can construct a 5-regular graph by taking the union of 5 perfect matchings?
    # Or just use a circulant-like structure and randomize.
    
    # Let's try a simple random walk construction or just retry.
    # Given the small size, we can also just generate a random graph G(n, p) 
    # and prune/add to make it regular?
    
    # Alternative: Use the fact that we just need a good starting point.
    # We can start with a random graph with average degree 5, 
    # then try to make it regular or just use it as is and check C6.
    # The C6 check is the bottleneck.
    
    # Let's stick to the configuration model but with a robust fix.
    # Actually, for n=40, k=5, the probability of a simple graph is high.
    # Let's just retry until we get a simple graph.
    
    for _ in range(100):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        random.shuffle(stubs)
        
        edges = set()
        is_simple = True
        for i in range(0, len(stubs), 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                is_simple = False
                break
            if (min(u, v), max(u, v)) in edges:
                is_simple = False
                break
            edges.add((min(u, v), max(u, v)))
            
        if is_simple:
            adj = np.zeros((n, n), dtype=bool)
            for u, v in edges:
                adj[u, v] = True
                adj[v, u] = True
            return adj
            
    return None

def construct():
    n = 40
    best_edges = []
    best_count = 0
    
    # Try multiple random seeds to find a good starting graph
    for seed in range(50):
        # Start with 5-regular (100 edges)
        adj = _generate_random_regular(n, 5, seed)
        if adj is None:
            continue
            
        # Check if initial graph is C6-free
        if not _is_c6_free(adj):
            continue
            
        # Greedy edge addition
        # Get all non-edges
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    non_edges.append((u, v))
                    
        random.seed(seed)
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if not adj[u, v]:
                if not _has_c6_path5(adj, u, v):
                    adj[u, v] = True
                    adj[v, u] = True
                    
        # Count edges
        count = int(np.sum(adj)) // 2
        if count > best_count:
            best_count = count
            best_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        best_edges.append([u, v])
                        
        # If we reached 105+, we might stop early? 
        # But let's try all seeds to be safe.
        
    # If no valid 5-regular graph found, try 4-regular (80 edges) and add
    if best_count == 0:
        for seed in range(50):
            adj = _generate_random_regular(n, 4, seed)
            if adj is None:
                continue
            if not _is_c6_free(adj):
                continue
            non_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not adj[u, v]:
                        non_edges.append((u, v))
            random.seed(seed)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if not adj[u, v]:
                    if not _has_c6_path5(adj, u, v):
                        adj[u, v] = True
                        adj[v, u] = True
            count = int(np.sum(adj)) // 2
            if count > best_count:
                best_count = count
                best_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u, v]:
                            best_edges.append([u, v])

    return best_edges
