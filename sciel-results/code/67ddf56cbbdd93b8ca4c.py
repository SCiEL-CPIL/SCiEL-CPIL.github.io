import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    n = 16
    # adj[i] is a set of neighbors
    adj = [set() for _ in range(n)]
    
    # Initialize with complete graph
    for i in range(n):
        for j in range(i+1, n):
            adj[i].add(j)
            adj[j].add(i)
            
    best_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
    best_count = len(best_edges)
    
    # Precompute subsets
    subsets = list(itertools.combinations(range(n), 8))
    
    def is_bipartite(sub):
        # sub is a tuple of 8 vertices
        # Check if induced subgraph is bipartite
        # Map sub to 0..7
        idx = {v: i for i, v in enumerate(sub)}
        # Build local adj
        local_adj = [set() for _ in range(8)]
        for u in sub:
            for v in adj[u]:
                if v in idx:
                    local_adj[idx[u]].add(idx[v])
        
        # BFS
        color = [-1] * 8
        for i in range(8):
            if color[i] == -1:
                color[i] = 0
                queue = [i]
                while queue:
                    u = queue.pop(0)
                    for v in local_adj[u]:
                        if color[v] == -1:
                            color[v] = color[u] ^ 1
                            queue.append(v)
                        elif color[v] == color[u]:
                            return False
        return True

    def find_q3():
        for sub in subsets:
            # Check 3-regularity
            valid = True
            for u in sub:
                if len(adj[u] & set(sub)) != 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check bipartite
            if is_bipartite(sub):
                return sub
        return None

    # Prune phase
    while time.time() < deadline - 10:
        q3 = find_q3()
        if q3 is None:
            break
        # Remove a random edge from q3
        # Get edges in q3
        edges_in_q3 = []
        for u in q3:
            for v in adj[u]:
                if v in q3 and u < v:
                    edges_in_q3.append((u, v))
        if edges_in_q3:
            u, v = random.choice(edges_in_q3)
            adj[u].remove(v)
            adj[v].remove(u)

    # Improve phase
    current_edges = len([1 for i in range(n) for j in adj[i] if i < j])
    
    # List of missing edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        # Try to add a random missing edge
        missing = [e for e in all_edges if e[1] not in adj[e[0]]]
        if not missing:
            break
        u, v = random.choice(missing)
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        
        # Check if valid
        q3 = find_q3()
        if q3 is None:
            # Keep edge
            current_edges += 1
            if current_edges > best_count:
                best_count = current_edges
                best_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
        else:
            # Invalid, remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            # Try a swap: remove a random existing edge, add this one?
            # Or just try another missing edge
            pass
            
    # Return best found
    # Ensure return format is list of [u, v]
    return [[u, v] for u, v in best_edges]
