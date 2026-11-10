import random
import time

def _has_c6_through(u, v, adj):
    """
    Check if adding edge (u, v) creates a C6.
    True iff there is a simple path of length 5 from u to v.
    """
    # DFS with pruning
    # We track the path vertices to ensure simplicity.
    # n=40, max depth 5.
    
    def dfs(node, depth, visited):
        if depth == 4:
            # Check if v is a neighbor
            # Using 'in' on a set is O(1)
            if v in adj[node]:
                return True
            return False
        
        for nb in adj[node]:
            if nb == u:
                continue
            if nb in visited:
                continue
            visited.add(nb)
            if dfs(nb, depth + 1, visited):
                return True
            visited.remove(nb)
        return False

    visited = {u}
    return dfs(u, 0, visited)

def _greedy_add(adj, candidates, time_limit):
    """
    Greedily add edges from candidates to adj.
    Returns the number of edges added.
    """
    added = 0
    random.shuffle(candidates)
    
    start = time.time()
    for u, v in candidates:
        if time.time() - start > time_limit:
            break
        if v not in adj[u]: # Not already an edge
            if not _has_c6_through(u, v, adj):
                adj[u].add(v)
                adj[v].add(u)
                added += 1
    return added

def construct():
    n = 40
    time_limit_total = 110
    start_time = time.time()
    
    # Initialize with 8 disjoint K5s
    adj = [set() for _ in range(n)]
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj[u].add(v)
                adj[v].add(u)
                
    def get_edge_list(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    best_edges = get_edge_list(adj)
    best_count = len(best_edges)
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Iterated Greedy Loop
    while time.time() - start_time < time_limit_total:
        # 1. Get current candidates
        current_edges_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    current_edges_set.add((u, v))
                else:
                    current_edges_set.add((v, u))
        
        candidates = [e for e in all_edges if e not in current_edges_set]
        
        if not candidates:
            break
            
        # 2. Greedy Add
        # Copy adjacency to try changes
        adj_copy = [set(s) for s in adj]
        
        # Time budget for this iteration
        time_left = time_limit_total - (time.time() - start_time)
        # Use a small chunk of time for each greedy phase to allow many iterations
        iter_time = min(2.0, time_left / 5.0)
        
        if iter_time < 0.1:
            break
            
        _greedy_add(adj_copy, candidates, iter_time)
        
        new_count = sum(len(s) for s in adj_copy) // 2
        
        if new_count > best_count:
            best_count = new_count
            best_edges = get_edge_list(adj_copy)
            adj = adj_copy # Accept the improvement
            
        # 3. Perturbation: Remove some random edges to escape local minimum
        # Perturb the current `adj` (which is the best so far)
        
        # Remove 2 to 5 random edges
        num_remove = random.randint(2, 5)
        current_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    current_list.append((u, v))
        
        if len(current_list) > num_remove:
            random.shuffle(current_list)
            edges_to_remove = current_list[:num_remove]
            
            for u, v in edges_to_remove:
                adj[u].remove(v)
                adj[v].remove(u)
                
            # The next loop iteration will try to re-add edges (Repair)
        else:
            # If we can't remove edges, we are stuck in a sparse state?
            # Or we just break?
            # Let's break to avoid infinite loop on sparse graph
            break

    return best_edges
