import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def _greedy_construct(seed):
        """
        Fast greedy construction using integer common-neighbor counts.
        Returns list of edges [u, v].
        """
        random.seed(seed)
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        
        adj = [[False] * n for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        current_edges = []
        
        for u, v in edges_to_try:
            if adj[u][v]:
                continue
            
            # Check 1: direct pair (u, v) must have < 3 common neighbors
            if common[u][v] >= 3:
                continue
            
            valid = True
            
            # Check 2: For every neighbor w of u (w != v), common[w][v] must be < 2
            # because adding (u,v) increases common[w][v] by 1
            for w in range(n):
                if w != v and adj[u][w]:
                    if common[w][v] >= 2:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check 3: For every neighbor w of v (w != u), common[u][w] must be < 2
            # because adding (u,v) increases common[u][w] by 1
            for w in range(n):
                if w != u and adj[v][w]:
                    if common[u][w] >= 2:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Add edge
            current_edges.append([u, v])
            adj[u][v] = adj[v][u] = True
            
            # Update common counts
            # For each neighbor w of u (w != v), common[w][v] increases by 1
            for w in range(n):
                if w != v and adj[u][w]:
                    common[w][v] += 1
                    common[v][w] += 1
                    
            # For each neighbor w of v (w != u), common[u][w] increases by 1
            for w in range(n):
                if w != u and adj[v][w]:
                    common[u][w] += 1
                    common[w][u] += 1
                    
        return current_edges

    def _local_search(initial_edges, seed):
        """
        Local search starting from initial_edges.
        Tries to add edges greedily; if stuck, removes a random edge to escape.
        """
        random.seed(seed)
        
        adj = [[False] * n for _ in range(n)]
        common = [[0] * n for _ in range(n)]
        edges = [e[:] for e in initial_edges]
        
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
        
        # Recompute common counts from scratch (O(n^3) = 8000 ops, fine)
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                cnt = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        cnt += 1
                common[u][v] = cnt
        
        current_count = len(edges)
        best_local_count = current_count
        best_local_edges = [e[:] for e in edges]
        
        def can_add(u, v):
            if adj[u][v]:
                return False
            if common[u][v] >= 3:
                return False
            for w in range(n):
                if w != v and adj[u][w]:
                    if common[w][v] >= 2:
                        return False
            for w in range(n):
                if w != u and adj[v][w]:
                    if common[u][w] >= 2:
                        return False
            return True
        
        def do_add(u, v):
            edges.append([u, v])
            adj[u][v] = adj[v][u] = True
            for w in range(n):
                if w != v and adj[u][w]:
                    common[w][v] += 1
                    common[v][w] += 1
            for w in range(n):
                if w != u and adj[v][w]:
                    common[u][w] += 1
                    common[w][u] += 1
        
        def do_remove(u, v):
            # Remove edge from list
            for i, e in enumerate(edges):
                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                    edges.pop(i)
                    break
            adj[u][v] = adj[v][u] = False
            for w in range(n):
                if w != v and adj[u][w]:
                    common[w][v] -= 1
                    common[v][w] -= 1
            for w in range(n):
                if w != u and adj[v][w]:
                    common[u][w] -= 1
                    common[w][u] -= 1
        
        # Local search loop
        max_iterations = 5000
        for it in range(max_iterations):
            if time.time() > deadline - 3.0:
                break
            
            # Try to add a valid edge
            added = False
            for _ in range(20):  # Try up to 20 random edge additions
                u, v = random.choice(all_edges)
                if can_add(u, v):
                    do_add(u, v)
                    current_count += 1
                    added = True
                    if current_count > best_local_count:
                        best_local_count = current_count
                        best_local_edges = [e[:] for e in edges]
                    break
            
            if not added:
                # Stuck: remove a random edge to escape
                if edges:
                    e = random.choice(edges)
                    do_remove(e[0], e[1])
                    current_count -= 1
        
        return best_local_edges, best_local_count
    
    # Phase 1: Run many greedy constructions to find good starting points
    greedy_results = []
    for seed in range(200):
        if time.time() > deadline - 30.0:
            break
        result = _greedy_construct(seed)
        if len(result) > best_count:
            best_count = len(result)
            best_edges = result[:]
        greedy_results.append((len(result), result))
    
    # Phase 2: Local search from the best few greedy results
    greedy_results.sort(key=lambda x: -x[0])
    
    for i in range(min(5, len(greedy_results))):
        if time.time() > deadline - 5.0:
            break
        _, start_edges = greedy_results[i]
        ls_edges, ls_count = _local_search(start_edges, seed=1000 + i)
        if ls_count > best_count:
            best_count = ls_count
            best_edges = ls_edges[:]
    
    # Convert to list of lists [u, v]
    return [[u, v] for u, v in best_edges]
