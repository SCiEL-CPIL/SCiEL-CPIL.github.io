import itertools
import random
import time

def _is_regular_subgraph(vertices, adj_matrix, n):
    """Check if the induced subgraph on 'vertices' is regular."""
    k = len(vertices)
    if k < 2:
        return True
    # Compute degrees within the induced subgraph
    degs = []
    for v in vertices:
        d = 0
        for u in vertices:
            if u != v and adj_matrix[v][u]:
                d += 1
        degs.append(d)
    # Check if all degrees are equal
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def construct():
    def solve(n, time_limit=115):
        start_time = time.time()
        random.seed(42)
        
        # Initialize with random graph
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Precompute all size-6 subsets
        subsets6 = list(itertools.combinations(range(n), 6))
        
        # Precompute for each edge, which size-6 subsets contain it
        edge_to_subsets6 = { (min(i,j), max(i,j)): [] for i in range(n) for j in range(i+1, n) }
        for sub in subsets6:
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = sub[i], sub[j]
                    edge_to_subsets6[(min(u,v), max(u,v))].append(sub)
        
        def count_reg6(adj):
            count = 0
            for sub in subsets6:
                if _is_regular_subgraph(sub, adj, n):
                    count += 1
            return count
        
        def check_large_regular(adj, n, max_k=10):
            """Check for regular subsets of size > 6. Returns a violating subset or None."""
            for k in range(7, min(n, max_k) + 1):
                for sub in itertools.combinations(range(n), k):
                    if _is_regular_subgraph(sub, adj, n):
                        return sub
            return None
            
        def is_valid(adj, n):
            if count_reg6(adj) > 0:
                return False
            if check_large_regular(adj, n) is not None:
                return False
            return True

        current_violations = count_reg6(adj)
        
        for iter in range(50000):
            if time.time() - start_time > time_limit:
                break
                
            if current_violations == 0:
                large_viol = check_large_regular(adj, n)
                if large_viol is None:
                    return adj, True
                
                best_edge = None
                best_delta = float('inf')
                for i in range(len(large_viol)):
                    for j in range(i+1, len(large_viol)):
                        u, v = large_viol[i], large_viol[j]
                        key = (min(u,v), max(u,v))
                        delta = 0
                        for sub in edge_to_subsets6[key]:
                            was_reg = _is_regular_subgraph(sub, adj, n)
                            adj[u][v] ^= 1
                            adj[v][u] ^= 1
                            is_reg_now = _is_regular_subgraph(sub, adj, n)
                            adj[u][v] ^= 1
                            adj[v][u] ^= 1
                            
                            if was_reg and not is_reg_now:
                                delta -= 1
                            elif not was_reg and is_reg_now:
                                delta += 1
                        if delta < best_delta:
                            best_delta = delta
                            best_edge = (u, v)
                
                if best_edge:
                    u, v = best_edge
                    adj[u][v] ^= 1
                    adj[v][u] ^= 1
                    current_violations += best_delta
                else:
                    u, v = random.sample(range(n), 2)
                    adj[u][v] ^= 1
                    adj[v][u] ^= 1
                    current_violations = count_reg6(adj)
                continue

            violating_edges = set()
            for sub in subsets6:
                if _is_regular_subgraph(sub, adj, n):
                    for i in range(6):
                        for j in range(i+1, 6):
                            u, v = sub[i], sub[j]
                            violating_edges.add((min(u,v), max(u,v)))
            
            if not violating_edges:
                break

            candidates = list(violating_edges)
            random.shuffle(candidates)
            candidates = candidates[:50]
            
            best_delta = 0
            best_edge = None
            
            for edge in candidates:
                u, v = edge
                delta = 0
                for sub in edge_to_subsets6[edge]:
                    was_reg = _is_regular_subgraph(sub, adj, n)
                    adj[u][v] ^= 1
                    adj[v][u] ^= 1
                    is_reg_now = _is_regular_subgraph(sub, adj, n)
                    adj[u][v] ^= 1
                    adj[v][u] ^= 1
                    
                    if was_reg and not is_reg_now:
                        delta -= 1
                    elif not was_reg and is_reg_now:
                        delta += 1
                
                if delta < best_delta:
                    best_delta = delta
                    best_edge = edge
            
            if best_edge:
                u, v = best_edge
                adj[u][v] ^= 1
                adj[v][u] ^= 1
                current_violations += best_delta
            else:
                for _ in range(5):
                    u, v = random.sample(range(n), 2)
                    adj[u][v] ^= 1
                    adj[v][u] ^= 1
                current_violations = count_reg6(adj)
                
        return adj, False

    for n in [21, 20]:
        adj, success = solve(n, time_limit=50)
        if success:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    return {'n': 5, 'edges': []}
