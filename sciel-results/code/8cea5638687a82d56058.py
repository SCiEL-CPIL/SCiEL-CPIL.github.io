import random
import time
import math

def construct():
    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_violations(adj, deg):
        # adj is n x n matrix
        # deg is list of n ints
        # Calculate A^3
        # A^2[i][j] = sum_k A[i][k] A[k][j]
        # A^3[i][j] = sum_k A[i][k] A^2[k][j]
        
        # Since n is small, we can just do it directly
        # But we only need to check pairs (s, t) where deg[s] == deg[t]
        
        violations = 0
        # Precompute A^2?
        # A2 = [[0]*n for _ in range(n)]
        # for i in range(n):
        #     for k in range(n):
        #         if adj[i][k]:
        #             for j in range(n):
        #                 if adj[k][j]:
        #                     A2[i][j] += 1
        # 
        # Then for s, t:
        # if deg[s] == deg[t] and s != t:
        #     count = 0
        #     for u in range(n):
        #         if adj[s][u]:
        #             count += A2[u][t]
        #     # Subtract non-simple paths
        #     if adj[s][t]:
        #         count -= (deg[s] + deg[t] - 1)
        #     if count > 0:
        #         violations += 1
        # return violations

        # Optimized:
        # Calculate A2
        A2 = [[0]*n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                if adj[i][k]:
                    for j in range(n):
                        if adj[k][j]:
                            A2[i][j] += 1
        
        for s in range(n):
            for t in range(s+1, n):
                if deg[s] == deg[t]:
                    count = 0
                    for u in range(n):
                        if adj[s][u]:
                            count += A2[u][t]
                    if adj[s][t]:
                        count -= (deg[s] + deg[t] - 1)
                    if count > 0:
                        violations += 1
        return violations

    def solve():
        random.seed(int(time.time() * 1000) % 10000)
        
        # Initialize
        adj = [[False]*n for _ in range(n)]
        deg = [0]*n
        edges = set()
        
        # Start with random graph
        for e in all_edges:
            if random.random() < 0.5:
                u, v = e
                adj[u][v] = True
                adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
                edges.add(e)
        
        current_violations = get_violations(adj, deg)
        current_edges = len(edges)
        
        best_edges = set(edges)
        best_violations = current_violations
        best_count = current_edges
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        start_time = time.time()
        time_limit = 55.0
        
        while T > min_T and (time.time() - start_time) < time_limit:
            # Generate move
            r = random.random()
            if r < 0.4:
                # Add edge
                non_edges = [e for e in all_edges if e not in edges]
                if not non_edges: continue
                e_add = random.choice(non_edges)
                e_rem = None
            elif r < 0.8:
                # Remove edge
                if not edges: continue
                e_rem = random.choice(list(edges))
                e_add = None
            else:
                # Swap
                if not edges: continue
                non_edges = [e for e in all_edges if e not in edges]
                if not non_edges: continue
                e_rem = random.choice(list(edges))
                e_add = random.choice(non_edges)
            
            # Apply
            new_adj = [row[:] for row in adj]
            new_deg = deg[:]
            new_edges = set(edges)
            
            if e_rem:
                u, v = e_rem
                new_adj[u][v] = False
                new_adj[v][u] = False
                new_deg[u] -= 1
                new_deg[v] -= 1
                new_edges.remove(e_rem)
            
            if e_add:
                u, v = e_add
                new_adj[u][v] = True
                new_adj[v][u] = True
                new_deg[u] += 1
                new_deg[v] += 1
                new_edges.add(e_add)
            
            new_violations = get_violations(new_adj, new_deg)
            new_count = len(new_edges)
            
            # Acceptance
            accept = False
            if new_violations < current_violations:
                accept = True
            elif new_violations == current_violations:
                if new_count > current_edges:
                    accept = True
                elif new_count == current_edges:
                    accept = random.random() < 0.5
            else:
                delta = new_violations - current_violations
                if random.random() < math.exp(-delta / T):
                    accept = True
            
            if accept:
                adj = new_adj
                deg = new_deg
                edges = new_edges
                current_violations = new_violations
                current_edges = new_count
                
                if current_violations < best_violations or (current_violations == best_violations and current_edges > best_count):
                    best_violations = current_violations
                    best_count = current_edges
                    best_edges = set(edges)
            
            T *= cooling
        
        return list(best_edges)

    best_overall = []
    start_time = time.time()
    
    # Run multiple instances
    while time.time() - start_time < 50:
        res = solve()
        # Verify validity
        if len(res) > len(best_overall):
            # Quick check
            adj = [[False]*n for _ in range(n)]
            deg = [0]*n
            for u, v in res:
                adj[u][v] = True
                adj[v][u] = True
                deg[u] += 1
                deg[v] += 1
            if get_violations(adj, deg) == 0:
                best_overall = res
    
    final_edges = []
    for u, v in best_overall:
        if isinstance(u, bool) or isinstance(v, bool):
            continue
        if not isinstance(u, int) or not isinstance(v, int):
            continue
        final_edges.append([int(u), int(v)])
        
    return final_edges
