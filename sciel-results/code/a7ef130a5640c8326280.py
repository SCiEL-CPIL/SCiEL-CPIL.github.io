import itertools
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Vertices 0..14
    # Base graph: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add(tuple(sorted((a, b))))
    
    # Precompute all possible intra-part edges
    candidates = []
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            candidates.append((B[i], B[j]))
    
    # Build adjacency and degree tracking for fast validation
    def build_adj(deg, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def check_valid_fast(deg, adj, edges):
        """Check if no path s-u-v-t has deg[s]==deg[t]"""
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    def is_valid_edges(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        adj = build_adj(deg, edges)
        return check_valid_fast(deg, adj, edges)
    
    # Greedy with multiple orderings
    def greedy_add(base_edge_set, cand_list):
        curr = set(base_edge_set)
        deg = [0] * n
        for u, v in curr:
            deg[u] += 1
            deg[v] += 1
        adj = build_adj(deg, curr)
        
        for c in cand_list:
            if c in curr:
                continue
            u, v = c
            # Quick check: if deg[u]==deg[v] and they share neighbors, likely bad
            # Just try adding
            new_deg = deg[:]
            new_deg[u] += 1
            new_deg[v] += 1
            
            # Build new adjacency incrementally
            new_adj = [s.copy() for s in adj]
            new_adj[u].add(v)
            new_adj[v].add(u)
            
            if check_valid_fast(new_deg, new_adj, curr | {c}):
                curr.add(c)
                deg = new_deg
                adj = new_adj
        
        return curr
    
    # Local search: try swapping edges
    def local_search(edge_set, max_iters=50):
        curr = set(edge_set)
        deg = [0] * n
        for u, v in curr:
            deg[u] += 1
            deg[v] += 1
        adj = build_adj(deg, curr)
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            # Try adding any candidate
            for c in candidates:
                if c in curr:
                    continue
                u, v = c
                new_deg = deg[:]
                new_deg[u] += 1
                new_deg[v] += 1
                new_adj = [s.copy() for s in adj]
                new_adj[u].add(v)
                new_adj[v].add(u)
                
                if check_valid_fast(new_deg, new_adj, curr | {c}):
                    curr.add(c)
                    deg = new_deg
                    adj = new_adj
                    improved = True
                    break
            
            if not improved:
                # Try removing an edge and adding two others
                # Find edges whose removal allows more additions
                for e in list(curr):
                    if e in base_edges:
                        continue  # Don't remove base edges
                    test_curr = curr - {e}
                    test_deg = deg[:]
                    u, v = e
                    test_deg[u] -= 1
                    test_deg[v] -= 1
                    test_adj = [s.copy() for s in adj]
                    test_adj[u].discard(v)
                    test_adj[v].discard(u)
                    
                    # Try to add back different edges
                    added = 0
                    for c in candidates:
                        if c in test_curr or c == e:
                            continue
                        cu, cv = c
                        td = test_deg[:]
                        td[cu] += 1
                        td[cv] += 1
                        ta = [s.copy() for s in test_adj]
                        ta[cu].add(cv)
                        ta[cv].add(cu)
                        
                        if check_valid_fast(td, ta, test_curr | {c}):
                            test_curr.add(c)
                            test_deg = td
                            test_adj = ta
                            added += 1
                        if added >= 2:
                            break
                    
                    if len(test_curr) > len(curr):
                        curr = test_curr
                        deg = test_deg
                        adj = test_adj
                        improved = True
                        break
        
        return curr
    
    # Initial best solution
    best_edges = set(base_edges)
    best_count = len(best_edges)
    
    # Multiple greedy passes with different orderings
    for restart in range(100):
        if time.time() > deadline:
            break
            
        if restart == 0:
            # First pass: sort by minimum degree endpoints
            ordered = sorted(candidates, key=lambda c: (min(c[0], c[1]), max(c[0], c[1])))
        elif restart % 3 == 1:
            ordered = candidates[:]
            random.shuffle(ordered)
        else:
            # Sort by "centrality" - prefer edges between high-degree potential vertices
            ordered = candidates[:]
            random.shuffle(ordered)
        
        curr = greedy_add(base_edges, ordered)
        
        if len(curr) > best_count:
            best_edges = curr
            best_count = len(curr)
        
        # Local search on the best result
        if best_count >= 56:
            if time.time() > deadline:
                break
            improved = local_search(best_edges, max_iters=30)
            if len(improved) > best_count:
                best_edges = improved
                best_count = len(best_edges)
    
    result = [list(e) for e in best_edges]
    result.sort()
    
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
        
    return result
