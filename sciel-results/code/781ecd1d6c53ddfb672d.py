import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initialize with complete graph
    # Edge representation: (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    E = len(all_edges)
    
    # State: set of active edges
    # We will represent the graph as a set of edges.
    
    def get_degrees(edges):
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        return deg

    def get_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid(edges):
        deg = get_degrees(edges)
        adj = get_adj(edges)
        # Check all paths s-u-v-t
        # Iterate over middle edges (u, v)
        for u, v in edges:
            # s is neighbor of u, s != v
            # t is neighbor of v, t != u
            # s != t
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def find_violation(edges):
        deg = get_degrees(edges)
        adj = get_adj(edges)
        for u, v in edges:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return (s, u, v, t)
        return None

    best_edges = set()
    best_count = 0
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start with complete graph
        curr_edges = set(all_edges)
        
        # Phase 1: Remove edges to make valid
        # We remove edges that are part of violations.
        max_removals = 200
        for _ in range(max_removals):
            if time.time() > deadline:
                break
            
            violation = find_violation(curr_edges)
            if violation is None:
                break
            
            s, u, v, t = violation
            # Candidate edges to remove:
            # Edges on the path: (s,u), (u,v), (v,t)
            # Edges incident to s or t (to change degrees)
            candidates = set()
            candidates.add(tuple(sorted((s, u))))
            candidates.add(tuple(sorted((u, v))))
            candidates.add(tuple(sorted((v, t))))
            
            # Add some random edges incident to s or t
            adj_s = [e for e in curr_edges if s in e]
            adj_t = [e for e in curr_edges if t in e]
            
            if adj_s:
                candidates.add(random.choice(adj_s))
            if adj_t:
                candidates.add(random.choice(adj_t))
            
            # Pick a random candidate to remove
            # Prefer edges on the path? Or random?
            # Let's pick randomly from candidates
            edge_to_remove = random.choice(list(candidates))
            
            if edge_to_remove in curr_edges:
                curr_edges.remove(edge_to_remove)
        
        if is_valid(curr_edges):
            if len(curr_edges) > best_count:
                best_edges = curr_edges
                best_count = len(curr_edges)
        
        # Phase 2: Local search - try to add edges back
        # Iterate through all possible edges not in curr_edges
        # If adding it keeps it valid, add it.
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Shuffle to break symmetry
            missing_edges = [e for e in all_edges if e not in curr_edges]
            random.shuffle(missing_edges)
            
            for e in missing_edges:
                if time.time() > deadline:
                    break
                new_edges = curr_edges | {e}
                if is_valid(new_edges):
                    curr_edges = new_edges
                    improved = True
                    # Restart the loop to check other edges again? 
                    # Or just continue? Continuing is faster but might miss combos.
                    # Let's break and restart the while loop for thoroughness if we have time.
                    # Actually, just continuing is a greedy pass.
        
        if len(curr_edges) > best_count:
            best_edges = curr_edges
            best_count = len(curr_edges)
            
        # Phase 3: Perturbation
        # Remove a random edge from best, re-run phase 2?
        # Or just rely on restarts.
        
    result = [list(e) for e in best_edges]
    result.sort()
    
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])
        
    return result
