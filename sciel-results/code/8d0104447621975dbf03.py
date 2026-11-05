import random
import itertools
import time

def construct():
    n = 20
    all_edges = list(itertools.combinations(range(n), 2))
    edge_set = set(all_edges)
    
    def is_valid_add(adj, u, v):
        # Check if adding (u,v) creates a K2,3
        # 1. Common neighbors of u and v must be < 3
        if len(adj[u] & adj[v]) >= 3:
            return False
        # 2. For any w in adj[u], common neighbors of v and w must be < 3
        #    (Note: u is a common neighbor of v and w now)
        for w in adj[u]:
            if w == v: continue
            # Common neighbors of v and w include u.
            # So old common neighbors of v and w must be < 2.
            if len(adj[v] & adj[w]) >= 2:
                return False
        # 3. For any w in adj[v], common neighbors of u and w must be < 3
        #    (Note: v is a common neighbor of u and w now)
        for w in adj[v]:
            if w == u: continue
            if len(adj[u] & adj[w]) >= 2:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u].add(v)
        adj[v].add(u)

    def remove_edge(adj, u, v):
        adj[u].discard(v)
        adj[v].discard(u)

    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    def greedy_build():
        adj = [set() for _ in range(n)]
        edges = list(all_edges)
        random.shuffle(edges)
        for u, v in edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
        return adj

    def local_search(adj, max_iters=5000):
        current_edges = count_edges(adj)
        best_adj = [set(s) for s in adj]
        best_edges = current_edges
        
        for _ in range(max_iters):
            # Try to add an edge
            improved = False
            # Randomly pick a few missing edges to try
            current_edge_set = set()
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        current_edge_set.add((u, v))
            
            missing = list(edge_set - current_edge_set)
            if missing:
                random.shuffle(missing)
                # Try first few missing edges
                for u, v in missing[:10]:
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        improved = True
                        break # Restart search after one improvement? Or continue?
                        # Usually better to continue adding if possible in one pass
                        # But checking all missing edges is O(N^2). 
                        # Let's just try to add as many as possible in one pass.
            
            # If no edge was added, try to remove one and re-add
            if not improved:
                if current_edges == 0:
                    break
                # Pick a random edge to remove
                # Prefer edges that are "central" or part of many potential conflicts?
                # For now, random.
                all_current = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            all_current.append((u, v))
                
                if not all_current:
                    break
                            
                u, v = random.choice(all_current)
                remove_edge(adj, u, v)
                current_edges -= 1
                
                # Now try to add edges again
                missing = list(edge_set - { (a,b) if a<b else (b,a) for a,b in [(x,y) for x in range(n) for y in adj[x] if x<y] })
                # Simpler: just iterate all missing
                added_any = False
                for eu, ev in all_edges:
                    if (eu, ev) in current_edge_set: continue
                    if (ev, eu) in current_edge_set: continue
                    if is_valid_add(adj, eu, ev):
                        add_edge(adj, eu, ev)
                        current_edges += 1
                        added_any = True
                
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = [set(s) for s in adj]
                elif current_edges < best_edges:
                    # Revert? Or keep exploring?
                    # Hill climbing usually reverts if worse.
                    # But we removed 1 and added k. If k < 1, we are worse.
                    # Let's revert to best_adj if we want strict hill climbing.
                    # However, to escape local minima, we might want to accept worse moves.
                    # Given time limits, let's stick to strict hill climbing or simulated annealing.
                    # Let's just keep the current state for now, but track best.
                    # Actually, if we don't revert, we might drift.
                    # Let's revert to the state before the removal if no net gain.
                    # But we already modified adj.
                    # This is messy. Let's simplify.
                    
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = [set(s) for s in adj]
                
        return best_adj, best_edges

    best_edges_count = 0
    best_final_adj = None
    
    # Run multiple restarts
    start_time = time.time()
    restarts = 0
    while time.time() - start_time < 110:
        adj = greedy_build()
        adj, edges = local_search(adj)
        if edges > best_edges_count:
            best_edges_count = edges
            best_final_adj = adj
        restarts += 1
        
    # Convert best_final_adj to list of edges
    edges = []
    if best_final_adj:
        for u in range(n):
            for v in best_final_adj[u]:
                if u < v:
                    edges.append([u, v])
    else:
        # Fallback to simple greedy if local search failed
        adj = greedy_build()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])

    return edges
