import random
import time

def construct():
    deadline = time.time() + 110  # 10s margin
    best_edges = 0
    best_graph = None
    
    n = 20
    
    def is_k33_free(adj):
        # adj is a list of sets
        for i in range(n):
            for j in range(i+1, n):
                # Check triples containing i, j? No, check all triples.
                # Optimization: Precompute common neighbors?
                # For n=20, full check is fast enough.
                pass
        
        # Full check
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    common = adj[i] & adj[j] & adj[k]
                    if len(common) >= 3:
                        return False
        return True

    def repair(adj):
        # adj is list of sets
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            # Find violation
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        common = adj[i] & adj[j] & adj[k]
                        if len(common) >= 3:
                            # Remove an edge. Pick one from common, say c.
                            # Remove (i, c)
                            c = next(iter(common))
                            adj[i].remove(c)
                            adj[c].remove(i)
                            changed = True
                            break
                    if changed: break
                if changed: break
            # If no change in this pass, we are safe? 
            # Removing an edge might create no new violations, only remove them.
            # So one pass of finding and removing is enough if we scan fully?
            # No, removing (i, c) might affect other triples.
            # But it only reduces degrees, so it cannot create a new K3,3.
            # So we just need to find ONE violation and remove an edge.
            # Repeat until no violation.
            pass

    def improve(adj):
        # Try to add edges
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Iterate non-edges
            for i in range(n):
                for j in range(i+1, n):
                    if j not in adj[i]:
                        # Try adding (i, j)
                        adj[i].add(j)
                        adj[j].add(i)
                        
                        # Check validity
                        # Only triples involving i or j can be affected
                        valid = True
                        # Triples with i
                        for a in range(n):
                            if a == i: continue
                            for b in range(a+1, n):
                                if b == i or b == j: continue
                                # Check triple (i, a, b)
                                common = adj[i] & adj[a] & adj[b]
                                if len(common) >= 3:
                                    valid = False
                                    break
                            if not valid: break
                            # Check triple (i, j, b) -- wait, j is neighbor of i now.
                            # The triple is (i, j, b).
                            # common = adj[i] & adj[j] & adj[b]
                            # This is covered if we iterate all pairs (a,b) where a,b != i?
                            # My loop above: a, b are other vertices.
                            # It checks (i, a, b).
                            # It does NOT check (i, j, b) because b != j in the loop condition?
                            # Yes, `if b == j: continue` skips it.
                            # So we must explicitly check triples involving both i and j.
                        
                        if valid:
                            # Check triples (i, j, b) for all b != i, j
                            for b in range(n):
                                if b == i or b == j: continue
                                common = adj[i] & adj[j] & adj[b]
                                if len(common) >= 3:
                                    valid = False
                                    break
                        
                        if valid:
                            improved = True
                        else:
                            adj[i].remove(j)
                            adj[j].remove(i)
        
    # Main loop
    while time.time() < deadline:
        # Initialize random graph
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.3:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Repair
        repair(adj)
        
        # Improve
        improve(adj)
        
        # Count edges
        edges = sum(len(s) for s in adj) // 2
        if edges > best_edges:
            best_edges = edges
            # Store graph
            best_graph = [[u, v] for u in range(n) for v in adj[u] if u < v]
            
    if best_graph is None:
        return []
    return best_graph
