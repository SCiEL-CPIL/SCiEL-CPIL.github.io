import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 18
    
    # Helper: check if adding edge (u,v) keeps graph K4-free
    # A K4 exists if there's a set {a,b,c,d} all pairwise connected
    # When adding edge (u,v), a new K4 must include both u and v
    # So check: is there a pair (w,x) such that w,x are both connected to u and v, and w-x edge exists?
    # Actually: K4 = {u,v,w,x} where all 6 edges exist. Since we're adding (u,v),
    # we need w,x such that: w-x edge, u-w edge, u-x edge, v-w edge, v-x edge all exist.
    
    def is_k4_free_with_edge(adj, u, v):
        """Check if adding edge (u,v) to graph with adjacency adj creates a K4."""
        # Find common neighbors of u and v
        common = adj[u] & adj[v]
        # Check if any two common neighbors are connected to each other
        for w in common:
            for x in common:
                if w < x and (x in adj[w]):
                    return False
        return True
    
    def max_triangle_free_size(adj):
        """Find the largest set S such that G[S] is triangle-free."""
        n = len(adj)
        best = 0
        # Use branch and bound
        # A set is triangle-free if for every vertex in the set, 
        # its neighbors within the set form an independent set (no edges among them)
        
        # We'll use a recursive search with pruning
        # State: current set, candidate vertices to add
        # Pruning: if len(current) + len(candidates) <= best, prune
        
        def dfs(current, candidates, depth):
            nonlocal best
            if len(current) + len(candidates) <= best:
                return
            if len(candidates) == 0:
                if len(current) > best:
                    best = len(current)
                return
            
            # Try adding the first candidate
            v = candidates[0]
            remaining = candidates[1:]
            
            # Option 1: include v
            # v can be included if it doesn't form a triangle with any two in current
            # i.e., no two vertices in current are both neighbors of v
            can_include = True
            neighbors_in_current = [u for u in current if u in adj[v]]
            # Check if neighbors_in_current has any edge
            for i in range(len(neighbors_in_current)):
                for j in range(i+1, len(neighbors_in_current)):
                    if neighbors_in_current[j] in adj[neighbors_in_current[i]]:
                        can_include = False
                        break
                if not can_include:
                    break
            
            if can_include:
                new_current = current + [v]
                # New candidates: remaining vertices that can still be added
                # A vertex w can be added if it doesn't form a triangle with any two in new_current
                new_candidates = []
                for w in remaining:
                    ok = True
                    nbrs = [u for u in new_current if u in adj[w]]
                    for i in range(len(nbrs)):
                        for j in range(i+1, len(nbrs)):
                            if nbrs[j] in adj[nbrs[i]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        new_candidates.append(w)
                dfs(new_current, new_candidates, depth + 1)
            
            # Option 2: exclude v
            dfs(current, remaining, depth + 1)
        
        dfs([], list(range(n)), 0)
        return best
    
    # Initialize with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    adj = [set() for _ in range(n)]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a].add(b)
                    adj[b].add(a)
    
    best_score = max_triangle_free_size(adj)
    best_adj = [set(s) for s in adj]
    
    if time.time() > deadline:
        # Return best found
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    # Local search: try to add edges within parts to reduce triangle-free set size
    random.seed(42)
    
    # Try adding edges within parts
    for part in parts:
        for u, v in itertools.combinations(part, 2):
            if u not in adj[v]:
                if is_k4_free_with_edge(adj, u, v):
                    adj[u].add(v)
                    adj[v].add(u)
                    new_score = max_triangle_free_size(adj)
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = [set(s) for s in adj]
                    else:
                        # Revert
                        adj[u].remove(v)
                        adj[v].remove(u)
    
    if time.time() > deadline:
        edges = []
        for u in range(n):
            for v in best_adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
    
    # Try removing some cross-part edges to allow more within-part edges
    # Actually, let's try a different approach: random restarts with local search
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Start from best_adj and perturb
        adj = [set(s) for s in best_adj]
        
        # Try random edge additions within parts
        for part in parts:
            for u, v in itertools.combinations(part, 2):
                if u not in adj[v]:
                    if is_k4_free_with_edge(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        new_score = max_triangle_free_size(adj)
                        if new_score < best_score:
                            best_score = new_score
                            best_adj = [set(s) for s in adj]
                        else:
                            adj[u].remove(v)
                            adj[v].remove(u)
        
        # Try removing some cross edges to enable more within-part edges
        # This is complex, so let's try a different strategy:
        # Randomly remove a cross-part edge, then try to add within-part edges
        
        adj = [set(s) for s in best_adj]
        # Pick a random cross-part edge to remove
        cross_edges = []
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        if b in adj[a]:
                            cross_edges.append((a, b))
        
        if cross_edges:
            a, b = random.choice(cross_edges)
            adj[a].remove(b)
            adj[b].remove(a)
            
            # Now try to add within-part edges
            for part in parts:
                for u, v in itertools.combinations(part, 2):
                    if u not in adj[v]:
                        if is_k4_free_with_edge(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
            
            new_score = max_triangle_free_size(adj)
            if new_score < best_score:
                best_score = new_score
                best_adj = [set(s) for s in adj]
    
    # Return the best graph found
    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
