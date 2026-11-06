import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n = 18
    # Start with K6,6,6
    # Parts: 0-5, 6-11, 12-17
    adj = np.zeros((n, n), dtype=bool)
    parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a][b] = adj[b][a] = True
    
    def is_k4_free(adj):
        n = len(adj)
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]: continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True
    
    def max_triangle_free_induced(adj):
        """Exact branch and bound for max triangle-free induced subgraph size."""
        n = len(adj)
        best = 0
        # Use a greedy coloring bound for triangle-free (independent set in triangle graph)
        # Actually, triangle-free induced subgraph = independent set in the "triangle conflict" graph?
        # No. A set S is triangle-free if no three vertices in S form a triangle in G.
        # This is not the same as independent set.
        
        # For n=18, use branch and bound
        # State: current set, candidates
        # A vertex can be added if it doesn't form a triangle with any two existing vertices
        
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(current_size, current_set, candidates):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, current_size)
                return
            # Greedy bound: at most current_size + len(candidates)
            # Try adding or not adding the first candidate
            v = candidates[0]
            remaining = candidates[1:]
            
            # Option 1: include v
            # Check if v forms a triangle with any pair in current_set
            can_add = True
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    u1 = current_set[i]
                    u2 = current_set[j]
                    if adj[v][u1] and adj[v][u2] and adj[u1][u2]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                new_current = current_set + [v]
                # Filter candidates: remove those that would form triangle with v and any in new_current
                new_candidates = []
                for w in remaining:
                    ok = True
                    for u in new_current:
                        if adj[v][u] and adj[v][w] and adj[u][w]:
                            ok = False
                            break
                        if adj[v][u] and adj[u][w] and adj[v][w]:
                            ok = False
                            break
                    # Actually need: w doesn't form triangle with any pair in new_current
                    # Simpler: for each pair (x,y) in new_current, check if (v,x,y) or (w,x,y) or (v,w,x) etc forms triangle
                    # This is getting complex. Let me just check all pairs in new_current + [w]
                    new_candidates.append(w)  # We'll filter in the recursive call implicitly
                # Actually, let me just pass remaining and let the recursion handle it
                dfs(current_size + 1, new_current, remaining)
            
            # Option 2: exclude v
            dfs(current_size, current_set, remaining)
        
        dfs(0, [], list(range(n)))
        return best
    
    # This exact method is too slow for repeated calls. Use a greedy estimate.
    def greedy_triangle_free_size(adj):
        """Greedy estimate of max triangle-free induced subgraph size."""
        n = len(adj)
        # Greedily build a triangle-free set
        # At each step, add a vertex that doesn't create a triangle
        # Try multiple random orderings
        best = 0
        for _ in range(20):
            order = list(range(n))
            random.shuffle(order)
            current = []
            for v in order:
                can_add = True
                for i in range(len(current)):
                    for j in range(i+1, len(current)):
                        if adj[v][current[i]] and adj[v][current[j]] and adj[current[i]][current[j]]:
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    current.append(v)
            best = max(best, len(current))
        return best
    
    # Better: use a simple exact method with memoization or better pruning
    def exact_max_tf(adj, timeout=2.0):
        """Exact max triangle-free induced subgraph with timeout."""
        n = len(adj)
        best = 0
        start = time.time()
        
        # Precompute: for each pair (i,j), which vertices k form a triangle with i,j
        # tri_conflict[i][j] = set of k such that i,j,k form a triangle
        tri_conflict = [[set() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(n):
                    if k != i and k != j and adj[i][k] and adj[j][k]:
                        tri_conflict[i][j].add(k)
                        tri_conflict[j][i].add(k)
        
        # Branch and bound
        def dfs(current, candidates):
            nonlocal best
            if time.time() - start > timeout:
                return
            if len(current) + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, len(current))
                return
            
            # Try to add a vertex from candidates
            # Pick the vertex with most constraints
            v = candidates[0]
            remaining = candidates[1:]
            
            # Check if v can be added to current
            can_add = True
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    if v in tri_conflict[current[i]][current[j]]:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                new_current = current + [v]
                # Filter remaining: remove vertices that conflict with v and any pair in new_current
                new_remaining = []
                for w in remaining:
                    ok = True
                    for u in new_current:
                        for w2 in new_current:
                            if u == w2: continue
                            if w in tri_conflict[u][w2]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        new_remaining.append(w)
                dfs(new_current, new_remaining)
            
            dfs(current, remaining)
        
        dfs([], list(range(n)))
        return best
    
    # Use greedy for search, exact for final verification
    current_adj = adj.copy()
    current_score = exact_max_tf(current_adj, timeout=3.0)
    
    def try_edge_change(adj, u, v, add):
        """Try adding or removing edge (u,v), return new adj if K4-free, else None."""
        new_adj = adj.copy()
        if add:
            new_adj[u][v] = new_adj[v][u] = True
        else:
            new_adj[u][v] = new_adj[v][u] = False
        
        # Check K4-freeness locally: only need to check K4s involving the changed edge
        if add:
            # Check if adding (u,v) creates a K4
            common = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
            for i in range(len(common)):
                for j in range(i+1, len(common)):
                    if adj[common[i]][common[j]]:
                        return None  # K4 formed
        else:
            # Removing edge can't create K4
            pass
        
        return new_adj
    
    # Simulated annealing
    T = 1.0
    cooling = 0.999
    best_adj = current_adj.copy()
    best_score = current_score
    
    while time.time() < deadline:
        # Pick random edge to flip
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        add = not current_adj[u][v]
        new_adj = try_edge_change(current_adj, u, v, add)
        if new_adj is None:
            T *= cooling
            continue
        
        new_score = greedy_triangle_free_size(new_adj)
        
        if new_score < current_score or random.random() < np.exp(-(new_score - current_score) / max(T, 0.01)):
            current_adj = new_adj
            current_score = new_score
            if new_score < best_score:
                best_adj = new_adj.copy()
                best_score = new_score
                # Verify exact
                exact_score = exact_max_tf(best_adj, timeout=2.0)
                if exact_score < best_score:
                    best_score = exact_score
        
        T *= cooling
    
    # Return best
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
