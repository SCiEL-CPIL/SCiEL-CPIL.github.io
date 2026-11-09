import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 125

    def solve(n, seed):
        rng = random.Random(seed)
        
        # adj[i][j] = True if edge exists
        adj = np.zeros((n, n), dtype=bool)
        # common[i][j] = number of common neighbors of i and j
        common = np.zeros((n, n), dtype=np.int32)
        
        def add_edge(u, v):
            """Add edge (u,v) and update common counts. Returns False if it would create C4."""
            if u == v or adj[u][v]:
                return False
            
            # 1. Check if (u,v) creates a C4 with existing edges
            # common[u][v] in the new graph would be the number of common neighbors of u and v
            # Current common neighbors of u and v: np.sum(adj[u] & adj[v])
            # Note: adj[u][v] is currently False, so this is correct.
            if np.sum(adj[u] & adj[v]) > 1:
                return False
            
            # 2. Check if adding (u,v) makes any other pair have > 1 common neighbor
            # For any j, new common[u][j] = common[u][j] + (1 if adj[v][j] else 0)
            # We need new common[u][j] <= 1 for all j
            # This is equivalent to: for all j where adj[v][j] is True, common[u][j] must be 0.
            # And for all j where adj[v][j] is False, common[u][j] must be <= 1.
            # Since common[u][j] is always <= 1 in a valid C4-free graph (for j != u),
            # we just need to check: if adj[v][j] is True, common[u][j] must be 0.
            # Wait, if common[u][j] is 1 and adj[v][j] is True, new common is 2 -> C4.
            # So: np.all(common[u][adj[v]] == 0) ?
            # adj[v] is a boolean mask. common[u][adj[v]] gives the common counts for neighbors of v.
            # We need all these to be 0.
            if np.any(common[u][adj[v]] > 0):
                return False
            
            # Similarly for v
            if np.any(common[v][adj[u]] > 0):
                return False
            
            # Safe to add
            adj[u][v] = adj[v][u] = True
            
            # Update common counts
            # common[u][v] = np.sum(adj[u] & adj[v]) (calculated above, but let's be safe)
            # Actually, we can just update the relevant parts.
            # common[u][j] += adj[v][j] for all j
            # common[v][j] += adj[u][j] for all j
            # common[j][u] += adj[v][j] for all j
            # common[j][v] += adj[u][j] for all j
            
            # Using NumPy broadcasting:
            # common[u, :] += adj[v, :]
            # common[v, :] += adj[u, :]
            # common[:, u] += adj[:, v]
            # common[:, v] += adj[:, u]
            
            # But we must be careful with u and v themselves.
            # common[u][u] and common[v][v] are not used, but let's keep them 0 or consistent.
            # The updates above will increment common[u][v] by adj[v][v] (which is 0) and common[v][u] by adj[u][u] (which is 0).
            # Wait, common[u][v] should become the number of common neighbors of u and v.
            # The formula `common[u, :] += adj[v, :]` adds 1 to common[u][j] if v is neighbor of j.
            # For j=v, adj[v][v] is 0, so common[u][v] is not incremented by this term?
            # No, `adj[v, v]` is 0.
            # So common[u][v] remains what it was?
            # Before adding, common[u][v] was the number of common neighbors of u and v.
            # After adding, u and v are connected.
            # The common neighbors of u and v are still the same set of vertices $w$ such that $w \sim u$ and $w \sim v$.
            # The edge $(u,v)$ does not change the set of common neighbors $w$.
            # So `common[u][v]` should NOT change?
            # Let's re-verify.
            # $N(u) \cap N(v)$.
            # Before: $N(u)$ does not contain $v$. $N(v)$ does not contain $u$.
            # Intersection is $\{w \mid w \sim u, w \sim v, w \neq u, w \neq v\}$.
            # After: $N(u)$ contains $v$. $N(v)$ contains $u$.
            # Intersection is $\{w \mid w \sim u, w \sim v, w \neq u, w \neq v\} \cup \{u \text{ if } u \sim v \text{ and } u \in N(v) \text{ (false, } u \notin N(u) \text{? No, } u \in N(v) \text{ is true. Is } u \in N(u) \text{? No.}\} \}$.
            # Wait, $u \in N(v)$ is True. Is $u \in N(u)$? No. So $u$ is not in $N(u) \cap N(v)$.
            # Similarly $v \notin N(u) \cap N(v)$.
            # So the set of common neighbors is UNCHANGED.
            # Therefore, `common[u][v]` should NOT be updated by the addition of the edge $(u,v)$ itself.
            # However, `common[u][j]` for $j \neq v$ changes because $v$ is added to $N(u)$.
            # If $j$ is a neighbor of $v$, then $v \in N(j)$, so $v$ becomes a common neighbor of $u$ and $j$.
            # So `common[u][j]` increases by 1 if `adj[v][j]` is True.
            # This matches `common[u, :] += adj[v, :]`.
            # For $j=v$, `adj[v][v]` is 0, so `common[u][v]` is not changed. Correct.
            
            common[u, :] += adj[v, :]
            common[v, :] += adj[u, :]
            common[:, u] += adj[:, v]
            common[:, v] += adj[:, u]
            
            # Note: This updates common[u][u] and common[v][v] as well, but we don't care about them.
            # It also updates common[u][v] and common[v][u] by 0.
            
            return True

        # Initialize
        vertices = list(range(n))
        rng.shuffle(vertices)
        
        # Heuristic: Prioritize adding edges to vertices with low degree
        # We can simulate this by shuffling edges, but that's random.
        # Better: Maintain a list of vertices.
        # Simple random greedy first, as it's robust.
        
        edge_list = []
        for i in range(n):
            for j in range(i+1, n):
                edge_list.append((i, j))
        rng.shuffle(edge_list)
        
        for (u, v) in edge_list:
            if time.time() > deadline:
                return None
            add_edge(u, v)
        
        # Check min degree
        degs = np.sum(adj, axis=1)
        md = int(np.min(degs))
        
        if md >= n - 39:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        return None
    
    best_result = None
    
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline - 3:
            break
        
        # Try more seeds since it's faster
        for seed in range(500):
            if time.time() > deadline - 3:
                break
            result = solve(n, seed)
            if result is not None:
                best_result = result
                break
        
        if best_result is not None:
            break
    
    if best_result is None:
        best_result = {'n': 39, 'edges': []}
    
    return best_result
