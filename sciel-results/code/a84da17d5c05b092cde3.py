import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n = 20
    best_edges = None
    best_count = -1
    
    def count_common_neighbors(adj, u, v):
        """Count common neighbors of u and v."""
        return np.sum(adj[u] & adj[v])
    
    def is_k23_free(adj):
        """Check if graph is K2,3-free."""
        for i in range(n):
            for j in range(i+1, n):
                if np.sum(adj[i] & adj[j]) > 2:
                    return False
        return True
    
    def get_violations(adj):
        """Return list of violating pairs (i,j) where common neighbors > 2."""
        violations = []
        for i in range(n):
            for j in range(i+1, n):
                if np.sum(adj[i] & adj[j]) > 2:
                    violations.append((i, j))
        return violations
    
    def try_add_edge(adj, u, v):
        """Try to add edge (u,v). Returns True if valid."""
        if adj[u][v]:
            return False
        # Check if adding this edge would create a K2,3 violation
        # For any pair (a,b), if both a and b are neighbors of u and v,
        # then adding edge (u,v) increases common neighbors of (a,b) by 1
        # Actually, we need to check: after adding (u,v), does any pair have >2 common neighbors?
        # The new edge (u,v) means u and v become adjacent.
        # This affects common neighbor counts for pairs involving u or v.
        # For pair (u, w): common neighbors increase if w is neighbor of v (and w != v)
        # For pair (v, w): common neighbors increase if w is neighbor of u (and w != u)
        # For pair (a, b) where a != u,v and b != u,v: no change
        # Wait, that's not right. Adding edge (u,v) doesn't change N(u) or N(v) in terms of 
        # who they're connected to. It just adds the edge between them.
        # Common neighbors of (u,v) = |N(u) ∩ N(v)| which is unchanged by adding edge (u,v)
        # since the edge (u,v) itself doesn't make u a neighbor of itself or v a neighbor of itself.
        # So adding edge (u,v) only matters if it creates new paths? No...
        # Actually, K2,3 is about the structure. Adding edge (u,v) doesn't change any 
        # common neighbor counts because common neighbors are defined by shared adjacency,
        # and adding (u,v) makes u adjacent to v and v adjacent to u, but this doesn't 
        # change who is adjacent to other vertices.
        # Wait - I think I'm confusing myself. Let me reconsider.
        # N(u) is the set of vertices adjacent to u. Adding edge (u,v) adds v to N(u) and u to N(v).
        # So for pair (u, w) where w != v: common neighbors = |N(u) ∩ N(w)|
        #   Before: N(u) didn't include v (if not adjacent), now it does.
        #   So if v ∈ N(w), then common neighbors of (u,w) increases by 1.
        # For pair (v, w) where w != u: similarly, if u ∈ N(w), common neighbors increase by 1.
        # For pair (u, v): common neighbors = |N(u) ∩ N(v)|, unchanged.
        # For pair (a, b) where a,b not in {u,v}: unchanged.
        
        # Check pair (u, w) for all w != v:
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[v][w]:  # v is neighbor of w
                if np.sum(adj[u] & adj[w]) >= 2:  # already 2 common neighbors, adding v would make 3
                    return False
        # Check pair (v, w) for all w != u:
        for w in range(n):
            if w == u or w == v:
                continue
            if adj[u][w]:  # u is neighbor of w
                if np.sum(adj[v] & adj[w]) >= 2:  # already 2 common neighbors, adding u would make 3
                    return False
        return True
    
    def remove_violating_edges(adj):
        """Remove edges to make graph K2,3-free. Returns new adj and number removed."""
        removed = 0
        adj = adj.copy()
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j] and np.sum(adj[i] & adj[j]) > 2:
                        # Remove this edge
                        adj[i][j] = False
                        adj[j][i] = False
                        removed += 1
                        changed = True
        return adj, removed
    
    def local_search(adj):
        """Perform local search to maximize edges while maintaining K2,3-free."""
        adj = adj.copy()
        # First, remove all violations
        adj, _ = remove_violating_edges(adj)
        
        improved = True
        while improved:
            improved = False
            # Try adding each missing edge
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        if try_add_edge(adj, i, j):
                            adj[i][j] = True
                            adj[j][i] = True
                            improved = True
            # Try edge swaps: remove edge (a,b) and add edge (c,d)
            # This is more complex, so let's also try: for each edge, see if removing it
            # allows adding two other edges
            for a in range(n):
                for b in range(a+1, n):
                    if adj[a][b]:
                        # Try removing (a,b) and see if we can add more edges
                        adj2 = adj.copy()
                        adj2[a][b] = False
                        adj2[b][a] = False
                        # Now try to add edges
                        added = 0
                        for i in range(n):
                            for j in range(i+1, n):
                                if not adj2[i][j]:
                                    if try_add_edge(adj2, i, j):
                                        adj2[i][j] = True
                                        adj2[j][i] = True
                                        added += 1
                        if added > 1:  # net gain
                            adj = adj2
                            improved = True
        return adj
    
    def run_one_trial(seed_val):
        """Run one construction trial."""
        rng = random.Random(seed_val)
        
        # Start with a random graph with moderate density
        # Try different densities
        densities = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_local = None
        best_local_count = -1
        
        for density in densities:
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < density:
                        adj[i][j] = True
                        adj[j][i] = True
            
            # Remove violations
            adj, _ = remove_violating_edges(adj)
            
            # Local search
            adj = local_search(adj)
            
            edge_count = int(np.sum(adj)) // 2
            if edge_count > best_local_count:
                best_local_count = edge_count
                best_local = adj.copy()
        
        return best_local, best_local_count
    
    # Run multiple trials
    trial_count = 0
    while time.time() < deadline:
        trial_count += 1
        adj, count = run_one_trial(trial_count)
        if adj is not None and count > best_count:
            best_count = count
            best_edges = adj.copy()
            # If we've found a good solution, we can do more focused search
            if best_count >= 60:
                # Do more local search from this point
                pass
    
    # Convert best adjacency matrix to edge list
    if best_edges is None:
        # Fallback: return the seed solution
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_edges[i][j]:
                edges.append([i, j])
    
    return edges
