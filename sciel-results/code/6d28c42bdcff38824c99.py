import time
import random

def _common_neighbor_count(adj, u, v):
    """Return number of common neighbors of u and v using bitmasks."""
    return bin(adj[u] & adj[v]).count('1')

def _is_valid(adj):
    """Check if graph is K2,3-free."""
    n = len(adj)
    for u in range(n):
        for v in range(u + 1, n):
            if _common_neighbor_count(adj, u, v) > 2:
                return False
    return True

def _edge_count(adj):
    """Count edges in the graph."""
    return sum(bin(x).count('1') for x in adj) // 2

def _repair(adj):
    """Remove edges until the graph is K2,3-free."""
    n = len(adj)
    changed = True
    while changed:
        changed = False
        # Find a violating pair
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                count = bin(common).count('1')
                if count > 2:
                    # We need to reduce the number of common neighbors.
                    # Pick a random common neighbor w and remove edge (u, w) or (v, w).
                    # Removing (u, w) reduces common neighbors of (u, v) by 1?
                    # Yes, because w is no longer in N(u).
                    # It might also fix other pairs or break others? 
                    # Removing an edge can only reduce common neighbor counts for pairs involving u or w.
                    # It cannot create new violations (increasing common neighbors).
                    # So this is a safe repair step.
                    
                    # Get list of common neighbors
                    common_list = [i for i in range(n) if (common >> i) & 1]
                    w = random.choice(common_list)
                    
                    # Remove edge (u, w)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    changed = True
                    break
            if changed:
                break
    return adj

def _local_search(adj, deadline):
    """Try to add edges greedily, with random restarts/kicks if stuck."""
    n = len(adj)
    best_edges = _edge_count(adj)
    best_adj = adj[:]
    
    while time.time() < deadline:
        # Try to add any possible edge
        added_any = False
        # Shuffle vertex pairs for randomness
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        
        for u, v in pairs:
            # Check if edge exists
            if (adj[u] >> v) & 1:
                continue
            
            # Check if adding (u, v) is valid
            # 1. Common neighbors of u and v must be <= 2
            # Currently, common neighbors are adj[u] & adj[v].
            # If we add (u, v), u becomes a neighbor of v and v of u.
            # The new common neighbors of u and v are (adj[u] & adj[v]) plus potentially u and v themselves?
            # No, common neighbors are vertices w != u, v such that w in N(u) and w in N(v).
            # Adding (u, v) does NOT change N(u) \ {v} or N(v) \ {u} regarding other vertices.
            # Wait, N(u) gains v. N(v) gains u.
            # Common neighbors of u and v:
            # Old: (adj[u] & adj[v])
            # New: (adj[u] | (1<<v)) & (adj[v] | (1<<u))
            #      = (adj[u] & adj[v]) | (adj[u] & (1<<u)) | ((1<<v) & adj[v]) | ((1<<v) & (1<<u))
            #      = (adj[u] & adj[v]) | 0 | 0 | 0  (since u not in adj[u], v not in adj[v])
            # So the set of common neighbors does NOT change when adding an edge between two non-adjacent vertices?
            # Let's verify.
            # N'(u) = N(u) U {v}
            # N'(v) = N(v) U {u}
            # N'(u) cap N'(v) = (N(u) U {v}) cap (N(v) U {u})
            # = (N(u) cap N(v)) U (N(u) cap {u}) U ({v} cap N(v)) U ({v} cap {u})
            # = (N(u) cap N(v)) U empty U empty U empty
            # = N(u) cap N(v)
            # So, the number of common neighbors of u and v remains the same!
            
            # However, adding (u, v) affects common neighbors of (u, w) for all w != v.
            # N'(u) = N(u) U {v}.
            # For any w != u, v:
            # Common neighbors of u and w become N'(u) cap N(w) = (N(u) U {v}) cap N(w) = (N(u) cap N(w)) U ({v} cap N(w)).
            # If v is in N(w), then the count increases by 1.
            # So we must check:
            # For all w != u, v:
            #   If v in N(w), then old_common(u, w) + 1 <= 2 => old_common(u, w) <= 1.
            #   If u in N(w), then old_common(v, w) + 1 <= 2 => old_common(v, w) <= 1.
            
            # Let's check these conditions.
            valid = True
            
            # Check 1: u and v themselves
            if bin(adj[u] & adj[v]).count('1') > 2:
                valid = False
            
            # Check 2: Effects on other pairs
            if valid:
                # For each w
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    # If v is neighbor of w, common(u, w) increases by 1
                    if (adj[w] >> v) & 1:
                        if bin(adj[u] & adj[w]).count('1') > 1:
                            valid = False
                            break
                    
                    # If u is neighbor of w, common(v, w) increases by 1
                    if (adj[w] >> u) & 1:
                        if bin(adj[v] & adj[w]).count('1') > 1:
                            valid = False
                            break
            
            if valid:
                # Add edge
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                added_any = True
        
        if added_any:
            current_edges = _edge_count(adj)
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj = adj[:]
        else:
            # Stuck. Try a random kick: remove a random edge to allow new additions?
            # Or just break and restart with a different initial graph.
            # Since we are in a loop that will restart if we exit this function, 
            # we can just break out of the local search loop.
            break
            
    return best_adj, best_edges

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    best_edges_count = 0
    best_adj = None
    
    # Try multiple random starts
    for trial in range(100):
        if time.time() > deadline:
            break
            
        # Generate random graph
        # Start with p=0.5 or slightly higher
        p = 0.5 + random.random() * 0.2
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair
        adj = _repair(adj)
        
        # Local Search
        adj, edge_count = _local_search(adj, deadline)
        
        if edge_count > best_edges_count:
            best_edges_count = edge_count
            best_adj = adj[:]
            
        # If we found a very high edge count, we might stop early?
        # ex(20; K2,3) is likely around 100-150? 
        # Max possible edges in K2,3 free graph is bounded.
        # Reiman bound: O(n^{3/2}). For n=20, 20^{1.5} ~ 89.
        # Actually, ex(n, K_{2,3}) is approx 1/2 * n^{3/2} ?
        # Let's just run until time is up.

    if best_adj is None:
        # Fallback: empty graph
        best_adj = [0] * n
        best_edges_count = 0

    # Convert bitmask adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

# Test locally
if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
    # Validate
    adj = [0] * 20
    for u, v in e:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    max_c = 0
    for i in range(20):
        for j in range(i + 1, 20):
            c = bin(adj[i] & adj[j]).count('1')
            if c > max_c:
                max_c = c
    print(f"Max common neighbors: {max_c}")
    if max_c <= 2:
        print("Valid")
    else:
        print("Invalid")
