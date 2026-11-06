import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 40
    m = 401
    # Initial construction: K20,20 + 1 edge
    # Vertices 0-19 in A, 20-39 in B
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    edges.add((0, 1))
    
    # Ensure exactly 401 edges
    # K20,20 has 400 edges. Adding (0,1) makes 401.
    
    # Build adjacency matrix for fast lookups
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    # Calculate current book sizes for all edges
    # Book size of edge (u,v) is number of common neighbors
    # common_neighbors(u,v) = sum(adj[u] & adj[v])
    
    def get_books():
        books = {}
        for u, v in edges:
            books[(u, v)] = np.sum(adj[u] & adj[v])
        return books

    current_books = get_books()
    max_book = max(current_books.values()) if current_books else 0
    
    # Surrogate objective: Sum of squares of book sizes
    # This penalizes large books more heavily, helping to flatten the distribution
    def surrogate(books):
        if not books:
            return 0
        vals = list(books.values())
        return sum(v * v for v in vals)
    
    current_score = surrogate(current_books)
    best_score = current_score
    best_edges = set(edges)
    best_max_book = max_book
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    alpha = 0.999
    
    # List of edges for random selection
    edge_list = list(edges)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random edge to remove
        e_remove = random.choice(edge_list)
        u, v = e_remove
        
        # Pick a random non-edge to add
        # To keep it fast, pick random pair, check if it's an edge
        # Retry a few times if it is an edge
        added_edge = None
        for _ in range(10):
            a, b = random.sample(range(n), 2)
            if not adj[a, b]:
                added_edge = (a, b)
                break
        
        if added_edge is None:
            continue
            
        a, b = added_edge
        
        # Evaluate change in book sizes
        # Removing (u,v) affects books of edges incident to u or v? No.
        # Book size of edge (x,y) is # common neighbors.
        # If we remove edge (u,v), the common neighbor count for any edge (x,y) changes only if u or v is a common neighbor of x,y AND u-v edge removal affects connectivity?
        # No, book size is defined by the graph structure.
        # common(x,y) = |N(x) ∩ N(y)|.
        # If we remove edge (u,v), N(u) loses v, N(v) loses u.
        # This affects common(x,y) if:
        # 1. x=u or y=u, and v is in N(other) -> common decreases by 1 if v was common.
        # Actually, let's just recompute books for affected edges.
        # Which edges' book sizes change?
        # An edge (x,y) has its book size changed if the set of common neighbors changes.
        # Common neighbors change if a vertex z is added/removed from N(x) or N(y).
        # Here, N(u) loses v, N(v) loses u. N(a) gains b, N(b) gains a.
        
        # Edges affected by removing (u,v):
        # Edges incident to u: (u, z) for z in N(u)\{v}. Their common neighbors might lose v if v is connected to z.
        # Edges incident to v: (v, z) for z in N(v)\{u}. Their common neighbors might lose u if u is connected to z.
        # Also, edges (x,y) where x,y are not u or v?
        # common(x,y) involves N(x) and N(y). N(u) and N(v) changed.
        # If x=u, N(x) changed. If y=u, N(y) changed.
        # If x is not u or v, N(x) didn't change directly, BUT N(x) contains u or v?
        # Wait, N(x) is the set of neighbors. If x is neighbor of u, then u is in N(x).
        # The value common(x,y) = |N(x) ∩ N(y)|.
        # If N(u) changes, it affects common(x,y) only if u is in N(x) or N(y)? No.
        # It affects common(x,y) if the intersection set changes.
        # The intersection set is composed of vertices.
        # If a vertex z is in N(x) ∩ N(y), and we remove edge (u,v), does z stop being in N(x) or N(y)?
        # No, z is a vertex. N(x) is the set of vertices connected to x.
        # Removing edge (u,v) only changes N(u) and N(v).
        # So common(x,y) changes ONLY if x=u or x=v or y=u or y=v.
        # Because only N(u) and N(v) change.
        
        # So, affected edges are those incident to u, v, a, or b.
        
        affected_edges = set()
        for z in range(n):
            if adj[u, z]: affected_edges.add((min(u,z), max(u,z)))
            if adj[v, z]: affected_edges.add((min(v,z), max(v,z)))
            if adj[a, z]: affected_edges.add((min(a,z), max(a,z)))
            if adj[b, z]: affected_edges.add((min(b,z), max(b,z)))
        
        # Store old book sizes
        old_books = {}
        for e in affected_edges:
            if e in current_books:
                old_books[e] = current_books[e]
        
        # Apply change
        adj[u, v] = False
        adj[v, u] = False
        adj[a, b] = True
        adj[b, a] = True
        
        edges.discard(e_remove)
        edges.add(added_edge)
        
        # Update books for affected edges
        new_books = {}
        delta_surrogate = 0
        for e in affected_edges:
            if e in edges:
                x, y = e
                new_size = np.sum(adj[x] & adj[y])
                new_books[e] = new_size
                current_books[e] = new_size
                if e in old_books:
                    delta_surrogate += new_size * new_size - old_books[e] * old_books[e]
                else:
                    # New edge, didn't exist before
                    delta_surrogate += new_size * new_size
            else:
                # Edge removed, was in current_books
                if e in old_books:
                    delta_surrogate -= old_books[e] * old_books[e]
                # Remove from current_books
                if e in current_books:
                    del current_books[e]
                
        # Note: We must be careful. `affected_edges` might contain edges not in `edges` (if they were removed) or edges in `edges` that were not in `affected_edges`?
        # No, `affected_edges` is derived from adjacency.
        # If an edge is in `affected_edges` but not in `edges`, it means it was just removed (if it was e_remove) or it was never there (impossible if derived from adj before change? No, adj changed).
        # Let's just recompute books for all edges in `affected_edges` that are currently in `edges`.
        # And handle the removal of `e_remove` from `current_books` if it's not in `edges`.
        
        # Actually, the logic above mixes up old and new. Let's simplify.
        # 1. Snapshot books for all edges in `affected_edges` that are in `current_books` (old state).
        # 2. Update adj and edges.
        # 3. Recompute books for all edges in `affected_edges` that are in `new edges`.
        # 4. Calculate delta.
        
        # Let's redo the delta calculation cleanly.
        # The code above did it in a mixed way. Let's trust the local re-evaluation but ensure we don't double count or miss.
        
        # A safer way:
        # old_total = sum(b^2 for b in old_books.values())
        # new_total = sum(b^2 for b in new_books.values())
        # But new_books only has edges currently in `edges`.
        # Edges in `affected_edges` that are NOT in `edges` anymore contribute 0.
        # Edges in `edges` that are NOT in `affected_edges` are unchanged.
        
        # So:
        # delta = sum(new_size^2 for e in affected_edges if e in edges) - sum(old_size^2 for e in affected_edges if e in old_edges_subset)
        
        # My previous code:
        # if e in edges: update current_books, add to delta (new^2 - old^2 if existed, else new^2)
        # else: if e in old_books: subtract old^2.
        # This seems correct.
        
        current_score += delta_surrogate
        
        # Calculate new max book
        # To avoid O(M) scan, we can maintain a heap or just scan if M is small (401).
        # Scanning 401 items is fast.
        new_max_book = max(current_books.values()) if current_books else 0
        
        if new_max_book < best_max_book:
            best_max_book = new_max_book
            best_score = current_score
            best_edges = set(edges)
            # Copy adj? No, we can just track edges.
            
        # Acceptance criterion
        if delta_surrogate < 0:
            accept = True
        else:
            accept = random.random() < np.exp(-delta_surrogate / T)
            
        if not accept:
            # Revert
            adj[u, v] = True
            adj[v, u] = True
            adj[a, b] = False
            adj[b, a] = False
            
            edges.discard(added_edge)
            edges.add(e_remove)
            
            # Revert books
            for e in affected_edges:
                if e in old_books:
                    current_books[e] = old_books[e]
                elif e in current_books:
                    # It was added in the forward pass, remove it
                    del current_books[e]
                    
            current_score -= delta_surrogate
            # Max book reversion not strictly needed for SA decision, but good for tracking
            # best_max_book is already stored if improved
            
        T *= alpha
        
    # Final verification and return
    # Ensure we return the best found or the current one?
    # The task asks for the graph. Returning the best found is safer.
    # However, `best_edges` is a set of tuples. We need to return list of lists.
    
    # One more check: is best_edges valid?
    # We should probably just return the current state if it's better, or best.
    # Let's return the best found.
    
    result = [[u, v] for u, v in best_edges]
    
    # Sanity check
    if len(result) != 401:
        # Fallback to initial construction if something went wrong
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
