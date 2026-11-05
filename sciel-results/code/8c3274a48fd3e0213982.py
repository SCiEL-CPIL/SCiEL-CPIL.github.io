import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    d = 7
    n_vertices = 2 ** d
    n_edges = d * (2 ** (d - 1))
    
    # Generate all edges
    edges = []
    edge_to_idx = {}
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                edge_to_idx[e] = len(edges)
                edges.append(e)
                
    # Precompute squares (4-cycles)
    # A square is defined by two distinct dimensions i, j and a base vertex x
    # Vertices: x, x^i, x^j, x^i^j
    # Edges: (x, x^i), (x^i, x^i^j), (x^i^j, x^j), (x^j, x)
    squares = []
    edge_to_squares = {e: [] for e in edges}
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                # Only count each square once.
                # A square is uniquely identified by its 4 vertices.
                # To avoid duplicates, we can require x to be the minimum vertex in the square?
                # Or just generate all and use a set of frozensets of edges?
                # There are 672 squares. Generating all is fast.
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                sq_edges = [e1, e2, e3, e4]
                sq_key = frozenset(sq_edges)
                
                # We need to map each edge to the squares it belongs to.
                # To avoid duplicate square entries in edge_to_squares, we should ensure we only process each unique square once.
                # Using a set of processed squares is easiest.
                if not hasattr(construct, '_processed_squares'):
                    construct._processed_squares = set()
                
                if sq_key in construct._processed_squares:
                    continue
                construct._processed_squares.add(sq_key)
                
                squares.append(sq_edges)
                for e in sq_edges:
                    edge_to_squares[e].append(sq_edges)

    # Reset the cache for subsequent calls if any (though construct is called once usually)
    # Better to use local variable for state if re-entrant, but construct is standalone.
    
    # 1. Greedy Construction
    selected = set()
    for e in edges:
        # Check if adding e creates a C4
        ok = True
        for sq in edge_to_squares[e]:
            # Count selected edges in sq other than e
            count = 0
            for se in sq:
                if se != e and se in selected:
                    count += 1
            if count == 3:
                ok = False
                break
        if ok:
            selected.add(e)
            
    best_count = len(selected)
    
    # 2. Local Search
    # We will try to improve the solution by swapping edges.
    # Move: Remove one edge e_out, add one or more edges e_in.
    # If len(e_in) > 1, we gain.
    
    # Helper: check if edge can be added
    def can_add(e, current_selected):
        for sq in edge_to_squares[e]:
            count = 0
            for se in sq:
                if se != e and se in current_selected:
                    count += 1
            if count == 3:
                return False
        return True

    # Helper: try to add edges to a set, return number added and the new set
    def try_add_edges(unselected_candidates, current_selected):
        added = []
        for e in unselected_candidates:
            if e in current_selected:
                continue
            if can_add(e, current_selected):
                current_selected.add(e)
                added.append(e)
        return len(added), current_selected

    # Local Search Loop
    # To speed up, we can shuffle the order of edges we try to remove.
    # We stop if no improvement is found in a full pass, or if time runs out.
    
    # Precompute list of edges for random access
    edge_list = list(edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(edge_list)
        
        for e_out in edge_list:
            if time.time() > deadline:
                break
            if e_out not in selected:
                continue
                
            # Remove e_out
            selected.remove(e_out)
            
            # Try to add edges that were previously blocked
            # Ideally, we only check edges that were in conflict with e_out.
            # An edge e_in conflicts with e_out if they share a square.
            # Let's collect candidates: edges in squares containing e_out.
            candidates = set()
            for sq in edge_to_squares[e_out]:
                for se in sq:
                    if se != e_out:
                        candidates.add(se)
                        
            # Filter candidates to those not currently selected
            candidates = [c for c in candidates if c not in selected]
            
            # Shuffle candidates to break symmetry
            random.shuffle(candidates)
            
            # Try to add as many as possible
            added_count, new_selected = try_add_edges(candidates, selected)
            
            if added_count > 1:
                # Improvement: removed 1, added > 1
                selected = new_selected
                improved = True
                if len(selected) > best_count:
                    best_count = len(selected)
            elif added_count == 1:
                # Neutral: removed 1, added 1.
                # Accept with some probability to escape local optima?
                # Or just accept if it changes the set (it does).
                # Let's accept it to explore, but be careful not to cycle.
                # Simplest: accept.
                selected = new_selected
                improved = True # Force another pass? No, just mark changed.
                # Actually, if we always accept neutral moves, we might cycle.
                # Let's only accept neutral moves with low probability or if it leads to immediate gain?
                # Given time, let's just accept and see.
            else:
                # No improvement, revert
                selected.add(e_out)
                
        # If no change in a full pass, we are stuck. 
        # To escape, we could try removing 2 edges and adding 3?
        # For now, simple 1-for-1+ swap is the primary driver.
        
    # Construct output list
    out = []
    for e in selected:
        out.append([e[0], e[1]])
        
    return out

# Clear cache on module load? No, construct is called once.
# But if construct is called multiple times in tests, the cache might interfere.
# Let's make the cache local to construct to be safe.
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    d = 7
    n_vertices = 2 ** d
    
    # Generate all edges
    edges = []
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edges.append((x, y))
                
    # Precompute squares
    squares = []
    edge_to_squares = {e: [] for e in edges}
    processed_squares = set()
    
    for i in range(d):
        for j in range(i + 1, d):
            for x in range(n_vertices):
                v1 = x
                v2 = x ^ (1 << i)
                v3 = x ^ (1 << j)
                v4 = x ^ (1 << i) ^ (1 << j)
                
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                
                sq_edges = [e1, e2, e3, e4]
                sq_key = frozenset(sq_edges)
                
                if sq_key in processed_squares:
                    continue
                processed_squares.add(sq_key)
                
                squares.append(sq_edges)
                for e in sq_edges:
                    edge_to_squares[e].append(sq_edges)

    # 1. Greedy Construction
    selected = set()
    for e in edges:
        ok = True
        for sq in edge_to_squares[e]:
            count = 0
            for se in sq:
                if se != e and se in selected:
                    count += 1
            if count == 3:
                ok = False
                break
        if ok:
            selected.add(e)
            
    # 2. Local Search
    def can_add(e, current_selected):
        for sq in edge_to_squares[e]:
            count = 0
            for se in sq:
                if se != e and se in current_selected:
                    count += 1
            if count == 3:
                return False
        return True

    def try_add_edges(unselected_candidates, current_selected):
        added_count = 0
        for e in unselected_candidates:
            if e in current_selected:
                continue
            if can_add(e, current_selected):
                current_selected.add(e)
                added_count += 1
        return added_count

    edge_list = list(edges)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(edge_list)
        
        for e_out in edge_list:
            if time.time() > deadline:
                break
            if e_out not in selected:
                continue
                
            selected.remove(e_out)
            
            candidates = set()
            for sq in edge_to_squares[e_out]:
                for se in sq:
                    if se != e_out:
                        candidates.add(se)
                        
            candidates = [c for c in candidates if c not in selected]
            random.shuffle(candidates)
            
            added_count = try_add_edges(candidates, selected)
            
            if added_count > 1:
                improved = True
            elif added_count == 1:
                # Accept neutral move to explore
                improved = True
            else:
                selected.add(e_out)
                
    out = []
    for e in selected:
        out.append([e[0], e[1]])
    return out
