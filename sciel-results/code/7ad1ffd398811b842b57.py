import time
import random

def construct():
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # Edge representation: index 0..447
    # We need a mapping from (u, v) to index and back
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    # Precompute face neighbors
    # For each edge e, we need to know the other edges in the faces it belongs to.
    # An edge (u, v) with diff bit i belongs to faces defined by (i, j) for j != i.
    # In face (i, j), the 4 edges are:
    # e1: (u, v) where v = u ^ (1<<i)
    # e2: (u ^ (1<<j), v ^ (1<<j))
    # e3: (u, u ^ (1<<j))
    # e4: (v, v ^ (1<<j))
    
    # We will store for each edge index, a list of tuples (neighbor_idx, other1_idx, other2_idx)
    # where neighbor_idx is the edge opposite to e in the face? No, just any neighbor.
    # Actually, for updating conflict_count, we need to know, for a neighbor n,
    # which other two edges a, b complete the face with e and n.
    
    # Let's build: face_neighbors[e] = list of (n, a, b)
    # This means e, n, a, b form a face.
    
    face_neighbors = [[] for _ in range(n_edges)]
    
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u >= v: continue
            e_idx = edge_to_idx[(u, v)]
            
            for j in range(d):
                if i == j: continue
                # Face dimensions i, j
                # Vertices: u, v, u^j, v^j
                # Edges:
                # e: (u, v)
                # e_opp: (u^j, v^j)
                # e_side1: (u, u^j)
                # e_side2: (v, v^j)
                
                u_j = u ^ (1 << j)
                v_j = v ^ (1 << j)
                
                # Ensure canonical ordering for edge lookup
                def get_idx(a, b):
                    if a > b: a, b = b, a
                    return edge_to_idx.get((a, b), -1)
                
                e_opp_idx = get_idx(u_j, v_j)
                e_side1_idx = get_idx(u, u_j)
                e_side2_idx = get_idx(v, v_j)
                
                # The neighbors of e in this face are e_opp, e_side1, e_side2
                # For e_opp: the other two are e_side1, e_side2
                # For e_side1: the other two are e_opp, e_side2
                # For e_side2: the other two are e_opp, e_side1
                
                face_neighbors[e_idx].append((e_opp_idx, e_side1_idx, e_side2_idx))
                face_neighbors[e_idx].append((e_side1_idx, e_opp_idx, e_side2_idx))
                face_neighbors[e_idx].append((e_side2_idx, e_opp_idx, e_side1_idx))
                
                # Also need to add these relationships to the neighbors' lists?
                # No, we only need to iterate neighbors of the edge being added/removed.
                # When we add e, we iterate face_neighbors[e].
                # For each (n, a, b), we check if a and b are selected.
                # This is sufficient.

    selected = [False] * n_edges
    conflict_count = [0] * n_edges
    
    def add_edge(e):
        if selected[e]: return
        if conflict_count[e] > 0: return
        selected[e] = True
        for n, a, b in face_neighbors[e]:
            if selected[a] and selected[b]:
                conflict_count[n] += 1
        # Note: We do not update conflict_count[e] here because e is now selected.
        # And we don't need to decrement anything else.
        
    def remove_edge(e):
        if not selected[e]: return
        selected[e] = False
        for n, a, b in face_neighbors[e]:
            if selected[a] and selected[b]:
                conflict_count[n] -= 1
        # When removing e, we must also consider that e was part of the "other 3" for its neighbors.
        # Wait, the logic above:
        # conflict_count[n] counts faces where the OTHER 3 edges (excluding n) are selected.
        # If n is a neighbor of e in face F, the other 3 for n are e, a, b.
        # If we remove e, then e is not selected. So the condition "e, a, b selected" becomes false.
        # So conflict_count[n] should decrease by 1 IF a and b are still selected.
        # This is exactly what the loop does.
        
    def greedy_fill():
        """Adds all possible edges in random order."""
        edge_indices = list(range(n_edges))
        random.shuffle(edge_indices)
        for e in edge_indices:
            if conflict_count[e] == 0:
                add_edge(e)

    def get_count():
        return sum(selected)

    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_count = 0
    best_selected = [False] * n_edges
    
    # Initial Greedy
    greedy_fill()
    best_count = get_count()
    best_selected = selected[:]
    
    # Local Search / Iterated Greedy
    while time.time() < deadline:
        # Try to add any valid edge (Greedy step)
        # We can just call greedy_fill, but it shuffles every time.
        # To be more "local", we can just scan for one.
        # But greedy_fill is O(N log N) for shuffle + O(N * K) for adds.
        # N=448, K=18. Very fast.
        
        # Check if we can improve
        # Instead of scanning for one, let's just try to add all possible.
        # If greedy_fill adds nothing, we are stuck.
        
        # Optimization: Check if there are any valid edges first?
        # Or just run greedy_fill. If count doesn't change, we were stuck.
        
        prev_count = get_count()
        greedy_fill()
        curr_count = get_count()
        
        if curr_count > prev_count:
            # We added some edges
            if curr_count > best_count:
                best_count = curr_count
                best_selected = selected[:]
        else:
            # Stuck. Perform a kick.
            # Remove k random selected edges
            # Choose k based on current count? Or fixed?
            # Larger k escapes deeper local optima but drops count more.
            # Let's try varying k.
            k = random.randint(2, 6)
            selected_indices = [i for i in range(n_edges) if selected[i]]
            if not selected_indices:
                break
            to_remove = random.sample(selected_indices, min(k, len(selected_indices)))
            for e in to_remove:
                remove_edge(e)
                
            # After kick, run greedy_fill to recover
            greedy_fill()
            
            # Update best if improved
            curr_count = get_count()
            if curr_count > best_count:
                best_count = curr_count
                best_selected = selected[:]
            
            # Optional: If we dropped significantly and didn't recover, we might want to revert?
            # But ILS usually accepts the new state to explore new basins.
            # We keep the current state (post-greedy-fill) as the new starting point.

    # Return the best found
    out = []
    for i in range(n_edges):
        if best_selected[i]:
            u, v = idx_to_edge[i]
            out.append([u, v])
    return out
