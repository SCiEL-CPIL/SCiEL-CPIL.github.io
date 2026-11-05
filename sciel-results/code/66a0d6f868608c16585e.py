import time
import random
import numpy as np

def construct():
    d = 7
    n_vertices = 2 ** d
    n_edges = d * 2 ** (d - 1)
    
    # Precompute edges
    edges = []
    edge_to_idx = {}
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                
    # Precompute faces and edge-face incidence
    face_edges = []
    edge_faces = [[] for _ in range(n_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for x in range(n_vertices):
                if (x & mask_i) == 0 and (x & mask_j) == 0:
                    v1 = x
                    v2 = x | mask_i
                    v3 = x | mask_j
                    v4 = x | mask_i | mask_j
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    idxs = []
                    for e in [e1, e2, e3, e4]:
                        idxs.append(edge_to_idx[e])
                    
                    face_idx = len(face_edges)
                    face_edges.append(idxs)
                    for idx in idxs:
                        edge_faces[idx].append(face_idx)
                        
    n_faces = len(face_edges)
    
    # Precompute conflict graph: two edges conflict if they share a face
    # For MCS, we need to know which edges conflict with which
    # Build adjacency sets for the conflict graph
    conflict_sets = [set() for _ in range(n_edges)]
    for f in range(n_faces):
        fe = face_edges[f]
        for a in range(len(fe)):
            for b in range(a + 1, len(fe)):
                conflict_sets[fe[a]].add(fe[b])
                conflict_sets[fe[b]].add(fe[a])
    
    # MCS (Maximum Cardinality Search) for Maximum Independent Set
    # We select edges in an order such that each selected edge has maximum
    # number of already-selected neighbors in the conflict graph (i.e., conflicts with most selected edges)
    # This is a heuristic for MIS: select vertices with high degree in remaining graph
    
    def mcs_mis():
        """Maximum Cardinality Search heuristic for Maximum Independent Set."""
        # Start with all vertices unselected
        unselected = set(range(n_edges))
        selected = []
        # Track which selected edges conflict with which unselected edges
        # For each unselected vertex, track how many selected vertices it conflicts with
        
        # Initialize: all unselected have 0 selected neighbors
        selected_conflict_count = np.zeros(n_edges, dtype=np.int32)
        
        while unselected:
            # Find unselected vertex with maximum number of selected conflicts
            # This is the MCS ordering: pick vertex with max cardinality (selected neighbors)
            best_v = -1
            best_count = -1
            for v in unselected:
                if selected_conflict_count[v] > best_count:
                    best_count = selected_conflict_count[v]
                    best_v = v
            
            # Select this vertex
            selected.append(best_v)
            unselected.remove(best_v)
            
            # Update conflict counts for remaining unselected vertices
            for u in unselected:
                if best_v in conflict_sets[u]:
                    selected_conflict_count[u] += 1
            
            # Prune: if a vertex has conflicts with all selected vertices, it can't be added
            # Actually for MIS, we want vertices that DON'T conflict with selected ones
            # So we should track non-conflicts instead
            
            # Wait, I have this backwards. For Maximum Independent Set:
            # We want to select vertices that don't conflict with each other.
            # MCS for MIS: select vertex with maximum number of neighbors already in the independent set?
            # No, that's wrong. Let me reconsider.
            
            # Actually, for Maximum Independent Set, a common heuristic is:
            # 1. Order vertices by degree (ascending or descending)
            # 2. Greedily select vertex if it doesn't conflict with already selected
            # 
            # MCS is typically for Maximum Clique. For MIS, we can use:
            # - Greedy: pick vertex with minimum degree in remaining graph
            # - Or: pick vertex that conflicts with fewest other unselected vertices
            #
            # Let me use a different approach: greedy with minimum conflicts
            pass
        
        return selected
    
    # Better approach: Greedy MIS with multiple random orderings + local search refinement
    
    def greedy_mis(order):
        """Greedy MIS given an ordering of vertices."""
        selected = []
        selected_set = set()
        for v in order:
            # Check if v conflicts with any selected vertex
            can_add = True
            for u in selected:
                if v in conflict_sets[u]:
                    can_add = False
                    break
            if can_add:
                selected.append(v)
                selected_set.add(v)
        return selected
    
    def local_search_improve(selected_set, time_budget):
        """Improve a MIS using local search (add, remove, swap)."""
        start = time.time()
        deadline = start + time_budget
        current = set(selected_set)
        best = set(current)
        best_size = len(best)
        
        # Precompute for speed
        # For each edge, list of edges it conflicts with
        conflict_list = [list(s) for s in conflict_sets]
        
        # Try to add edges that don't conflict with current
        # Then try swaps
        
        T = 0.5
        cooling = 0.999
        
        while time.time() < deadline:
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random edge
                v = random.randint(0, n_edges - 1)
                if v in current:
                    continue
                # Check if v conflicts with any in current
                can_add = True
                for u in current:
                    if v in conflict_sets[u]:
                        can_add = False
                        break
                if can_add:
                    new_set = current | {v}
                    if len(new_set) > best_size:
                        best = new_set
                        best_size = len(best)
                    current = new_set
                T *= cooling
            elif move_type < 0.7:
                # Try to remove a random edge
                if not current:
                    continue
                v = random.choice(list(current))
                new_set = current - {v}
                # Only accept if it allows adding more later (optimistic)
                # For now, just cool down
                T *= cooling
            else:
                # Try swap: remove v1, add v2
                if not current:
                    continue
                v1 = random.choice(list(current))
                v2 = random.randint(0, n_edges - 1)
                if v2 in current:
                    continue
                # Check if v2 conflicts with (current - {v1})
                can_add = True
                for u in current:
                    if u == v1:
                        continue
                    if v2 in conflict_sets[u]:
                        can_add = False
                        break
                if can_add:
                    new_set = (current - {v1}) | {v2}
                    # Size stays same, but might enable future additions
                    # Accept with some probability to explore
                    if random.random() < np.exp(0.0 / T):  # Always accept same-size swaps
                        current = new_set
                T *= cooling
            
            if T < 0.01:
                T = 0.01
        
        # Final greedy fill
        for v in range(n_edges):
            if v not in current:
                can_add = True
                for u in current:
                    if v in conflict_sets[u]:
                        can_add = False
                        break
                if can_add:
                    current.add(v)
        
        if len(current) > best_size:
            best = current
            best_size = len(best)
        
        return best
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_result = set()
    best_size = 0
    
    # Multiple restarts with different random orderings
    num_restarts = 20
    per_restart_time = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create random ordering
        order = list(range(n_edges))
        random.shuffle(order)
        
        # Greedy MIS
        selected = greedy_mis(order)
        selected_set = set(selected)
        
        if len(selected_set) > best_size:
            best_size = len(selected_set)
            best_result = set(selected_set)
        
        # Local search refinement
        remaining_time = deadline - time.time()
        ls_time = min(per_restart_time * 0.5, remaining_time * 0.4)
        if ls_time > 1.0:
            improved = local_search_improve(selected_set, ls_time)
            if len(improved) > best_size:
                best_size = len(improved)
                best_result = set(improved)
    
    # Final greedy fill on best result
    for v in range(n_edges):
        if v not in best_result:
            can_add = True
            for u in best_result:
                if v in conflict_sets[u]:
                    can_add = False
                    break
            if can_add:
                best_result.add(v)
    
    # Construct output
    out = []
    for idx in sorted(best_result):
        u, v = edges[idx]
        out.append([u, v])
        
    return out
