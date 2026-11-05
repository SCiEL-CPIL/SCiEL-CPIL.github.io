import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))
    
    # 1. Generate all edges and map them to indices
    edges = []
    edge_index = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = idx
                edges.append((u, v))
                idx += 1
    
    # 2. Generate all squares (2D faces)
    # A square is defined by a vertex u and two distinct dimensions i, j
    # Vertices: u, u^i, u^j, u^i^j
    # Edges: (u, u^i), (u^i, u^i^j), (u^i^j, u^j), (u^j, u)
    squares = []
    # To avoid duplicates, we can just iterate u and i<j
    # But each square is counted 4 times (once per vertex) if we are not careful?
    # Actually, if we fix u to be the "minimum" vertex in the square, we get unique squares.
    # Or simpler: just generate all and use a set of frozensets of edge indices.
    
    square_set = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, u), max(v2, u))
                
                # Get indices
                idxs = []
                for e in (e1, e2, e3, e4):
                    if e not in edge_index:
                        # Should not happen
                        continue
                    idxs.append(edge_index[e])
                
                if len(idxs) == 4:
                    square_set.add(frozenset(idxs))
    
    squares = [list(s) for s in square_set]
    
    # 3. Build incidence lists
    # edge_squares[i] = list of square indices containing edge i
    edge_squares = [[] for _ in range(n_edges)]
    for sq_idx, sq in enumerate(squares):
        for e_idx in sq:
            edge_squares[e_idx].append(sq_idx)

    # Helper to check time
    def time_exceeded():
        return (time.time() - start_time) > time_limit

    def run_local_search(initial_set):
        # initial_set is a set of edge indices
        in_set = [False] * n_edges
        for e in initial_set:
            in_set[e] = True
        
        # Initialize blockers
        # blockers[i] = number of squares Q containing i such that other 3 edges of Q are in in_set
        blockers = [0] * n_edges
        
        for sq_idx, sq in enumerate(squares):
            # Count how many edges in this square are in in_set
            count_in = 0
            for e in sq:
                if in_set[e]:
                    count_in += 1
            
            if count_in == 3:
                # The missing edge is blocked by this square
                for e in sq:
                    if not in_set[e]:
                        blockers[e] += 1
            elif count_in == 4:
                # This square is full. This should not happen if initial_set is valid.
                # If it does, we have an invalid start.
                pass

        current_size = len(initial_set)
        
        # Phase 1: Add all possible edges
        def add_all_possible():
            nonlocal in_set, blockers, current_size
            added_edges = []
            changed = True
            while changed:
                changed = False
                for i in range(n_edges):
                    if not in_set[i] and blockers[i] == 0:
                        # Add edge i
                        in_set[i] = True
                        current_size += 1
                        added_edges.append(i)
                        changed = True
                        # Update blockers for neighbors
                        for sq_idx in edge_squares[i]:
                            sq = squares[sq_idx]
                            # For each other edge j in sq
                            for j in sq:
                                if j == i:
                                    continue
                                if not in_set[j]:
                                    # Check if other 2 edges in sq are in in_set
                                    # sq has 4 edges: i, j, k, l
                                    # We need to check k, l
                                    # Simple way: count in_set in sq excluding i and j
                                    # Since i is now in_set, we just check if the other 2 are in_set
                                    count_other = 0
                                    for k in sq:
                                        if k != i and k != j:
                                            if in_set[k]:
                                                count_other += 1
                                    if count_other == 2:
                                        blockers[j] += 1
            return added_edges

        # Run initial add-all
        add_all_possible()
        
        best_size = current_size
        best_state = in_set[:]
        
        # Phase 2: Hill Climbing
        # We try to remove one edge and add back as many as possible
        # We use a list of edges to try, shuffled
        edges_to_try = list(range(n_edges))
        
        while not time_exceeded():
            random.shuffle(edges_to_try)
            improvement_found = False
            for e in edges_to_try:
                if time_exceeded():
                    break
                if not in_set[e]:
                    continue
                
                # Save state
                saved_in_set = in_set[:]
                saved_blockers = blockers[:]
                saved_size = current_size
                
                # Remove e
                in_set[e] = False
                current_size -= 1
                
                # Update blockers for neighbors of e
                for sq_idx in edge_squares[e]:
                    sq = squares[sq_idx]
                    for j in sq:
                        if j == e:
                            continue
                        if not in_set[j]:
                            # Check if other 2 edges in sq are in in_set
                            count_other = 0
                            for k in sq:
                                if k != e and k != j:
                                    if in_set[k]:
                                        count_other += 1
                            if count_other == 2:
                                blockers[j] -= 1
                                if blockers[j] < 0:
                                    # Should not happen if logic is correct
                                    pass
                                # If it becomes 0, it's unblocked, but we handle that in add_all_possible
                
                # Try to add back
                added = add_all_possible()
                
                if current_size > best_size:
                    best_size = current_size
                    best_state = in_set[:]
                    improvement_found = True
                else:
                    # Revert
                    in_set = saved_in_set
                    blockers = saved_blockers
                    current_size = saved_size
                    # Note: added edges were added to in_set and blockers.
                    # By restoring saved_in_set and saved_blockers, we effectively revert.
                    # But we must ensure we don't leave dangling state.
                    # The restore is a full copy, so it's safe.

            if not improvement_found:
                # Local optimum reached. Break or restart?
                # Since we are in a single run, we can break and let the outer loop restart.
                break
        
        # Return the best state found in this run
        best_edges = []
        for i in range(n_edges):
            if best_state[i]:
                best_edges.append(i)
        return best_edges

    # Outer Loop: Multiple restarts
    best_edges_global = []
    best_size_global = 0
    
    # Initial greedy to get a decent start
    # We can also just start with empty set and let local search build it?
    # No, local search "remove one, add many" works best if we start with a maximal set.
    
    while not time_exceeded():
        # Generate a random initial valid set
        # Use randomized greedy
        random_edges = list(range(n_edges))
        random.shuffle(random_edges)
        
        init_set = []
        in_set = [False] * n_edges
        # Simple validity check for init
        # We can just use the blockers logic or simple square check
        # For init, simple check is fine: O(448 * 6)
        for e in random_edges:
            ok = True
            for sq_idx in edge_squares[e]:
                sq = squares[sq_idx]
                # Check if other 3 are in init_set
                count_in = 0
                for k in sq:
                    if k == e:
                        continue
                    if in_set[k]:
                        count_in += 1
                if count_in == 3:
                    ok = False
                    break
            if ok:
                in_set[e] = True
                init_set.append(e)
        
        # Run local search
        result_edges = run_local_search(init_set)
        
        if len(result_edges) > best_size_global:
            best_size_global = len(result_edges)
            best_edges_global = result_edges
            
        # If we hit the theoretical max (unlikely) or time is low, break
        if best_size_global >= 308: # Upper bound mentioned
            break

    # Convert edge indices back to [u, v] format
    final_edges = []
    for e_idx in best_edges_global:
        u, v = edges[e_idx]
        final_edges.append([u, v])
        
    return final_edges
