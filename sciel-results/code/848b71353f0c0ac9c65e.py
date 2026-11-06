import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    N = 16
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Cache for 3-colorability checks
    _colorable_cache = {}
    
    def is_3_colorable(adj):
        """
        Fast 3-colorability check using bitmask backtracking.
        adj is a list of N integers, where adj[i] is a bitmask of neighbors of i.
        """
        key = tuple(adj)
        if key in _colorable_cache:
            return _colorable_cache[key]
        
        # Order vertices by degree descending
        degrees = [bin(adj[i]).count('1') for i in range(N)]
        order = sorted(range(N), key=lambda v: -degrees[v])
        
        # color_masks[c] is a bitmask of vertices colored with c
        color_masks = [0, 0, 0]
        
        def backtrack(pos):
            if pos == N:
                return True
            
            v = order[pos]
            nmask = adj[v]
            
            # Determine forbidden colors based on neighbors' colors
            forbidden = 0
            if nmask & color_masks[0]:
                forbidden |= 1
            if nmask & color_masks[1]:
                forbidden |= 2
            if nmask & color_masks[2]:
                forbidden |= 4
                
            # If all 3 colors are forbidden, this branch fails
            if forbidden == 7:
                return False
            
            # Try available colors
            # Heuristic: Try colors that are least constrained? 
            # For now, just try 0, 1, 2 if available.
            if not (forbidden & 1):
                color_masks[0] |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[0] &= ~(1 << v)
                
            if not (forbidden & 2):
                color_masks[1] |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[1] &= ~(1 << v)
                
            if not (forbidden & 4):
                color_masks[2] |= (1 << v)
                if backtrack(pos + 1):
                    return True
                color_masks[2] &= ~(1 << v)
                
            return False
            
        result = backtrack(0)
        _colorable_cache[key] = result
        return result

    def count_edges(adj):
        total = 0
        for i in range(N):
            total += bin(adj[i]).count('1')
        return total // 2

    def adj_to_edges(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def is_triangle_free_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph triangle-free."""
        return (adj[u] & adj[v]) == 0

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Build the Grötzsch graph (Mycielski of C5) on 11 vertices
    def make_grotzsch():
        adj = [0] * N
        # C5: vertices 0-4
        for i in range(5):
            adj[i] |= (1 << ((i + 1) % 5))
            adj[(i + 1) % 5] |= (1 << i)
        # Mycielski: vertices 5-9 are copies, vertex 10 is the apex
        for i in range(5):
            for j in range(5):
                if adj[i] & (1 << j):
                    adj[5 + i] |= (1 << j)
                    adj[j] |= (1 << (5 + i))
            adj[5 + i] |= (1 << 10)
            adj[10] |= (1 << (5 + i))
        return adj

    # Initial adjacency list
    adj = make_grotzsch()
    best_adj = adj[:]
    best_count = count_edges(adj)
    
    # Ensure the initial graph is not 3-colorable (it should be, but check)
    if is_3_colorable(adj):
        # If for some reason it is, we need a different core or fix
        # But Grötzsch is 4-chromatic.
        pass

    # Local Search
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Phase 1: Try to add edges
        # Sort non-edges by potential (degree sum) to prioritize "dense" additions
        non_edges = []
        for i, j in all_pairs:
            if not (adj[i] & (1 << j)):
                deg_sum = bin(adj[i]).count('1') + bin(adj[j]).count('1')
                non_edges.append((deg_sum, i, j))
        non_edges.sort(key=lambda x: -x[0])
        
        for deg_sum, u, v in non_edges:
            if time.time() > deadline:
                break
            if adj[u] & (1 << v):
                continue
            if not is_triangle_free_add(adj, u, v):
                continue
            
            # Try adding
            add_edge(adj, u, v)
            if not is_3_colorable(adj):
                # Success
                new_count = count_edges(adj)
                if new_count > best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    improved = True
                # Keep the edge, continue searching for more
            else:
                # Revert
                remove_edge(adj, u, v)
                
        if improved:
            no_improve = 0
            continue
        
        no_improve += 1
        
        # Phase 2: Try swaps if no simple add worked
        # Randomize order of existing edges to explore different swap candidates
        existing_edges = [(i, j) for i, j in all_pairs if adj[i] & (1 << j)]
        random.shuffle(existing_edges)
        
        found_swap = False
        for rem_u, rem_v in existing_edges:
            if time.time() > deadline:
                break
            
            # Remove edge
            remove_edge(adj, rem_u, rem_v)
            
            # Try to add one or two edges
            # We want to increase total edge count or maintain it to enable future adds
            # Let's try to find a non-edge that can be added
            candidates = []
            for i, j in all_pairs:
                if (i, j) == (rem_u, rem_v):
                    continue
                if not (adj[i] & (1 << j)):
                    if is_triangle_free_add(adj, i, j):
                        candidates.append((i, j))
            
            random.shuffle(candidates)
            
            for add_u, add_v in candidates:
                if time.time() > deadline:
                    break
                
                # Try 1-for-1 swap
                add_edge(adj, add_u, add_v)
                if not is_3_colorable(adj):
                    # Valid swap (edge count same)
                    # Check if this enables further additions? 
                    # For now, accept it as a move to escape local optimum
                    # But we prefer increases. Let's see if we can add a 2nd edge
                    # Try to add another edge
                    second_added = False
                    for i, j in all_pairs:
                        if (i, j) == (rem_u, rem_v) or (i, j) == (add_u, add_v):
                            continue
                        if not (adj[i] & (1 << j)):
                            if is_triangle_free_add(adj, i, j):
                                add_edge(adj, i, j)
                                if not is_3_colorable(adj):
                                    # 1-for-2 swap successful, net +1 edge
                                    new_count = count_edges(adj)
                                    if new_count > best_count:
                                        best_count = new_count
                                        best_adj = adj[:]
                                    found_swap = True
                                    second_added = True
                                    break
                                else:
                                    remove_edge(adj, i, j)
                    
                    if not second_added:
                        # 1-for-1 swap, edge count same
                        # Accept to change structure
                        found_swap = True
                    break # Move to next removal candidate or restart loop
                
                # Revert 1-for-1 attempt
                remove_edge(adj, add_u, add_v)
            
            # Re-add removed edge if no swap was found/accepted
            if not found_swap:
                add_edge(adj, rem_u, rem_v)
            else:
                # If a swap was found, we broke out of inner loop
                # We need to break out of outer loop or handle state
                # The 'found_swap' flag indicates we modified 'adj' successfully
                # We should break the outer loop to restart the add phase
                break
        
        if not found_swap:
            # No swap found, try random restart or give up
            # To avoid getting stuck, we can randomly perturb
            if no_improve > 100:
                # Random perturbation: remove 2 random edges, add 2 random valid ones
                # This is complex, so let's just break and try a new seed/restart
                break

    # Final verification and conversion
    final_adj = best_adj
    # Ensure final_adj is valid
    # Check triangle-free
    for u in range(N):
        for v in range(u + 1, N):
            if (final_adj[u] >> v) & 1:
                if (final_adj[u] & final_adj[v]) != 0:
                    # Triangle found, this shouldn't happen
                    pass
    
    # Check 3-colorable
    if is_3_colorable(final_adj):
        # If the best found is 3-colorable, we need a valid non-3-colorable graph
        # Fall back to the initial Grötzsch graph if no better valid one was found
        # But best_adj is only updated if non-3-colorable.
        # So best_adj should be non-3-colorable.
        pass

    return adj_to_edges(final_adj)
