import random
import time

def construct():
    d = 7
    n = 2 ** d
    total_edges = d * (n // 2)
    
    # Precompute all edges as (u, v, dim)
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))
                
    # Map edge (u,v) to its index in all_edges for faster lookup? 
    # Or just use set of (u,v) pairs.
    # For speed, let's create a dictionary mapping (min, max) to index
    edge_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        key = (u, v)
        edge_to_idx[key] = idx

    def is_safe_add(u, v, i, selected_set):
        """Check if adding edge (u,v) with dimension i to selected_set creates a C4."""
        for j in range(d):
            if j == i:
                continue
            mask_j = 1 << j
            u_j = u ^ mask_j
            v_j = v ^ mask_j
            
            e1 = (min(u, u_j), max(u, u_j))
            e2 = (min(v, v_j), max(v, v_j))
            e3 = (min(u_j, v_j), max(u_j, v_j))
            
            if e1 in selected_set and e2 in selected_set and e3 in selected_set:
                return False
        return True

    def is_safe_remove(u, v, i, selected_set):
        """Removing an edge never creates a C4, so this is always safe. 
           But we need to ensure the rest of the graph is valid? 
           No, if the current state is valid, removing an edge keeps it valid."""
        return True

    best_count = 0
    best_edges = []
    
    start_time = time.time()
    max_time = 115.0
    
    # Initialize with a greedy solution to give a good starting point
    # This is a hybrid: Greedy start + Local Search improvement
    initial_edges = []
    initial_set = set()
    random.shuffle(all_edges)
    for u, v, i in all_edges:
        if is_safe_add(u, v, i, initial_set):
            initial_set.add((u, v))
            initial_edges.append([u, v])
            
    best_count = len(initial_edges)
    best_edges = initial_edges
    
    # Local Search / Simulated Annealing
    # We will run multiple restarts or just one long run?
    # Given time constraints, one long run with restarts if stuck is good.
    
    current_edges = list(initial_edges)
    current_set = set(initial_set)
    current_count = best_count
    
    # Pre-compute neighbors for faster swap checks? 
    # For now, on-the-fly is O(d) = 7 ops. 448 edges. 
    # 100,000 iterations * 7 * 3 lookups ~ 2M ops. Very fast in Python? 
    # Set lookups are fast. 2M is fine within 100s.
    
    no_improve_count = 0
    threshold = 2000
    
    while time.time() - start_time < max_time:
        # Pick a random move type
        # 50% Add, 25% Remove, 25% Swap
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to Add a random edge
            if len(current_set) < total_edges:
                u, v, i = random.choice(all_edges)
                key = (u, v)
                if key not in current_set:
                    if is_safe_add(u, v, i, current_set):
                        current_set.add(key)
                        current_edges.append([u, v])
                        current_count += 1
                        no_improve_count = 0
                        if current_count > best_count:
                            best_count = current_count
                            best_edges = list(current_edges)
                            if best_count >= 308:
                                break
                    else:
                        no_improve_count += 1
        elif move_type < 0.75:
            # Try to Remove a random edge
            if current_count > 0:
                idx = random.randint(0, current_count - 1)
                edge_to_remove = current_edges[idx]
                key = (edge_to_remove[0], edge_to_remove[1])
                
                # Find dimension to update set correctly? 
                # We don't strictly need dim for removal from set, 
                # but we need to update current_edges list.
                
                # Remove from set
                current_set.remove(key)
                
                # Remove from list (swap with last for O(1))
                last_idx = len(current_edges) - 1
                current_edges[idx] = current_edges[last_idx]
                current_edges.pop()
                
                current_count -= 1
                no_improve_count += 1
        else:
            # Try to Swap: Remove one, Add one
            if current_count > 0 and len(current_set) < total_edges:
                # Pick edge to remove
                idx_out = random.randint(0, current_count - 1)
                edge_out = current_edges[idx_out]
                key_out = (edge_out[0], edge_out[1])
                
                # Pick edge to add
                # We need to find an edge not in current_set.
                # Randomly picking from all_edges and checking is O(1) avg if density is low.
                # Density is ~300/448 ~ 67%. So we might need a few tries.
                e_in_found = None
                for _ in range(10):
                    u_in, v_in, i_in = random.choice(all_edges)
                    key_in = (u_in, v_in)
                    if key_in not in current_set:
                        e_in_found = (u_in, v_in, i_in, key_in)
                        break
                
                if e_in_found:
                    u_in, v_in, i_in, key_in = e_in_found
                    
                    # Tentatively remove out
                    # We need to check if adding in is safe given the set WITHOUT out.
                    # is_safe_add checks against the current set.
                    # If key_out and key_in are "independent" (don't form C4 with each other or share vertices in a way that matters for the C4 check logic?), 
                    # we can just check is_safe_add on the current set.
                    # BUT, if removing key_out "unblocks" key_in, we need to check against set - {key_out}.
                    
                    # Optimization: If key_out is not one of the edges forming the C4 with key_in, 
                    # then is_safe_add(key_in, current_set) is sufficient.
                    # If key_out IS part of the C4 blocking key_in, we need to check against current_set - {key_out}.
                    
                    # Simple robust approach:
                    # Temporarily remove key_out, check is_safe_add for key_in, then restore if failed.
                    
                    current_set.remove(key_out)
                    
                    safe = is_safe_add(u_in, v_in, i_in, current_set)
                    
                    if safe:
                        # Accept swap
                        current_set.add(key_in)
                        
                        # Update list: replace edge_out with edge_in
                        # We already have idx_out
                        current_edges[idx_out] = [u_in, v_in]
                        # Count stays same
                        no_improve_count = 0
                        # best_count doesn't change, but we might have found a better configuration? 
                        # No, count is same. But we might escape local minima.
                    else:
                        # Reject swap, restore
                        current_set.add(key_out)
                        no_improve_count += 1
            
        # Check for restart
        if no_improve_count > threshold:
            # Restart from best found so far
            current_edges = list(best_edges)
            current_set = set(tuple(e) for e in current_edges)
            current_count = best_count
            no_improve_count = 0
            # Optionally shuffle best edges? No, just restart from that state.
            # To avoid cycling, we can randomize the order of operations or just accept the restart.
            # A more aggressive restart: Go back to a random greedy solution?
            # Let's stick to restarting from best.

    return best_edges
