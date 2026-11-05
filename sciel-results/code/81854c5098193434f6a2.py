import time
import random
import sys

def construct():
    deadline = time.time() + 115.0  # Safety margin before 150s kill (plus overhead)
    
    d = 7
    n_vertices = 1 << d
    num_edges = d * (1 << (d - 1))  # 448
    
    # 1. Generate Edges and Indexing
    # Edge index: (u, v) where u < v
    edge_u = [0] * num_edges
    edge_v = [0] * num_edges
    edge_idx_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_u[idx] = u
                edge_v[idx] = v
                edge_idx_map[(u, v)] = idx
                idx += 1
                
    # 2. Generate Squares
    # Square defined by base vertex and two dimensions i, j
    # Vertices: u, u^i, u^j, u^i^j
    # Edges: (u, u^i), (u, u^j), (u^i, u^i^j), (u^j, u^i^j)
    squares = []
    edges_in_square = []  # List of lists of 4 edge indices
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_idx_map.get((min(u, v1), max(u, v1)))
                e2 = edge_idx_map.get((min(u, v2), max(u, v2)))
                e3 = edge_idx_map.get((min(v1, v3), max(v1, v3)))
                e4 = edge_idx_map.get((min(v2, v3), max(v2, v3)))
                
                if None not in (e1, e2, e3, e4):
                    sq_edges = [e1, e2, e3, e4]
                    edges_in_square.append(sq_edges)
                    squares.append(sq_edges)
                    
    num_squares = len(squares)
    
    # Map edge -> list of square indices it belongs to
    squares_by_edge = [[] for _ in range(num_edges)]
    for s_idx, sq in enumerate(squares):
        for e in sq:
            squares_by_edge[e].append(s_idx)
            
    # 3. State Management
    # selected[e] is True if edge e is in the set
    # sq_counts[s] is the number of selected edges in square s
    # count is the total number of selected edges
    
    def can_add(e, selected, sq_counts):
        """Check if edge e can be added without creating a full square."""
        for s_idx in squares_by_edge[e]:
            if sq_counts[s_idx] >= 3:
                return False
        return True

    def add_edge(e, selected, sq_counts):
        selected[e] = True
        for s_idx in squares_by_edge[e]:
            sq_counts[s_idx] += 1

    def remove_edge(e, selected, sq_counts):
        selected[e] = False
        for s_idx in squares_by_edge[e]:
            sq_counts[s_idx] -= 1

    def greedy_init(rng):
        selected = [False] * num_edges
        sq_counts = [0] * num_squares
        count = 0
        edges_order = list(range(num_edges))
        rng.shuffle(edges_order)
        
        for e in edges_order:
            if can_add(e, selected, sq_counts):
                add_edge(e, selected, sq_counts)
                count += 1
        return selected, sq_counts, count

    def local_search(selected, sq_counts, count, rng, time_limit):
        """
        Improve the solution using local search.
        1. Add any possible edge.
        2. If stuck, try 1-for-1 swaps (neutral) to move around the plateau.
        3. If stuck, try 1-for-2 or 2-for-1 swaps (improving).
        4. If completely stuck, return.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Add edges
            added = False
            # Randomize order to break symmetry
            unselected_edges = [e for e in range(num_edges) if not selected[e]]
            rng.shuffle(unselected_edges)
            for e in unselected_edges:
                if time.time() >= time_limit:
                    break
                if can_add(e, selected, sq_counts):
                    add_edge(e, selected, sq_counts)
                    count += 1
                    improved = True
                    added = True
                    # Don't break, keep adding if possible
                    # But re-shuffling every time is expensive. 
                    # Since adding an edge only blocks others, we can just continue scanning.
                    # However, the list `unselected_edges` is now stale (e is selected).
                    # It's safer to break and restart the phase if we added many, 
                    # or just continue and skip selected ones.
                    # Let's continue and skip.
            
            if not added:
                # Phase 2: Local Optimum. Try swaps.
                
                # 2a. 1-for-1 Swaps (Neutral moves to explore landscape)
                # Pick a random selected edge to remove, then try to add a random unselected one.
                selected_edges = [e for e in range(num_edges) if selected[e]]
                unselected_edges = [e for e in range(num_edges) if not selected[e]]
                
                if not unselected_edges:
                    break # All edges selected? Unlikely, but safe.
                
                swap_success = False
                # Limit attempts to avoid getting stuck in infinite neutral loops
                for _ in range(100): 
                    if time.time() >= time_limit:
                        return selected, sq_counts, count
                    
                    if not selected_edges:
                        break
                        
                    e_out = rng.choice(selected_edges)
                    e_in = rng.choice(unselected_edges)
                    
                    if e_out == e_in:
                        continue
                        
                    # Check if e_in can be added if e_out is removed
                    # Temporarily remove e_out
                    remove_edge(e_out, selected, sq_counts)
                    
                    if can_add(e_in, selected, sq_counts):
                        # Accept the swap (size remains same, but structure changes)
                        add_edge(e_in, selected, sq_counts)
                        # Update lists
                        selected_edges.remove(e_out)
                        selected_edges.append(e_in)
                        unselected_edges.remove(e_in)
                        unselected_edges.append(e_out)
                        improved = True
                        swap_success = True
                        break
                    else:
                        # Revert
                        add_edge(e_out, selected, sq_counts)
                        
                if swap_success:
                    continue # Restart loop, maybe now we can add edges
                
                # 2b. 2-for-1 Swaps (Improving: remove 1, add 2)
                # This is harder to find directly. 
                # Strategy: Remove a random edge e_out. Then try to add two edges.
                # If we can add 2, net gain +1.
                if selected_edges:
                    e_out = rng.choice(selected_edges)
                    remove_edge(e_out, selected, sq_counts)
                    
                    # Try to add as many as possible
                    added_count = 0
                    added_edges = []
                    unselected_edges = [e for e in range(num_edges) if not selected[e]]
                    rng.shuffle(unselected_edges)
                    
                    for e_in in unselected_edges:
                        if time.time() >= time_limit:
                            # Revert and return
                            add_edge(e_out, selected, sq_counts)
                            return selected, sq_counts, count - added_count
                        if can_add(e_in, selected, sq_counts):
                            add_edge(e_in, selected, sq_counts)
                            added_count += 1
                            added_edges.append(e_in)
                            if added_count >= 2:
                                break
                                
                    if added_count >= 2:
                        count += added_count - 1 # Removed 1, added 2
                        improved = True
                        # Continue local search
                    else:
                        # Revert
                        for e in added_edges:
                            remove_edge(e, selected, sq_counts)
                        add_edge(e_out, selected, sq_counts)
                        # If we couldn't improve, we might be stuck.
                        # Break to allow perturbation in outer loop?
                        # Or try 1-for-2?
                        # Let's just break to outer loop for perturbation.
                        break
                        
        return selected, sq_counts, count

    # Main ILS Loop
    best_selected = None
    best_sq_counts = None
    best_count = 0
    
    rng = random.Random(42)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() >= deadline - 10:
            break
            
        # 1. Greedy Init
        selected, sq_counts, count = greedy_init(rng)
        
        # 2. Local Search
        selected, sq_counts, count = local_search(selected, sq_counts, count, rng, deadline - 10)
        
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            best_sq_counts = sq_counts[:]
            
        # 3. Perturbation (for next restart, if we were doing ILS on the same solution)
        # Here we do fresh restarts, which is a form of perturbation.
        # To make it more effective, we could perturb the best solution found so far.
        # Let's try a hybrid: 
        # - First 10 restarts: Fresh random greedy.
        # - Next 20 restarts: Perturb the current best.
        # - Rest: Fresh random.
        
        if restart >= 10 and restart < 30 and best_selected is not None:
            # Perturb best solution
            perturbed_selected = best_selected[:]
            perturbed_sq_counts = best_sq_counts[:]
            perturbed_count = best_count
            
            # Remove 5-10 random edges
            num_removals = rng.randint(5, 10)
            selected_edges_list = [e for e in range(num_edges) if perturbed_selected[e]]
            rng.shuffle(selected_edges_list)
            for e in selected_edges_list[:num_removals]:
                remove_edge(e, perturbed_selected, perturbed_sq_counts)
                perturbed_count -= 1
                
            # Re-localize
            perturbed_selected, perturbed_sq_counts, perturbed_count = local_search(
                perturbed_selected, perturbed_sq_counts, perturbed_count, rng, deadline - 10
            )
            
            if perturbed_count > best_count:
                best_count = perturbed_count
                best_selected = perturbed_selected
                best_sq_counts = perturbed_sq_counts

    # Final Polish on Best Solution
    if best_selected is not None and time.time() < deadline - 2:
        final_selected, final_sq_counts, final_count = local_search(
            best_selected, best_sq_counts, best_count, rng, deadline - 1
        )
        if final_count > best_count:
            best_count = final_count
            best_selected = final_selected
            best_sq_counts = final_sq_counts

    # Construct Result
    result = []
    for e in range(num_edges):
        if best_selected[e]:
            result.append([edge_u[e], edge_v[e]])
            
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of edges: {len(result)}")
