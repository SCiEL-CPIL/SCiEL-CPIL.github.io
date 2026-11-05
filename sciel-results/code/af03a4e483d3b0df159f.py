import time
import random
import sys

def construct():
    """
    Constructs a C4-free edge set of Q7 maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    d = 7
    n_vertices = 2 ** d
    num_edges = d * (2 ** (d - 1)) # 448
    
    # 1. Generate Edges
    # Edge index mapping
    edge_u = [0] * num_edges
    edge_v = [0] * num_edges
    edge_idx = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_u[idx] = u
                edge_v[idx] = v
                edge_idx[(u, v)] = idx
                idx += 1
                
    # 2. Generate Squares (2-faces)
    # A square is defined by a vertex and two distinct dimensions i, j.
    # Vertices: u, u^i, u^j, u^i^j
    # Edges: (u, u^i), (u, u^j), (u^i, u^i^j), (u^j, u^i^j)
    squares = []
    edges_in_square = [] # List of lists of edge indices
    
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                
                e1 = edge_idx.get((min(u, v1), max(u, v1)))
                e2 = edge_idx.get((min(u, v2), max(u, v2)))
                e3 = edge_idx.get((min(v1, v3), max(v1, v3)))
                e4 = edge_idx.get((min(v2, v3), max(v2, v3)))
                
                if e1 is not None and e2 is not None and e3 is not None and e4 is not None:
                    sq_edges = [e1, e2, e3, e4]
                    squares.append(sq_edges)
                    edges_in_square.append(sq_edges)
                    
    # Map edge -> list of squares it belongs to
    squares_by_edge = [[] for _ in range(num_edges)]
    for s_idx, sq in enumerate(squares):
        for e in sq:
            squares_by_edge[e].append(s_idx)
            
    # 3. Greedy Initial Solution
    # Randomized greedy to break symmetry
    random.seed(42) # Fixed seed for reproducibility in this block, but we can vary it in restarts
    # Actually, let's use a deterministic greedy first, then perturb.
    
    def greedy_init():
        selected = [False] * num_edges
        count = 0
        # Try all edges in random order? Or fixed order?
        # Fixed order is fast. Random order might find different optima.
        # Let's use a heuristic: pick edge with min degree in remaining conflict graph?
        # Too slow. Just random shuffle.
        edges_order = list(range(num_edges))
        random.shuffle(edges_order)
        
        for e in edges_order:
            # Check if adding e creates a full square
            ok = True
            for s_idx in squares_by_edge[e]:
                sq = edges_in_square[s_idx]
                # Check if other 3 are selected
                # sq has 4 edges. One is e.
                # We need to check the other 3.
                other_selected = 0
                for oe in sq:
                    if oe != e and selected[oe]:
                        other_selected += 1
                if other_selected == 3:
                    ok = False
                    break
            if ok:
                selected[e] = True
                count += 1
        return selected, count

    best_selected, best_count = greedy_init()
    
    # Local Search Function
    # Returns the improved selected list and count
    def local_search(selected, count):
        # While we can add an edge, add it
        # And try swaps to escape local optima
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any valid edge
            # We can scan all unselected edges
            added = False
            for e in range(num_edges):
                if not selected[e]:
                    ok = True
                    for s_idx in squares_by_edge[e]:
                        sq = edges_in_square[s_idx]
                        other_selected = 0
                        for oe in sq:
                            if oe != e and selected[oe]:
                                other_selected += 1
                        if other_selected == 3:
                            ok = False
                            break
                    if ok:
                        selected[e] = True
                        count += 1
                        improved = True
                        added = True
                        break # Restart scan from beginning? Or continue?
                        # Continuing is fine, but a new edge might block others.
                        # Since we only add, it only makes constraints tighter.
                        # So we can just continue.
            
            if not added:
                # Local optimum reached. Try swaps.
                # Pick a random selected edge to remove, and a random unselected to add
                # If the new state is valid, keep it (size same).
                # This changes the structure.
                
                # Get lists
                selected_edges = [i for i in range(num_edges) if selected[i]]
                unselected_edges = [i for i in range(num_edges) if not selected[i]]
                
                if not unselected_edges:
                    break
                
                # Try a few random swaps
                swap_success = False
                for _ in range(50): # Limit swap attempts per local opt
                    if time.time() > deadline:
                        return selected, count
                    
                    e_out = random.choice(selected_edges)
                    e_in = random.choice(unselected_edges)
                    
                    if e_out == e_in:
                        continue
                        
                    # Temporarily remove e_out
                    selected[e_out] = False
                    
                    # Check if e_in is valid
                    ok = True
                    for s_idx in squares_by_edge[e_in]:
                        sq = edges_in_square[s_idx]
                        other_selected = 0
                        for oe in sq:
                            if oe != e_in and selected[oe]:
                                other_selected += 1
                        if other_selected == 3:
                            ok = False
                            break
                    
                    if ok:
                        selected[e_in] = True
                        # e_out remains False
                        # Count is same
                        improved = True
                        swap_success = True
                        # Update lists? 
                        # selected_edges and unselected_edges are now stale.
                        # We can break out and let the outer loop re-evaluate.
                        break
                    else:
                        # Revert
                        selected[e_out] = True
                        # e_in remains False
                        
                if not swap_success:
                    # If no swap worked, we are stuck.
                    # We can break out of local search and return to main loop for perturbation.
                    break
                    
        return selected, count

    # Main ILS Loop
    current_selected, current_count = best_selected, best_count
    
    while time.time() < deadline:
        # 1. Local Search
        current_selected, current_count = local_search(current_selected, current_count)
        
        if current_count > best_count:
            best_count = current_count
            best_selected = current_selected[:]
            
        # 2. Perturbation
        # Remove k random edges
        k = random.randint(3, 15)
        selected_edges = [i for i in range(num_edges) if current_selected[i]]
        if len(selected_edges) <= k:
            # If too few, just restart greedy
            current_selected, current_count = greedy_init()
            continue
            
        to_remove = random.sample(selected_edges, k)
        for e in to_remove:
            current_selected[e] = False
            current_count -= 1
            
        # 3. Greedy Repair
        # Try to add back edges in random order
        edges_order = list(range(num_edges))
        random.shuffle(edges_order)
        
        for e in edges_order:
            if not current_selected[e]:
                ok = True
                for s_idx in squares_by_edge[e]:
                    sq = edges_in_square[s_idx]
                    other_selected = 0
                    for oe in sq:
                        if oe != e and current_selected[oe]:
                            other_selected += 1
                    if other_selected == 3:
                        ok = False
                        break
                if ok:
                    current_selected[e] = True
                    current_count += 1
        
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_selected = current_selected[:]
            
    # Convert best_selected to list of edges [u, v]
    result = []
    for i in range(num_edges):
        if best_selected[i]:
            result.append([edge_u[i], edge_v[i]])
            
    return result

# Test locally
if __name__ == "__main__":
    edges = construct()
    print(f"Number of edges: {len(edges)}")
    # Verify no C4
    # ... (verification logic omitted for brevity in thought, included in code if needed)
