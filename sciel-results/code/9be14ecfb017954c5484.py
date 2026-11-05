import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # State
    adj = [0] * n  # Bitmasks
    common = [[0] * n for _ in range(n)]  # Common neighbor counts
    edges = set()
    edge_count = 0
    
    def update_common_on_add(u, v):
        """Update common counts when adding edge (u, v)."""
        # u and v become neighbors
        # For any w != u, v:
        # If w ~ v, then v is a common neighbor of u and w? No.
        # Common neighbors of u and w are x such that x~u and x~w.
        # Adding u-v:
        # v is now ~ u.
        # If v ~ w, then v is a common neighbor of u and w.
        # So common[u][w] increases by 1 if v ~ w.
        # Similarly, u is now ~ v.
        # If u ~ w, then u is a common neighbor of v and w.
        # So common[v][w] increases by 1 if u ~ w.
        
        mask_v = adj[v]
        mask_u = adj[u]
        
        # Iterate over all w != u, v
        for w in range(n):
            if w == u or w == v:
                continue
            if mask_v & (1 << w):
                common[u][w] += 1
                common[w][u] += 1
            if mask_u & (1 << w):
                common[v][w] += 1
                common[w][v] += 1

    def update_common_on_remove(u, v):
        """Update common counts when removing edge (u, v)."""
        # Reverse of add
        mask_v = adj[v]
        mask_u = adj[u]
        
        for w in range(n):
            if w == u or w == v:
                continue
            if mask_v & (1 << w):
                common[u][w] -= 1
                common[w][u] -= 1
            if mask_u & (1 << w):
                common[v][w] -= 1
                common[w][v] -= 1

    def can_add(u, v):
        """Check if edge (u, v) can be added to the current graph."""
        if u == v:
            return False
        if adj[u] & (1 << v):
            return False
        
        # Check current common neighbors of u and v
        if common[u][v] > 2:
            return False
            
        # Check impact on other pairs
        # For any w != u, v:
        # If v ~ w, common[u][w] will increase by 1. Must be < 2 currently.
        # If u ~ w, common[v][w] will increase by 1. Must be < 2 currently.
        
        mask_v = adj[v]
        mask_u = adj[u]
        
        for w in range(n):
            if w == u or w == v:
                continue
            if mask_v & (1 << w):
                if common[u][w] >= 2:
                    return False
            if mask_u & (1 << w):
                if common[v][w] >= 2:
                    return False
        return True

    def add_edge(u, v):
        nonlocal edge_count
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edges.add((u, v))
        edge_count += 1
        update_common_on_add(u, v)

    def remove_edge(u, v):
        nonlocal edge_count
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        edges.remove((u, v))
        edge_count -= 1
        update_common_on_remove(u, v)

    def greedy_fill():
        """Add all possible edges."""
        improved = True
        while improved:
            improved = False
            for u, v in all_pairs:
                if can_add(u, v):
                    add_edge(u, v)
                    improved = True

    def local_search(max_iters=1000):
        """Local search with 1-1 and 1-2 swaps."""
        best_count = edge_count
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            greedy_fill()
            
            if edge_count > best_count:
                best_count = edge_count
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    break

            # Try 1-1 swaps
            found_improvement = False
            # Randomize order of edges to remove
            current_edges = list(edges)
            random.shuffle(current_edges)
            
            for u, v in current_edges:
                # Remove u-v
                remove_edge(u, v)
                
                # Try to add x-y
                # We want to find an x-y that is valid and increases count (net 0 change from swap, but maybe allows further fills?)
                # Actually, a 1-1 swap keeps count same. It's useful to escape local optima.
                # But we primarily want to INCREASE count.
                # So we should focus on 1-2 swaps (remove 1, add 2) which increases count by 1.
                
                # Try to add two edges
                found_pair = False
                for x, y in all_pairs:
                    if x == u or x == v or y == u or y == v:
                        continue
                    if can_add(x, y):
                        add_edge(x, y)
                        # Now try to add a second edge
                        for z, w in all_pairs:
                            if z == x or z == y or w == x or w == y:
                                continue
                            if z == u or z == v or w == u or w == v:
                                continue
                            if can_add(z, w):
                                add_edge(z, w)
                                found_pair = True
                                break
                        if found_pair:
                            break
                        
                        if not found_pair:
                            remove_edge(x, y)
                
                if found_pair:
                    found_improvement = True
                    # Don't break, keep trying? Or break to restart greedy?
                    # Let's break to let greedy_fill run again
                    break
                else:
                    # If we couldn't add 2, maybe we can add 1?
                    # But that results in net 0 change (remove 1, add 1).
                    # Let's try to add 1 to see if it unlocks something.
                    added_one = False
                    for x, y in all_pairs:
                        if x == u or x == v or y == u or y == v:
                            continue
                        if can_add(x, y):
                            add_edge(x, y)
                            added_one = True
                            break
                    if not added_one:
                        # Restore original state
                        add_edge(u, v)
                    else:
                        # We are in a state with same edge count but different structure.
                        # Accept this move to escape local optimum.
                        pass
                        
            if found_improvement:
                no_improve_count = 0

        return best_count

    # Main loop: Iterated Local Search
    best_overall = 0
    best_edges_overall = set()
    
    # Initial randomization
    random.seed(42)
    
    # Try a few different initial strategies
    strategies = [
        "random_sparse",
        "greedy_random_order",
        "empty"
    ]
    
    for strat in strategies:
        if time.time() > deadline - 10.0:
            break
            
        # Reset state
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = set()
        edge_count = 0
        
        if strat == "random_sparse":
            # Add ~20 random edges
            for _ in range(20):
                u, v = random.choice(all_pairs)
                if can_add(u, v):
                    add_edge(u, v)
        elif strat == "greedy_random_order":
            shuffled_pairs = list(all_pairs)
            random.shuffle(shuffled_pairs)
            for u, v in shuffled_pairs:
                if can_add(u, v):
                    add_edge(u, v)
        
        # Local Search
        local_search(max_iters=200)
        
        if edge_count > best_overall:
            best_overall = edge_count
            best_edges_overall = set(edges)
            
        # Perturb and continue
        if time.time() > deadline - 10.0:
            break
            
        # Perform several perturbations from the best state
        # Restore best state
        # (Simpler: just run ILS loop)
        
    # ILS Loop
    if time.time() < deadline - 20.0:
        # Restore best state found so far
        # We need to re-construct adj/common from best_edges_overall
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = set()
        edge_count = 0
        
        for u, v in best_edges_overall:
            # Just add blindly, assuming it was valid
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add((u, v))
            edge_count += 1
            update_common_on_add(u, v)
            
        # Run ILS
        for i in range(100): # Max iterations
            if time.time() > deadline - 5.0:
                break
                
            # Local Search
            local_search(max_iters=50)
            
            if edge_count > best_overall:
                best_overall = edge_count
                best_edges_overall = set(edges)
                
            # Perturbation: Remove 3-5 random edges
            num_remove = random.randint(3, 5)
            current_edges = list(edges)
            random.shuffle(current_edges)
            for u, v in current_edges[:num_remove]:
                remove_edge(u, v)
                
            # If we removed edges, we might be able to add better ones
            # The next local_search will handle it

    # Convert set to list of lists
    result = [[u, v] for u, v in best_edges_overall]
    return result

# Quick sanity check
if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
