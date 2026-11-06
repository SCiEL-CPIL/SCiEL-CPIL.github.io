import time
import random
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute vertex index from bitmask lsb
    def lsb_index(mask):
        return mask.bit_length() - 1
    
    # Precompute all pairs for potential edges
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # State: adj[i] is bitmask of neighbors
    adj = [0] * n
    
    # common[u][v] = number of common neighbors of u and v
    # We maintain this incrementally for O(1) validity checks
    # common is a 20x20 list of ints
    common = [[0] * n for _ in range(n)]
    
    # Validity check for adding edge (u, v):
    # Adding (u, v) increases common[u][w] by 1 for all w in adj[v] (w != u)
    # and common[v][w] by 1 for all w in adj[u] (w != v)
    # Also common[u][v] increases by... wait, common[u][v] is count of w adj to both.
    # When we add (u,v), for any w, if w is adj to u and v, common[u][v] doesn't change directly from the edge (u,v) itself?
    # No. common[u][v] = |N(u) ∩ N(v)|.
    # If we add edge (u,v), u becomes a neighbor of v and v becomes a neighbor of u.
    # Does u become a common neighbor of u and v? No, simple graph, no self-loops.
    # Does the addition of (u,v) change N(u) ∩ N(v)?
    # N(u) gains v. N(v) gains u.
    # Intersection: (N(u) ∪ {v}) ∩ (N(v) ∪ {u}) = (N(u) ∩ N(v)) ∪ (N(u) ∩ {u}) ∪ ({v} ∩ N(v)) ∪ ({v} ∩ {u})
    # Since u not in N(u) and v not in N(v), and u!=v:
    # The intersection size does NOT change when adding edge (u,v).
    # 
    # However, for any other vertex w:
    # common[u][w] = |N(u) ∩ N(w)|.
    # N(u) gains v. So if v in N(w), common[u][w] increases by 1.
    # common[v][w] = |N(v) ∩ N(w)|.
    # N(v) gains u. So if u in N(w), common[v][w] increases by 1.
    
    # So to check if adding (u,v) is valid:
    # 1. For all w in adj[v], common[u][w] must be < 2 (since it will become common[u][w]+1)
    # 2. For all w in adj[u], common[v][w] must be < 2 (since it will become common[v][w]+1)
    # Note: w can be u or v?
    # If w=u: common[u][u] is undefined/0. We don't care about self-commons.
    # But adj[v] contains u? No, (u,v) is not added yet. So u is not in adj[v].
    # So w != u, v is guaranteed in the loops if we just iterate adj masks.
    
    def is_valid_add(u, v):
        # Check w in adj[v]
        mask = adj[v]
        while mask:
            lsb = mask & -mask
            w = lsb_index(lsb)
            mask -= lsb
            if common[u][w] >= 2:
                return False
        
        # Check w in adj[u]
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            w = lsb_index(lsb)
            mask -= lsb
            if common[v][w] >= 2:
                return False
        
        return True

    def add_edge(u, v):
        # Update adj
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update common
        # For w in adj[v] (before adding u, but u is not in adj[v] yet):
        # common[u][w] += 1
        # common[w][u] += 1
        mask = adj[v]
        while mask:
            lsb = mask & -mask
            w = lsb_index(lsb)
            mask -= lsb
            common[u][w] += 1
            common[w][u] += 1
            
        # For w in adj[u] (before adding v, but v is not in adj[u] yet):
        # common[v][w] += 1
        # common[w][v] += 1
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            w = lsb_index(lsb)
            mask -= lsb
            common[v][w] += 1
            common[w][v] += 1

    def remove_edge(u, v):
        # Update adj
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Update common (reverse of add)
        # Note: adj[v] now does NOT contain u.
        # We need to iterate neighbors of v (excluding u) and neighbors of u (excluding v).
        # This is exactly what adj[v] and adj[u] are now.
        
        mask = adj[v]
        while mask:
            lsb = mask & -mask
            w = lsb_index(lsb)
            mask -= lsb
            common[u][w] -= 1
            common[w][u] -= 1
            
        mask = adj[u]
        while mask:
            lsb = mask & -mask
            w = lsb_index(lsb)
            mask -= lsb
            common[v][w] -= 1
            common[w][v] -= 1

    def get_edge_count():
        # Sum degrees / 2
        return sum(m.bit_count() for m in adj) // 2

    def is_valid_graph():
        # Check all pairs
        for u in range(n):
            for v in range(u + 1, n):
                if common[u][v] >= 3:
                    return False
        return True

    # Strategy:
    # 1. Generate a good initial solution using a structured greedy or random restarts.
    # 2. Use Simulated Annealing / Local Search to improve.
    #    Moves: Add an edge, Remove an edge, Swap an edge (remove one, add another).
    #    To be efficient, we only consider valid moves.
    
    best_edges = []
    best_count = 0
    
    def update_best():
        nonlocal best_edges, best_count
        cnt = get_edge_count()
        if cnt > best_count:
            best_count = cnt
            # Reconstruct edges
            edges = []
            for u in range(n):
                mask = adj[u]
                while mask:
                    lsb = mask & -mask
                    v = lsb_index(lsb)
                    mask -= lsb
                    if u < v:
                        edges.append([u, v])
            best_edges = edges
            # Verify validity just in case
            if not is_valid_graph():
                print("ERROR: Invalid graph found in best update")
                # Reset to empty if invalid? Or just ignore?
                # Let's assume logic is correct.
            
    # Initial Solution: Greedy with randomization
    # We will try multiple random orderings and greedy fills.
    
    def generate_initial():
        # Reset state
        for i in range(n):
            adj[i] = 0
        for i in range(n):
            for j in range(n):
                common[i][j] = 0
                
        # Try a few random permutations of vertices and greedy add
        best_local_count = 0
        best_local_adj = None
        best_local_common = None
        
        for trial in range(50):
            # Random order
            order = list(range(n))
            random.shuffle(order)
            
            # Greedy add in this order
            # We need to simulate.
            # To avoid copying large arrays 50 times, we can just do it once and keep the best.
            # But we need to reset state for each trial.
            # Copying 20x20 ints is cheap.
            
            trial_adj = [0] * n
            trial_common = [[0] * n for _ in range(n)]
            
            # We need to implement add_edge that works on trial_adj/trial_common
            # Or we can just use the global ones and reset.
            # Let's use a helper class or just inline logic for speed.
            
            # Actually, let's just use the global adj/common and reset them for each trial.
            # Since we are in a loop, we must be careful.
            
            for i in range(n):
                adj[i] = 0
            for i in range(n):
                for j in range(n):
                    common[i][j] = 0
                    
            count = 0
            for u in order:
                # Try to connect u to all previous vertices in the order?
                # Or all vertices < u in the permutation?
                # Let's try connecting u to all v in order such that v comes before u.
                prev_vertices = []
                for v in order:
                    if v == u:
                        break
                    prev_vertices.append(v)
                
                # Try to add edges to all prev_vertices
                # We should try to add as many as possible.
                # Order of trying? Random or sorted?
                # Let's try random order of prev_vertices.
                random.shuffle(prev_vertices)
                
                for v in prev_vertices:
                    # Check validity with current state
                    # We need a check function that works on current adj/common
                    valid = True
                    # Check w in adj[v]
                    mask = adj[v]
                    while mask:
                        lsb = mask & -mask
                        w = lsb_index(lsb)
                        mask -= lsb
                        if common[u][w] >= 2:
                            valid = False
                            break
                    if valid:
                        # Check w in adj[u]
                        mask = adj[u]
                        while mask:
                            lsb = mask & -mask
                            w = lsb_index(lsb)
                            mask -= lsb
                            if common[v][w] >= 2:
                                valid = False
                                break
                    
                    if valid:
                        # Add edge
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                        # Update common
                        mask = adj[v]
                        while mask:
                            lsb = mask & -mask
                            w = lsb_index(lsb)
                            mask -= lsb
                            common[u][w] += 1
                            common[w][u] += 1
                            
                        mask = adj[u]
                        while mask:
                            lsb = mask & -mask
                            w = lsb_index(lsb)
                            mask -= lsb
                            common[v][w] += 1
                            common[w][v] += 1
                        count += 1
            
            if count > best_local_count:
                best_local_count = count
                # Copy state
                best_local_adj = adj[:]
                best_local_common = [row[:] for row in common]
                
            if time.time() > deadline - 60:
                break
        
        # Restore best local state
        if best_local_adj:
            adj = best_local_adj
            common = best_local_common
            update_best()
        else:
            # Fallback: empty graph
            update_best()

    generate_initial()
    
    # Local Search: Simulated Annealing
    # Moves:
    # 1. Add a random valid edge.
    # 2. Remove a random edge.
    # 3. Swap: Remove a random edge, then try to add a different random edge.
    
    # To make it faster, we can precompute list of missing edges.
    # But it changes.
    
    current_count = get_edge_count()
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999
    
    T = T_start
    
    # List of all possible edges
    # We will pick random non-edges to try adding.
    # And random edges to try removing.
    
    # To speed up random selection:
    # Maintain a list of missing edges?
    # Or just pick random u, v and check if edge exists.
    # n=20, 190 edges. Picking random pair is O(1).
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to add an edge
            u, v = random.sample(range(n), 2)
            if not (adj[u] & (1 << v)):
                if is_valid_add(u, v):
                    add_edge(u, v)
                    new_count = current_count + 1
                    # Always accept improvements
                    current_count = new_count
                    update_best()
                else:
                    # Invalid, do nothing? Or try to swap?
                    # For SA, we might accept invalid? No, we must maintain validity.
                    # So we can't add invalid edges.
                    # We can only add valid ones.
                    # So "Add" move only succeeds if valid.
                    # To allow exploration, we need "Remove" and "Swap".
                    pass
        elif move_type < 0.8:
            # Try to remove an edge
            # Pick a random existing edge
            # Find one
            # We can pick random u, v and check if edge exists
            for _ in range(10): # Try a few times to find an edge
                u, v = random.sample(range(n), 2)
                if adj[u] & (1 << v):
                    remove_edge(u, v)
                    current_count -= 1
                    # Accept removal?
                    # This is a "hill climbing" step down.
                    # We should only do this if we are stuck or via SA probability.
                    # Since we are maximizing, removing is bad.
                    # We only remove to enable better additions later (Swap).
                    # So pure removal is rarely useful unless we are doing SA properly.
                    # Let's implement Swap instead.
                    # Put it back?
                    # Actually, let's just do Swap.
                    # Re-add it
                    add_edge(u, v)
                    current_count += 1
                    break
        else:
            # Swap: Remove (u,v), Add (x,y)
            # Pick random edge (u,v)
            found = False
            for _ in range(10):
                u, v = random.sample(range(n), 2)
                if adj[u] & (1 << v):
                    found = True
                    break
            
            if found:
                # Remove (u,v)
                remove_edge(u, v)
                current_count -= 1
                
                # Try to add a random non-edge (x,y)
                # We want to add an edge that is valid in the new state.
                # Also, we want to improve or stay same.
                # Since we removed one, we need to add one to keep count same, or add more?
                # We can only add one edge at a time in a simple swap.
                
                # Try a few random non-edges
                added = False
                for _ in range(20):
                    x, y = random.sample(range(n), 2)
                    if not (adj[x] & (1 << y)):
                        if is_valid_add(x, y):
                            add_edge(x, y)
                            current_count += 1
                            added = True
                            # Check if this is an improvement?
                            # Count is same as before swap.
                            # But maybe it opens up more edges?
                            # We don't know yet.
                            # We accept the swap if it doesn't decrease count (which it doesn't).
                            # But we should probably only accept if it leads to a better local optimum?
                            # For now, accept all valid swaps.
                            break
                
                if not added:
                    # Re-add (u,v) to restore state
                    add_edge(u, v)
                    current_count += 1
                    # State restored
                    pass
                else:
                    # State changed
                    # Check if we improved best?
                    # Count is same. But structure is different.
                    # update_best() only triggers on count > best_count.
                    # So it won't update unless count increased.
                    # This is a plateau move.
                    pass
        
        # Update Temperature
        T *= alpha
        
        # If we are stuck (no improvement in last X steps), we might need to cool down faster or restart?
        # Let's just let it run.
        
        # Periodic check for time
        if time.time() > deadline:
            break

    # Final check
    if not best_edges:
        # Fallback to greedy if SA failed completely
        for i in range(n):
            adj[i] = 0
        for i in range(n):
            for j in range(n):
                common[i][j] = 0
                
        for i in range(n):
            for j in range(i):
                if is_valid_add(i, j):
                    add_edge(i, j)
        update_best()

    return best_edges

if __name__ == "__main__":
    # Quick test
    import time
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, Edges: {len(edges)}")
    # Verify
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] & adj[v]).bit_count() >= 3:
                valid = False
                print(f"Invalid K2,3 at {u},{v}")
    print(f"Valid: {valid}")
