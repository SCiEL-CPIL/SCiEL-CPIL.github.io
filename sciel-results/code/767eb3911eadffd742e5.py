import time
import sys

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # State:
    # adj is a list of n integers, where adj[i] is a bitmask of neighbors of i.
    # We will solve for the edges.
    # To use CSP, we define variables as vertices.
    # Domain of variable i: subset of {0, ..., i-1} (to avoid symmetry, we only consider edges to lower indices)
    # However, standard CSP usually fixes the order. 
    # A better CSP approach for graphs:
    # Variables: vertices 0..n-1
    # Value: adjacency mask to ALL other vertices? No, that's symmetric.
    # Let's stick to the triangular structure but use MCV/AC-3 style propagation.
    
    # Actually, a powerful CSP approach for this specific problem (K2,3-free) is:
    # Variables: v_0, ..., v_{n-1}
    # Domain: D_i = {mask | mask is a subset of {0..i-1}}
    # Constraint: For any i < j, if j in D_i (i.e., edge exists), then for all k < i, 
    # if k in D_i and k in D_j, then common(i, k) < 2 and common(j, k) < 2.
    
    # This is still complex. Let's use a different structural change:
    # **Integer Linear Programming (ILP) relaxation or local search on the complement?**
    # No, let's use **Simulated Annealing with a better move set** or **Genetic Algorithm**?
    
    # The prompt asks for a MATERIAL DIFFERENCE.
    # Current: Backtracking DFS.
    # Inspiration 1: Symmetry breaking.
    # Inspiration 2: Local search.
    # Inspiration 3: SA.
    
    # Let's try **Simulated Annealing** but with a **Bitmask-based** representation and **Fast Validation**.
    # The current program uses a slow Python loop for validation.
    # A structural change: Use **NumPy** for vectorized operations if possible? 
    # Or, more importantly, use **Randomized Local Search (SLA - Simulated Annealing)** 
    # which explores the space of *all* graphs, not just those built sequentially.
    # The current DFS is trapped in the "build order" structure.
    # SLA can jump between non-sequential structures.
    
    # Strategy:
    # 1. Start with a random valid graph (or greedy).
    # 2. Define moves: Add edge, Remove edge, Swap edge.
    # 3. Objective: Maximize edges.
    # 4. Constraint: No K2,3.
    # 5. Use Simulated Annealing to escape local optima.
    
    # Key optimization: Fast constraint checking.
    # We maintain `common_neighbors[u][v]` counts? 
    # If we add edge (u, v), it increases common neighbors for all pairs (u, w) and (v, w) where w is common neighbor?
    # No.
    # Adding edge (u, v):
    # For any w != u, v:
    #   common(u, w) increases if v is neighbor of w? No.
    #   common(u, w) is |N(u) ∩ N(w)|.
    #   Adding (u, v) adds v to N(u) and u to N(v).
    #   So for any w:
    #     If v in N(w), then common(u, w) increases by 1 (because v is now in N(u) and was already in N(w)).
    #     If u in N(w), then common(v, w) increases by 1.
    #   We must ensure that for all w, if common(u, w) was 2, it doesn't become 3.
    #   So we need: For all w in N(v), common(u, w) < 2.
    #   And: For all w in N(u), common(v, w) < 2.
    
    # This check is O(deg(u) + deg(v)).
    # With n=20, max deg is 19. This is very fast.
    
    import random
    
    def count_edges(adj):
        # adj is list of masks
        total = 0
        for i in range(n):
            total += adj[i].bit_count()
        return total // 2

    def is_valid_add(adj, u, v):
        if u == v: return False
        if adj[u] & (1 << v): return True # already exists, valid state
        # Check if adding (u, v) creates K2,3
        # 1. For all w in N(v), common(u, w) must be < 2
        mask_v = adj[v]
        w = mask_v
        while w:
            lsb = w & -w
            w_idx = lsb.bit_length() - 1
            w -= lsb
            # common(u, w_idx) = bit_count(adj[u] & adj[w_idx])
            if (adj[u] & adj[w_idx]).bit_count() >= 2:
                return False
        
        # 2. For all w in N(u), common(v, w) must be < 2
        mask_u = adj[u]
        w = mask_u
        while w:
            lsb = w & -w
            w_idx = lsb.bit_length() - 1
            w -= lsb
            if (adj[v] & adj[w_idx]).bit_count() >= 2:
                return False
                
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Initialize with a greedy graph
    adj = [0] * n
    # Randomize vertex order for diversity? No, keep 0..n-1 but randomize edge addition order?
    # Let's just use a simple greedy first to get a decent start.
    for i in range(n):
        for j in range(i):
            if is_valid_add(adj, i, j):
                add_edge(adj, i, j)
    
    best_adj = adj[:]
    best_count = count_edges(adj)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    alpha = 0.999
    steps = 0
    max_steps = 1000000 # Limit steps, but time is the real constraint
    
    current_adj = adj[:]
    current_count = best_count
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i)]
    
    while time.time() < deadline:
        # Pick a random edge to flip
        u, v = random.choice(all_edges)
        
        if adj[u] & (1 << v):
            # Currently present. Try to remove.
            # Removing always maintains validity.
            # But we want to maximize, so removing is only good if it allows adding more later?
            # In SA, we accept worse states with probability exp(-delta/T).
            # Here, removing decreases count by 1.
            # We might do this to escape a local optimum where we are stuck.
            remove_edge(current_adj, u, v)
            new_count = current_count - 1
            delta = new_count - current_count # -1
            current_adj = current_adj[:] # Copy for simplicity, or modify in place and revert
            # Actually, modify in place is faster.
            # Let's use a single adj array and revert.
            
            # Wait, the logic above is messy. Let's do:
            # 1. Generate move
            # 2. Apply move
            # 3. Calculate new score
            # 4. Accept or revert
            
            # Re-implementing loop body properly:
            
            pass # Placeholder, see below
        
        # Correct SA Loop
        break # Exit this while, use the one below

    # Proper SA Implementation
    current_adj = best_adj[:]
    current_count = best_count
    
    # To make SA effective, we need to allow "bad" moves (removing edges) to explore.
    # But our objective is strictly maximizing edges.
    # A "bad" move is removing an edge.
    # A "good" move is adding an edge.
    
    # We can bias the move selection:
    # 50% chance to try to add a missing edge.
    # 50% chance to try to remove an existing edge.
    
    missing_edges = []
    existing_edges = []
    for u, v in all_edges:
        if current_adj[u] & (1 << v):
            existing_edges.append((u, v))
        else:
            missing_edges.append((u, v))
    
    # This list maintenance is O(E) per step, which is too slow for 1M steps.
    # Better: Pick random u, v.
    
    while time.time() < deadline:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        
        is_present = bool(current_adj[u] & (1 << v))
        
        if is_present:
            # Try to remove
            remove_edge(current_adj, u, v)
            new_count = current_count - 1
            delta = -1
            # Always accept removal? No, that leads to empty graph.
            # Accept with probability exp(delta/T).
            if random.random() < (2.718281828 ** (delta / T)):
                current_count = new_count
                # Keep removal
            else:
                # Revert
                add_edge(current_adj, u, v)
                # current_count unchanged
        else:
            # Try to add
            if is_valid_add(current_adj, u, v):
                add_edge(current_adj, u, v)
                new_count = current_count + 1
                delta = 1
                # Always accept valid addition? 
                # Yes, because it improves the objective and maintains constraint.
                current_count = new_count
            else:
                # Invalid addition, do nothing
                pass
        
        # Update best
        if current_count > best_count:
            best_count = current_count
            best_adj = current_adj[:]
            
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            # Restart with higher T? Or just stop?
            # Let's do a restart to explore new regions.
            # But we need to keep the best solution.
            # Reset current to best? Or random?
            # Reset to best to exploit, or random to explore.
            # Let's mix: 50% reset to best, 50% random restart from best.
            if random.random() < 0.5:
                current_adj = best_adj[:]
                current_count = best_count
            else:
                # Perturb best
                current_adj = best_adj[:]
                # Remove a few random edges
                for _ in range(5):
                    if not existing_edges: break
                    # Pick random edge from best_adj
                    # Need list of existing edges in current_adj
                    # Just pick random u,v and if present, remove
                    ru = random.randint(0, n-1)
                    rv = random.randint(0, n-1)
                    if ru == rv: continue
                    if ru > rv: ru, rv = rv, ru
                    if current_adj[ru] & (1 << rv):
                        remove_edge(current_adj, ru, rv)
                        current_count -= 1
            T = 1.0 # Reset temperature
            
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        mask = best_adj[u]
        k = mask
        while k:
            lsb = k & -k
            v = lsb.bit_length() - 1
            k -= lsb
            if u < v:
                edges.append([u, v])
                
    return edges

# Test locally
if __name__ == "__main__":
    start = time.time()
    edges = construct()
    end = time.time()
    print(f"Time: {end-start:.2f}s, Edges: {len(edges)}")
    # Verify validity
    n = 20
    adj = [0]*n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    valid = True
    for u in range(n):
        for v in range(u+1, n):
            common = (adj[u] & adj[v]).bit_count()
            if common >= 3:
                valid = False
                print(f"Invalid: {u}, {v} have {common} common neighbors")
    print(f"Valid: {valid}")
