import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # Represent adjacency as a list of integers (bitsets)
    # adj[i] is an integer where bit j is 1 if (i, j) is an edge.
    adj = [0] * n
    
    # Helper to add edge
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Helper to remove edge
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    # Helper to check if edge exists
    def has_edge(u, v):
        return (adj[u] >> v) & 1
        
    # Helper to get neighbors as a list (only when needed, e.g., for iteration)
    def get_neighbors(u):
        bits = adj[u]
        neighbors = []
        while bits:
            lsb = bits & -bits
            v = lsb.bit_length() - 1
            neighbors.append(v)
            bits ^= lsb
        return neighbors

    # Check if adding edge (u, v) creates a C6
    # A C6 containing (u, v) implies a path u-x-y-z-v of length 4 exists in G - (u, v).
    # Since we are checking before adding, we just check if such a path exists in current G.
    # Path: u - x - y - z - v
    # x in N(u), z in N(v), y in N(x) & N(z).
    # Distinctness: u, v, x, y, z must be distinct.
    # x != v (since u,v not connected yet, x in N(u) could be v? No, if u,v not connected, v not in N(u) unless we are checking a non-edge. 
    # Wait, if we are checking a non-edge (u,v), then v is NOT in N(u) and u is NOT in N(v).
    # So x != v and z != u is automatic.
    # We need x != z, x != y, z != y.
    # y != u, y != v is required for simple path.
    
    def would_create_c6(u, v):
        nu = adj[u]
        nv = adj[v]
        
        # Iterate over x in N(u)
        # We can iterate over bits of nu
        # Optimization: If we find one valid x, we can return early? No, we need to check all x to be sure? 
        # No, if ANY x leads to a C6, adding the edge creates a C6. So we return True if we find one.
        
        bits_x = nu
        while bits_x:
            lsb_x = bits_x & -bits_x
            x = lsb_x.bit_length() - 1
            bits_x ^= lsb_x
            
            # x cannot be v (already ensured by non-edge status, but safe to check)
            if x == v:
                continue
                
            # Iterate over z in N(v)
            bits_z = nv
            while bits_z:
                lsb_z = bits_z & -bits_z
                z = lsb_z.bit_length() - 1
                bits_z ^= lsb_z
                
                if z == u:
                    continue
                if z == x:
                    continue
                
                # Check if x and z have a common neighbor y
                # y in N(x) & N(z)
                common = adj[x] & adj[z]
                
                # We need y such that y != u, y != v, y != x, y != z.
                # Mask out u, v, x, z from common
                mask = ~( (1 << u) | (1 << v) | (1 << x) | (1 << z) )
                valid_y = common & mask
                
                if valid_y != 0:
                    return True
        return False

    # Initial Construction: 8 disjoint K5s
    # This gives 8 * 10 = 80 edges.
    # We can try to add inter-cluster edges.
    # Let's start with just K5s and use local search to add/swap.
    
    def init_graph():
        adj = [0] * n
        for k in range(8):
            base = 5 * k
            for a in range(5):
                for b in range(a + 1, 5):
                    u = base + a
                    v = base + b
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_edges(adj):
        count = 0
        for i in range(n):
            count += bin(adj[i]).count('1')
        return count // 2

    best_adj = None
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    random.seed(42)
    
    # Local Search Strategy:
    # 1. Start with a random valid graph (or the K5 base).
    # 2. Try to ADD an edge if it doesn't create C6.
    # 3. If stuck (no add possible), try SWAPS: remove edge e1, add edge e2.
    #    Or remove e1, add e2, remove e3... (2-opt style).
    #    Simplest: 1-swap. Remove e1, try to add e2. If e2 was invalid before, it might be valid now.
    #    Actually, removing an edge never creates a C6. So if we remove e1, the graph is still C6-free.
    #    Then we try to add any non-edge e2. If it fits, we have same number of edges.
    #    This doesn't increase edge count!
    #    
    #    To increase edge count, we need to ADD an edge.
    #    If no single edge can be added, we are at a local maximum for 1-add moves.
    #    To escape, we can do a "2-exchange": Remove e1, Remove e2, Add e3, Add e4.
    #    Or Remove e1, Add e2 (neutral), then Add e3 (gain).
    #    
    #    Let's use a "Random Restart with Greedy Add" first, as it's simpler and effective.
    #    If greedy add gets stuck, we perturb by removing a few random edges and re-adding greedily.
    
    max_restarts = 100
    for r in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        adj = init_graph()
        current_count = count_edges(adj)
        
        # Greedy Phase
        # Generate list of non-edges
        present_set = set()
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    present_set.add((u, v))
        
        non_edges = [e for e in all_edges if e not in present_set]
        
        added_any = True
        # We will iterate through non_edges multiple times if we keep adding
        # But adding an edge changes the set of non_edges and validity of others.
        # Simple approach: Shuffle non_edges. Iterate. If add succeeds, restart scan? 
        # Or just do one pass, then another pass on remaining non-edges.
        
        # To be efficient, we maintain a list of candidates.
        candidates = non_edges.copy()
        random.shuffle(candidates)
        
        while True:
            if time.time() > deadline - 10:
                break
            
            added_in_pass = False
            new_candidates = []
            
            for u, v in candidates:
                if (adj[u] >> v) & 1:
                    continue # Already added
                if not would_create_c6(u, v):
                    add_edge(adj, u, v)
                    current_count += 1
                    added_in_pass = True
                else:
                    new_candidates.append((u, v))
            
            if not added_in_pass:
                break
            candidates = new_candidates
        
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            
        # Perturbation for next restart:
        # Remove 2-5 random edges from current best (or current state) to allow different structure
        # We'll perturb the current state before next restart
        if current_count > 0:
            # Get list of current edges
            current_edges = []
            for u in range(n):
                bits = adj[u]
                while bits:
                    lsb = bits & -bits
                    v = lsb.bit_length() - 1
                    if v > u:
                        current_edges.append((u, v))
                    bits ^= lsb
            
            # Remove k random edges
            k = random.randint(2, 5)
            random.shuffle(current_edges)
            for e in current_edges[:k]:
                remove_edge(adj, e[0], e[1])
                current_count -= 1
                
            # Store this perturbed state as the start for the next greedy run?
            # Yes, by not re-initializing adj to K5s in the next loop iteration,
            # but using the perturbed adj.
            # Wait, the loop starts with adj = init_graph().
            # I need to change the loop structure to allow perturbation from the previous best/current.
            
    # The above loop re-initializes to K5s every time.
    # Let's refine: 
    # 1. Start with K5s.
    # 2. Greedy add.
    # 3. If stuck, perturb (remove random edges) and Greedy add again.
    # 4. Repeat until time limit.
    
    # Revised Search Loop
    adj = init_graph()
    current_count = count_edges(adj)
    best_count = current_count
    best_adj = adj[:]
    
    # We will do a sequence of "Greedy Add" phases separated by "Perturbations"
    # To speed up, we'll keep a list of "blocked" edges (non-edges that created C6).
    # When we remove an edge, some blocked edges might become unblocked.
    
    # For simplicity and robustness within time limits, we'll do:
    # 1. Greedy Add on current graph.
    # 2. If no edges added, remove 3 random edges.
    # 3. Repeat.
    
    # We need to efficiently find addable edges.
    # Checking all 780 edges every time is O(780 * Cost(C6)).
    # Cost(C6) is roughly O(deg^2). Deg is ~10-20. So ~400 ops.
    # 780 * 400 = 312,000 ops per pass.
    # In Python, this might be ~0.1s per pass.
    # We have ~140s. So ~1400 passes.
    # Each pass adds ~1-5 edges or removes 3.
    # This should be sufficient to explore the space.
    
    while time.time() < deadline - 5:
        # Pass: Try to add any edge
        added_any = False
        # Randomize order to avoid bias
        candidates = all_edges.copy()
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (adj[u] >> v) & 1:
                continue
            if not would_create_c6(u, v):
                add_edge(adj, u, v)
                current_count += 1
                added_any = True
                if current_count > best_count:
                    best_count = current_count
                    best_adj = adj[:]
        
        if not added_any:
            # Stuck. Perturb.
            # Remove 2-4 random edges to open up space
            current_edges = []
            for u in range(n):
                bits = adj[u]
                while bits:
                    lsb = bits & -bits
                    v = lsb.bit_length() - 1
                    if v > u:
                        current_edges.append((u, v))
                    bits ^= lsb
            
            if not current_edges:
                break # Empty graph? Unlikely.
                
            k = random.randint(2, 4)
            random.shuffle(current_edges)
            for e in current_edges[:k]:
                remove_edge(adj, e[0], e[1])
                current_count -= 1
            # Note: We don't update best here because count decreased.
            # But we might find a better configuration later.
            
    # Convert best_adj to list of edges
    result_edges = []
    for u in range(n):
        bits = best_adj[u]
        while bits:
            lsb = bits & -bits
            v = lsb.bit_length() - 1
            if v > u:
                result_edges.append([u, v])
            bits ^= lsb
            
    return result_edges

# EVOLVE-BLOCK-END
