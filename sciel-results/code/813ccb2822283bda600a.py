# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 256
    nums = list(range(1, N + 1))

    # 1. Precompute the conflict graph
    # conflict[a][b] is True if (a*b) % (a+b) == 0
    # We want to find the Maximum Independent Set (MIS) in this graph.
    # MIS in G is equivalent to Maximum Clique in the complement graph G'.
    
    # Using bitmasks for efficiency.
    # adj[i] will be a bitmask of nodes that CONFLICT with i.
    # comp_adj[i] will be a bitmask of nodes that DO NOT CONFLICT with i (and i != j).
    
    adj = [0] * (N + 1)
    comp_adj = [0] * (N + 1)
    
    # Initialize comp_adj with all bits set, then clear self and conflicts
    # Actually, it's faster to build adj first, then derive comp_adj.
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
    # Build complement adjacency for the clique search
    # In the complement graph, an edge exists if there is NO conflict in G.
    # comp_adj[i] has bit j set iff i != j and (i,j) is NOT in conflict.
    for i in range(1, N + 1):
        # All other nodes
        all_others = ((1 << (N + 1)) - 1) ^ (1 << i)
        # Nodes that do NOT conflict with i
        comp_adj[i] = all_others & ~adj[i]

    # 2. Tomita's Algorithm for Maximum Clique
    # We represent the set of candidate vertices as a Python integer (bitmask).
    # We use a recursive branch-and-bound with greedy coloring for upper bounds.
    
    best_clique_size = 0
    best_clique_mask = 0
    
    # Global variables for the search
    # We need to track the best solution found so far
    
    def color_sort(P):
        """
        Greedy coloring of the vertices in P (bitmask).
        Returns a list of vertices sorted by their color class (descending color number).
        The number of colors used is the upper bound on the clique size in P.
        """
        if P == 0:
            return [], 0
        
        # Extract vertices from P
        verts = []
        temp = P
        while temp:
            bit = temp & (-temp)
            v = bit.bit_length() - 1
            verts.append(v)
            temp ^= bit
            
        # For better bounds, we should order vertices by degree in the subgraph induced by P
        # But standard Tomita uses a simple order or degree-based order.
        # Let's use a simple greedy coloring:
        # 1. Initialize color classes as empty sets (bitmasks)
        # 2. For each vertex v in verts (maybe sorted by degree in P?):
        #    - Find the smallest color class c such that v has no neighbors in c within P.
        #    - If none, create a new color class.
        
        # To make the upper bound tighter, we often process vertices in order of decreasing degree in the induced subgraph.
        # However, calculating degrees for every node in every recursive call is expensive.
        # A common heuristic is to use the order from the parent call or a static order.
        # Let's try a simple greedy: process in the order they appear in P (low to high bit).
        # Actually, a better heuristic for coloring bound is to pick vertices with high degree first.
        # Let's compute degrees in P for the vertices in P.
        
        # Optimization: If P is small, just do it.
        # For now, let's use a standard greedy coloring.
        
        color_classes = [] # List of bitmasks
        color_of = {} # v -> color index
        
        # Sort vertices by degree in the subgraph induced by P (descending)
        # degree(v) = popcount(comp_adj[v] & P)
        # This is O(|P|^2) if done naively, but bit ops are fast.
        
        # Let's just use the natural order for simplicity first, or maybe reverse?
        # Tomita's original paper uses a specific ordering.
        # Let's try sorting by degree in P descending.
        
        # Calculate degrees
        degs = []
        for v in verts:
            deg = (comp_adj[v] & P).bit_count()
            degs.append((deg, v))
        
        # Sort by degree descending
        degs.sort(reverse=True)
        
        for _, v in degs:
            v_bit = 1 << v
            # Check existing color classes
            placed = False
            for ci, cc in enumerate(color_classes):
                # v can join color class ci if v has no neighbors in cc within P
                # Neighbors of v in P are comp_adj[v] & P
                # We need (comp_adj[v] & P) & cc == 0
                if (comp_adj[v] & P) & cc == 0:
                    color_classes[ci] |= v_bit
                    color_of[v] = ci
                    placed = True
                    break
            if not placed:
                color_classes.append(v_bit)
                color_of[v] = len(color_classes) - 1
        
        num_colors = len(color_classes)
        
        # Return vertices sorted by color (descending color number, so high colors first)
        # This helps in the B&B: we explore vertices with "tighter" bounds first?
        # Actually, we return the list of vertices in the order they were colored?
        # No, we return them sorted by color index descending.
        # Vertices with higher color indices are in "later" color classes.
        # In Tomita's algorithm, we process vertices in reverse order of their color assignment?
        # Let's return the list of vertices sorted by color_of[v] descending.
        
        sorted_verts = sorted(verts, key=lambda x: color_of[x], reverse=True)
        
        return sorted_verts, num_colors

    def expand(R_size, R_mask, P):
        """
        Recursive expansion for Maximum Clique.
        R_size: size of current clique
        R_mask: bitmask of current clique
        P: bitmask of candidate vertices
        """
        nonlocal best_clique_size, best_clique_mask
        
        if P == 0:
            if R_size > best_clique_size:
                best_clique_size = R_size
                best_clique_mask = R_mask
            return

        if time.time() > deadline:
            return

        # Get sorted vertices and upper bound
        sorted_verts, ub = color_sort(P)
        
        # If upper bound + current size <= best, prune
        if R_size + ub <= best_clique_size:
            return

        # Iterate through vertices in sorted order
        # We need to maintain the candidate set P as we go
        # Standard Tomita:
        # for v in sorted_verts:
        #   if v is in P:
        #      P = P \ {v}
        #      if R_size + 1 + popcount(P & comp_adj[v]) <= best: continue
        #      expand(R_size+1, R_mask | (1<<v), P & comp_adj[v])
        
        current_P = P
        for v in sorted_verts:
            v_bit = 1 << v
            if not (current_P & v_bit):
                continue
            
            # Remove v from current_P
            current_P ^= v_bit
            
            # New candidates are neighbors of v in the current P
            new_P = current_P & comp_adj[v]
            
            # Upper bound check for this branch
            # The max clique size in this branch is R_size + 1 + |new_P|
            # A tighter bound is R_size + 1 + color_bound(new_P)
            # But calculating color_bound again is expensive.
            # Simple bound: R_size + 1 + new_P.bit_count()
            if R_size + 1 + new_P.bit_count() <= best_clique_size:
                continue
            
            expand(R_size + 1, R_mask | v_bit, new_P)

    # Initialize with a greedy solution to have a good lower bound
    # Greedy: pick vertex with highest degree in complement? Or just iterate?
    # Let's do a simple greedy on the complement graph (i.e., pick non-conflicting nodes)
    # Actually, let's just run the B&B from scratch, but a good initial bound helps.
    
    # Greedy Initial Solution
    init_P = (1 << (N + 1)) - 2  # bits 1 to N set
    current_clique = 0
    current_size = 0
    temp_P = init_P
    while temp_P:
        # Pick vertex with max degree in temp_P?
        # For simplicity, pick the first one.
        bit = temp_P & (-temp_P)
        v = bit.bit_length() - 1
        current_clique |= bit
        current_size += 1
        # Update candidates: must be neighbors of v in complement
        temp_P = temp_P & comp_adj[v]
        
    best_clique_size = current_size
    best_clique_mask = current_clique

    # Run Tomita's Algorithm
    # We start with R = empty, P = all vertices
    # Note: The order of expansion matters.
    # We call expand(0, 0, init_P)
    
    # One optimization: We can order the initial vertices by degree in complement descending.
    # But expand handles the sorting internally via color_sort.
    
    expand(0, 0, init_P)

    # Convert best_clique_mask to a list
    result = []
    mask = best_clique_mask
    while mask:
        bit = mask & (-mask)
        v = bit.bit_length() - 1
        result.append(v)
        mask ^= bit
    
    # Verify the result (safety check)
    # If verification fails, fall back to a simple valid set (e.g., all odds)
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
            
    if not valid or len(result) == 0:
        # Fallback: All odds are a valid set (size 128)
        return list(range(1, N + 1, 2))
        
    return result
# EVOLVE-BLOCK-END
