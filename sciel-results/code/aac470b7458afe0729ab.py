import time
import math
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s limit
    
    N = 256
    # 1. Precompute squares and conflicts
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # conflicts[i] is a bitmask of j such that (i+1) + (j+1) is a square
    # i, j in 0..255 representing values 1..256
    conflicts = [0] * N
    
    for i in range(N):
        val_i = i + 1
        mask = 0
        for j in range(N):
            val_j = j + 1
            if val_i + val_j in squares:
                mask |= (1 << j)
        conflicts[i] = mask

    # 2. Identify Forbidden Vertices (diagonal constraint: 2*val is square)
    forbidden_mask = 0
    for i in range(N):
        if (i + 1) * 2 in squares:
            forbidden_mask |= (1 << i)
            
    active_mask = ((1 << N) - 1) & ~forbidden_mask
    active_vertices = [i for i in range(N) if (active_mask >> i) & 1]
    M = len(active_vertices)
    
    # Map original indices to compact indices 0..M-1 for the solver
    # We will work with the compact graph of active vertices only.
    # Let's build compact conflict masks.
    # compact_conflicts[c] is bitmask of other compact indices conflicting with c.
    
    # First, map original index to compact index
    orig_to_compact = [-1] * N
    for c_idx, orig_idx in enumerate(active_vertices):
        orig_to_compact[orig_idx] = c_idx
        
    compact_conflicts = [0] * M
    for c_idx in range(M):
        orig_idx = active_vertices[c_idx]
        orig_mask = conflicts[orig_idx]
        compact_mask = 0
        # Iterate through bits in orig_mask
        m = orig_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            orig_j = lsb.bit_length() - 1
            m ^= lsb
            c_j = orig_to_compact[orig_j]
            if c_j != -1 and c_j != c_idx:
                compact_mask |= (1 << c_j)
        compact_conflicts[c_idx] = compact_mask

    # 3. Modular Decomposition Strategy
    # The sum a+b being a square has periodicity properties.
    # Specifically, if we look at residues mod 4 or mod 8, some pairs are "safe".
    # However, a more robust structural approach is to find large independent sets
    # within residue classes or use the structure to guide the search.
    
    # Let's try a different structural angle:
    # The conflict graph is relatively sparse? 
    # Let's check average degree.
    total_deg = sum(bin(mask).count("1") for mask in compact_conflicts)
    avg_deg = total_deg / M if M > 0 else 0
    
    # 4. Bitmask-accelerated Maximum Independent Set (MIS)
    # We use a recursive backtracking with greedy coloring bound.
    # Since M <= 256, we can't use a single 256-bit int in Python easily for bit operations 
    # across the whole set if we need to manipulate subsets dynamically, but Python ints 
    # are arbitrary precision, so we CAN use a 256-bit int to represent the set P.
    
    # State: 
    # P: int representing the set of available vertices (bit i is 1 if vertex i is in P)
    # X: int representing the set of vertices already chosen (bit i is 1 if vertex i is in X)
    
    best_size = 0
    best_X = 0
    
    # Greedy Coloring Bound
    # We need to color the graph induced by P to get an upper bound on the size of MIS in P.
    # Standard Tomita algorithm for Max Clique on Complement Graph.
    # Here, we want MIS in G. This is Max Clique in G_complement.
    # G_complement edge (u,v) exists if NOT (u,v in G).
    # So in G_comp, u and v are connected if they are NOT conflicting.
    
    # Let's precompute compatibility masks (complement of conflicts within active set)
    # comp_mask[c] has bits set for all d in active set such that d != c and (c,d) NOT in conflicts.
    comp_masks = [0] * M
    all_active_mask = (1 << M) - 1
    for c in range(M):
        # Start with all active vertices
        m = all_active_mask
        # Remove self
        m &= ~(1 << c)
        # Remove conflicts
        m &= ~compact_conflicts[c]
        comp_masks[c] = m

    def get_color_bound(P_mask):
        """
        Returns an upper bound on the size of the MIS in the subgraph induced by P_mask.
        Uses greedy coloring on the Complement Graph induced by P_mask.
        """
        if P_mask == 0:
            return 0
        
        # We need to order the vertices in P. 
        # A good heuristic for coloring bound is to order by degree in the subgraph (or just index).
        # For speed, we'll just iterate through bits.
        # To make the bound tighter, we should color the graph such that we use few colors.
        # Greedy coloring: assign smallest available color.
        
        # Extract vertices
        verts = []
        m = P_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
            
        if not verts:
            return 0
            
        # To get a tight bound, the order of coloring matters.
        # Often, ordering by degree (descending) in the induced subgraph helps.
        # Let's compute degrees in the induced subgraph of the COMPLEMENT graph.
        # Degree of v in Comp[P] is the number of neighbors in P.
        # Neighbors in Comp are those in comp_masks[v] that are also in P.
        
        degs = []
        for v in verts:
            # Count bits in (comp_masks[v] & P_mask)
            deg = bin(comp_masks[v] & P_mask).count("1")
            degs.append(deg)
            
        # Sort vertices by degree descending
        # This is O(K log K) where K is size of P.
        # For small K, this is fine. For large K, it might be slow, but P shrinks quickly in B&B.
        order = sorted(range(len(verts)), key=lambda i: degs[i], reverse=True)
        
        colors = {}
        max_color = 0
        for idx in order:
            v = verts[idx]
            # Find neighbors of v in P that are already colored
            # Neighbors in Comp are in comp_masks[v]
            # We need to check which of those are in verts (and already processed)
            # Since we process in order, we can just check colors dict.
            used_colors = set()
            # Optimization: instead of iterating all neighbors, we can iterate all colored vertices?
            # No, only neighbors matter.
            # Iterate neighbors in P
            neigh_mask = comp_masks[v] & P_mask
            nm = neigh_mask
            while nm:
                lsb = nm & -nm
                u = lsb.bit_length() - 1
                nm ^= lsb
                if u in colors:
                    used_colors.add(colors[u])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
                
        return max_color + 1

    def expand(P_mask, X_mask, depth):
        nonlocal best_size, best_X
        
        if time.time() - start_time > DEADLINE:
            return
            
        if P_mask == 0:
            if depth > best_size:
                best_size = depth
                best_X = X_mask
            return
            
        # Bound
        bound = get_color_bound(P_mask)
        if depth + bound <= best_size:
            return
            
        # Heuristic: Pick a vertex to branch on.
        # Standard: Pick vertex with highest degree in P (in the original graph? or complement?)
        # For MIS, picking a vertex with high degree in G (conflict graph) is often good 
        # because it removes many candidates if we include it, but also means it's "harder" to place.
        # Alternatively, pick vertex with high degree in Comp (compatibility).
        # Let's stick to a simple heuristic: Pick the first vertex in P? 
        # Better: Pick vertex v in P that maximizes the number of conflicts within P?
        # Or minimizes? 
        # Tomita's algorithm for Max Clique picks vertex with highest degree in the candidate set.
        # Here, we are solving MIS in G, which is Clique in G_comp.
        # So we should pick vertex with highest degree in G_comp induced by P.
        
        verts = []
        m = P_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
            
        if not verts:
            return
            
        # Calculate degrees in Comp[P]
        best_v = -1
        max_deg = -1
        for v in verts:
            deg = bin(comp_masks[v] & P_mask).count("1")
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
        v = best_v
        v_bit = 1 << v
        
        # Branch 1: Include v
        # New P = P \ {v} \ N_G(v)
        # N_G(v) in compact indices are bits in compact_conflicts[v].
        # So we remove v and its conflicts from P.
        # P_new = P & ~v_bit & ~compact_conflicts[v]
        P_new = P_mask & ~v_bit & ~compact_conflicts[v]
        X_new = X_mask | v_bit
        expand(P_new, X_new, depth + 1)
        
        # Branch 2: Exclude v
        # P_new_excl = P \ {v}
        P_new_excl = P_mask & ~v_bit
        expand(P_new_excl, X_mask, depth)

    # Initial call
    # To help the search, we can order the initial P.
    # The expand function picks the best vertex dynamically, so initial order of bits doesn't matter for correctness,
    # but might affect which vertex is picked if degrees are tied? 
    # The code picks the first one with max degree in iteration order (lowest index).
    # Let's just start with all active vertices.
    initial_P = (1 << M) - 1
    expand(initial_P, 0, 0)
    
    # Convert best_X (compact indices) back to values
    result_values = []
    m = best_X
    while m:
        lsb = m & -m
        c_idx = lsb.bit_length() - 1
        m ^= lsb
        orig_idx = active_vertices[c_idx]
        result_values.append(orig_idx + 1)
        
    # Fallback if something went wrong
    if not result_values:
        # Try a simple greedy
        current = set()
        for i in range(M):
            # Check if i conflicts with any in current
            ok = True
            mask = compact_conflicts[i]
            # Check intersection with current mask
            # Build current mask
            cm = 0
            for x in current:
                cm |= (1 << x)
            if mask & cm:
                ok = False
            if ok:
                current.add(i)
                result_values.append(active_vertices[i] + 1)
                
    if not result_values:
        result_values = [1]

    # Verify validity
    squares_check = set()
    k = 1
    while k * k <= 512:
        squares_check.add(k * k)
        k += 1
        
    valid = True
    res_set = set(result_values)
    for a in result_values:
        for b in result_values:
            if a <= b and (a + b) in squares_check:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid small set or empty? No, must be non-empty.
        # Let's just return [1] if invalid, though it shouldn't happen.
        return [1]
        
    return result_values

if __name__ == "__main__":
    res = construct()
    print(len(res))

