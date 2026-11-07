# EVOLVE-BLOCK-START
import time
import sys
import random

N = 128

def construct():
    # 1. Precompute forbidden hyperedges (sets of 6 integers)
    forbidden_masks = set()
    
    for c in range(1, N + 1):
        for b in range(1, c):
            if b + c > N:
                break
            for a in range(1, b):
                if a + c > N:
                    break
                h = {a, b, c, a + b, a + c, b + c}
                mask = 0
                for v in h:
                    mask |= (1 << v)
                forbidden_masks.add(mask)
                
    forbidden_list = list(forbidden_masks)
    
    # 2. Precompute conflicts for each vertex
    conflicts = [[] for _ in range(N + 1)]
    for mask in forbidden_list:
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            rest = mask ^ lsb
            conflicts[v].append(rest)
            m ^= lsb
            
    # 3. Order vertices for B&B
    vertices = list(range(1, N + 1))
    vertices.sort(key=lambda v: len(conflicts[v]), reverse=True)
    sorted_vertices = vertices
    
    best_size = 0
    best_mask = 0
    
    # Greedy Initialization
    def greedy_init(order):
        S = 0
        for v in order:
            can_add = True
            for rest in conflicts[v]:
                if (S & rest) == rest:
                    can_add = False
                    break
            if can_add:
                S |= (1 << v)
        return S

    S1 = greedy_init(sorted_vertices)
    best_size = S1.bit_count()
    best_mask = S1
    
    S2 = greedy_init(sorted_vertices[::-1])
    if S2.bit_count() > best_size:
        best_size = S2.bit_count()
        best_mask = S2
        
    random.seed(42)
    for _ in range(50):
        rand_order = sorted_vertices.copy()
        random.shuffle(rand_order)
        S = greedy_init(rand_order)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_mask = S
            
    # 4. Improved B&B Search with better pruning
    sys.setrecursionlimit(10000)
    start_time = time.time()
    timeout = 55.0
    
    # Precompute a greedy coloring upper bound for available sets
    # For a given available_mask, compute an upper bound on how many
    # vertices we can add without violating constraints.
    
    def compute_upper_bound(available_mask, selected_mask):
        """Compute a tighter upper bound using greedy independent set estimation."""
        if available_mask == 0:
            return 0
        
        # Quick upper bound: count of available vertices
        avail_count = available_mask.bit_count()
        if avail_count <= 1:
            return avail_count
        
        # Greedy coloring approach: 
        # Build a conflict graph among available vertices where two vertices
        # conflict if adding both would create a forbidden hyperedge with selected vertices.
        # Two available vertices u, v conflict if there exists a hyperedge H
        # such that H \ {u,v} ⊆ selected_mask.
        
        # For efficiency, use a simpler bound: 
        # Count how many available vertices are "compatible" with each other.
        
        # Extract available vertices
        avail_verts = []
        m = available_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            avail_verts.append(v)
            m ^= lsb
        
        if len(avail_verts) <= 2:
            return len(avail_verts)
        
        # Greedy independent set on the "compatibility" graph
        # Two vertices u, v are compatible if we can add both
        # without creating a forbidden set with selected_mask.
        
        # For each pair (u,v), check if they conflict.
        # u and v conflict if ∃ hyperedge H with u,v ∈ H and H\{u,v} ⊆ selected_mask
        
        # Build conflict matrix among available vertices
        n = len(avail_verts)
        # conflict[i][j] = True if avail_verts[i] and avail_verts[j] conflict
        # Use bitsets for efficiency
        conflict_bits = [0] * n
        
        for i in range(n):
            ui = avail_verts[i]
            for j in range(i + 1, n):
                uj = avail_verts[j]
                # Check if ui and uj conflict
                # They conflict if there's a forbidden mask H where
                # ui, uj ∈ H and (H \ {ui, uj}) ⊆ selected_mask
                # This is equivalent to: ∃ rest in conflicts[ui] such that
                # uj ∈ rest and (rest \ {uj}) ⊆ selected_mask
                # Or more directly: check all hyperedges containing both ui and uj
                
                # Efficient check: for each rest in conflicts[ui], if uj is in rest
                # and (rest & ~selected_mask) has only uj as a bit, then conflict.
                # Actually: rest = H \ {ui}. We need uj ∈ rest and rest \ {uj} ⊆ selected_mask.
                # rest \ {uj} = rest & ~(1 << uj)
                # So: (rest & ~(1 << uj)) ⊆ selected_mask means
                # (rest & ~(1 << uj) & ~selected_mask) == 0
                
                found_conflict = False
                for rest in conflicts[ui]:
                    if rest & (1 << uj):
                        # uj is in rest
                        remaining = rest & ~(1 << uj)
                        if (remaining & ~selected_mask) == 0:
                            found_conflict = True
                            break
                if found_conflict:
                    conflict_bits[i] |= (1 << j)
                    conflict_bits[j] |= (1 << i)
        
        # Now compute greedy independent set size
        # Greedy: repeatedly pick vertex with minimum degree in remaining graph
        remaining = (1 << n) - 1
        independent_set_size = 0
        
        while remaining:
            # Find vertex with minimum degree in remaining subgraph
            best_v = -1
            best_deg = float('inf')
            m = remaining
            while m:
                lsb = m & -m
                v_idx = lsb.bit_length() - 1
                deg = (conflict_bits[v_idx] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v_idx
                m ^= lsb
            
            if best_v == -1:
                break
                
            independent_set_size += 1
            # Remove best_v and its neighbors from remaining
            neighbors = conflict_bits[best_v]
            remaining &= ~(1 << best_v)
            remaining &= ~neighbors
            
        return min(avail_count, independent_set_size + 0)  # +0 is placeholder
    
    def search(selected_mask, available_mask, current_size, depth):
        nonlocal best_size, best_mask
        
        # Time check
        if time.time() - start_time > timeout:
            return
            
        # Basic pruning
        avail_count = available_mask.bit_count()
        if current_size + avail_count <= best_size:
            return
            
        # Tighter upper bound using greedy independent set estimation
        # Only compute for smaller available sets to save time
        if avail_count <= 20:
            ub = compute_upper_bound(available_mask, selected_mask)
            if current_size + ub <= best_size:
                return
        else:
            # For larger sets, use a simpler bound
            # Estimate: at most ceil(avail_count / 2) can be added (heuristic)
            # This is weak but fast
            pass
        
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
            
        # Find the next vertex to process - use most constrained first
        v = -1
        max_conflicts = -1
        for candidate in sorted_vertices:
            if available_mask & (1 << candidate):
                # Count how many available vertices this conflicts with
                # (i.e., how many would be removed if we include this vertex)
                removed_count = 0
                for rest in conflicts[candidate]:
                    if (selected_mask & rest) == rest:
                        # This vertex cannot be added
                        removed_count = -1
                        break
                    # Count how many available vertices are in rest
                    removed_count += (rest & available_mask).bit_count()
                if removed_count == -1:
                    # Cannot add this vertex, prefer it (will just exclude)
                    max_conflicts = float('inf')
                    v = candidate
                    break
                if removed_count > max_conflicts:
                    max_conflicts = removed_count
                    v = candidate
        if v == -1:
            return
            
        lsb = 1 << v
        
        # Check if v can be added
        can_add = True
        for rest in conflicts[v]:
            if (selected_mask & rest) == rest:
                can_add = False
                break
        
        # Branch order: try include first if it's promising, else exclude
        if can_add:
            # Try Include first
            new_selected = selected_mask | lsb
            new_available = available_mask ^ lsb
            search(new_selected, new_available, current_size + 1, depth + 1)
            
            # Try Exclude
            search(selected_mask, available_mask ^ lsb, current_size, depth + 1)
        else:
            # Must exclude
            search(selected_mask, available_mask ^ lsb, current_size, depth + 1)
            
    # Run search
    init_available = 0
    for i in range(1, N + 1):
        init_available |= (1 << i)
        
    search(0, init_available, 0, 0)
    
    # Convert best_mask to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << i):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
