"""Solve cochromatic partition via branch-and-bound set cover over valid parts."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # A subset S (bitmask) is a clique iff for every v in S, S \ {v} subset of adj[v]
    # A subset S is an independent set iff for every v in S, S & adj[v] == 0
    
    def is_clique(mask):
        if mask == 0:
            return True
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if (mask & ~adj[v]) & (mask & ~lsb):
                # There's a vertex in mask\{v} not adjacent to v
                return False
            m ^= lsb
        return True
    
    def is_independent(mask):
        if mask == 0:
            return True
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if mask & adj[v]:
                return False
            m ^= lsb
        return True
    
    # Enumerate all valid parts (nonempty subsets that are clique or independent)
    all_masks = []
    for mask in range(1, 1 << n):
        if is_clique(mask) or is_independent(mask):
            all_masks.append(mask)
    
    # Remove dominated: if part A is subset of part B, A is never better (B covers more with same "cost")
    # Keep only maximal valid parts? Not exactly - we might need smaller parts. But for set cover, 
    # using a superset is always at least as good. So we can keep only maximal valid parts.
    # Actually for minimum set cover, using larger sets is always beneficial or neutral.
    # So keep only maximal (by inclusion) valid parts.
    
    # Sort by size descending for better pruning
    all_masks.sort(key=lambda m: m.bit_count(), reverse=True)
    
    # Remove non-maximal: if mask A is subset of mask B, drop A
    maximal = []
    for i, m in enumerate(all_masks):
        dominated = False
        for j, m2 in enumerate(all_masks):
            if i != j and (m & m2) == m and m != m2:
                dominated = True
                break
        if not dominated:
            maximal.append(m)
    
    # Sort maximal by size descending
    maximal.sort(key=lambda m: m.bit_count(), reverse=True)
    
    # Lower bound on number of parts
    # Each part covers at most max_size vertices
    max_size = max(m.bit_count() for m in maximal) if maximal else 1
    lb = (n + max_size - 1) // max_size  # ceil(n / max_size)
    
    full_mask = (1 << n) - 1
    
    best_parts = None
    
    def search(remaining, parts_used, current_parts, depth):
        nonlocal best_parts
        if time.time() > deadline:
            return False
        if remaining == 0:
            if len(current_parts) < len(best_parts) if best_parts else True:
                best_parts = list(current_parts)
            return True
        if len(current_parts) + depth >= (len(best_parts) if best_parts else n + 1):
            return False
        # Lower bound: need at least ceil(popcount(remaining) / max_size) more parts
        rem_count = remaining.bit_count()
        if len(current_parts) + depth + (rem_count + max_size - 1) // max_size >= (len(best_parts) if best_parts else n + 1):
            return False
        
        # Pick a vertex in remaining to branch on
        # Choose the vertex with fewest valid parts covering it (most constrained)
        lsb = remaining & (-remaining)
        v = lsb.bit_length() - 1
        
        # Find valid parts that contain v and are subset of remaining
        candidates = []
        for pm in maximal:
            if (pm & lsb) and (pm & remaining) == pm:
                candidates.append(pm)
        
        # Sort candidates by size descending (greedy preference)
        candidates.sort(key=lambda m: m.bit_count(), reverse=True)
        
        for pm in candidates:
            new_remaining = remaining & ~pm
            current_parts.append(pm)
            if search(new_remaining, parts_used + 1, current_parts, depth + 1):
                current_parts.pop()
                return True
            current_parts.pop()
        
        return False
    
    # Try increasing number of parts from lower bound
    for max_parts in range(lb, n + 1):
        if time.time() > deadline:
            break
        best_parts = None
        # Reset and search with limit
        found = search(full_mask, 0, [], max_parts)
        if found:
            # Convert bitmask parts to vertex lists
            result = []
            for pm in best_parts:
                verts = []
                for v in range(n):
                    if pm & (1 << v):
                        verts.append(v)
                result.append(verts)
            # Ensure all vertices covered exactly once
            all_verts = []
            for part in result:
                all_verts.extend(part)
            if sorted(all_verts) == list(range(n)):
                return [list(p) for p in result]
    
    # Fallback: greedy result (same as original)
    es = {tuple(e) for e in P['edges']}
    es |= {(v, u) for u, v in es}
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            # Check if trial is all-clique or all-independent
            is_c = True
            is_i = True
            for i in range(len(trial)):
                for j in range(i + 1, len(trial)):
                    a, b = trial[i], trial[j]
                    if (a, b) in es:
                        is_i = False
                    else:
                        is_c = False
            if is_c or is_i:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return [list(p) for p in parts]
