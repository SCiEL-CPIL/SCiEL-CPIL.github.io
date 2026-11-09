import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Bitmask representation
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    non_adj = [0] * n
    all_mask = (1 << n) - 1
    for v in range(n):
        non_adj[v] = all_mask ^ adj[v] ^ (1 << v)

    # Precompute is_clique and is_independent for all masks
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
            
        if is_independent[rest] and (rest & ~non_adj[v]) == 0:
            is_independent[mask] = True
            
    valid_part = [False] * (1 << n)
    for mask in range(1 << n):
        valid_part[mask] = is_clique[mask] or is_independent[mask]

    # Precompute popcount
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)

    # Find max size of valid clique and independent set separately for tighter bounds
    max_clique_size = 0
    max_indep_size = 0
    for mask in range(1, 1 << n):
        if is_clique[mask]:
            if popcount[mask] > max_clique_size:
                max_clique_size = popcount[mask]
        if is_independent[mask]:
            if popcount[mask] > max_indep_size:
                max_indep_size = popcount[mask]
    
    max_part_size = max(max_clique_size, max_indep_size)

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 85s to be safe within 90s limit

    def dfs_solve(unassigned, parts, k_limit):
        if time.time() > deadline:
            return False
        if unassigned == 0:
            return True
        remaining_parts = k_limit - len(parts)
        if remaining_parts < 0:
            return False
        unassigned_count = popcount[unassigned]
        if unassigned_count > remaining_parts * max_part_size:
            return False
        
        # Select vertex: pick the one with fewest valid placements
        # For speed, just pick the lowest set bit (simple but effective)
        lsb = unassigned & -unassigned
        v = lsb.bit_length() - 1
        new_unassigned = unassigned ^ lsb
        
        # Try adding v to existing parts
        # Symmetry breaking: only try distinct part masks
        seen_masks = set()
        for i in range(len(parts)):
            p_mask = parts[i]
            if p_mask in seen_masks:
                continue
            seen_masks.add(p_mask)
            new_p_mask = p_mask | lsb
            if valid_part[new_p_mask]:
                parts[i] = new_p_mask
                if dfs_solve(new_unassigned, parts, k_limit):
                    return True
                parts[i] = p_mask
                
        # Try creating a new part
        if len(parts) < k_limit:
            parts.append(lsb)
            if dfs_solve(new_unassigned, parts, k_limit):
                return True
            parts.pop()
            
        return False

    # Try k from lower bound to n
    k_lower = max(1, (n + max_part_size - 1) // max_part_size)
    
    for k in range(k_lower, n + 1):
        parts = []
        if dfs_solve((1 << n) - 1, parts, k):
            # Convert masks to lists of vertices
            result = []
            for mask in parts:
                vertices = []
                for v in range(n):
                    if mask & (1 << v):
                        vertices.append(v)
                if vertices:
                    result.append(vertices)
            return result
        if time.time() > deadline - 5:
            break
            
    # Fallback: Greedy
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es or (w, u) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
