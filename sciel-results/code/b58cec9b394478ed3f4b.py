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

    # Precompute popcount for speed
    popcount = [0] * (1 << n)
    for i in range(1, 1 << n):
        popcount[i] = popcount[i >> 1] + (i & 1)

    max_part_size = 0
    for mask in range(1, 1 << n):
        if valid_part[mask]:
            if popcount[mask] > max_part_size:
                max_part_size = popcount[mask]

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 75.0  # 75s to be safe within 90s limit

    # Better vertex selection: pick vertex with fewest valid part placements
    def select_vertex(unassigned, parts):
        best_v = -1
        best_score = float('inf')
        temp_parts = list(parts)
        for lsb in [1 << v for v in range(n) if unassigned & (1 << v)]:
            v = lsb.bit_length() - 1
            score = 0
            for i in range(len(temp_parts)):
                p_mask = temp_parts[i]
                new_p_mask = p_mask | lsb
                if valid_part[new_p_mask]:
                    score += 1
            if len(temp_parts) < len(parts) + 1:  # Can create new part
                score += 1
            if score < best_score:
                best_score = score
                best_v = v
        return best_v

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
            
        # Select vertex with fewest options
        v = select_vertex(unassigned, parts)
        if v < 0:
            return False
        lsb = 1 << v
        new_unassigned = unassigned ^ lsb
        
        # Try adding v to existing parts (symmetry breaking: only try first part with same size)
        seen_sizes = set()
        for i in range(len(parts)):
            p_mask = parts[i]
            # Symmetry: skip if we've already tried a part of this size
            if p_mask in seen_sizes:
                continue
            seen_sizes.add(p_mask)
            new_p_mask = p_mask | lsb
            if valid_part[new_p_mask]:
                parts[i] = new_p_mask
                if dfs_solve(new_unassigned, parts, k_limit):
                    return True
                parts[i] = p_mask
                
        # Try creating a new part (only if we haven't created one yet in this level)
        if len(parts) < k_limit:
            parts.append(lsb)
            if dfs_solve(new_unassigned, parts, k_limit):
                return True
            parts.pop()
            
        return False

    # Try k from lower bound to n
    # Lower bound: at least ceil(n / max_part_size)
    k_lower = max(1, (n + max_part_size - 1) // max_part_size)
    
    best_solution = None
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
