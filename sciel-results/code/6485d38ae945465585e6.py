import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # Bitmask representation
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    all_mask = (1 << n) - 1
    non_adj = [0] * n
    for v in range(n):
        non_adj[v] = all_mask ^ adj[v] ^ (1 << v)  # Exclude self

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

    # Precompute popcount for all masks
    popcount = [0] * (1 << n)
    for mask in range(1, 1 << n):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)

    # Find max size of a valid part for pruning
    max_part_size = 0
    for mask in range(1 << n):
        if valid_part[mask]:
            size = popcount[mask]
            if size > max_part_size:
                max_part_size = size

    # Vertex ordering heuristic: sort vertices by degree (descending)
    # This helps prune earlier in the search
    degrees = [bin(adj[v]).count('1') for v in range(n)]
    vertex_order = sorted(range(n), key=lambda v: -degrees[v])
    
    # Map from original vertex to position in ordering
    pos_in_order = [0] * n
    for i, v in enumerate(vertex_order):
        pos_in_order[v] = i

    # We'll work with the reordered vertices
    # adj_reordered[i] is the bitmask of neighbors in reordered space
    adj_reordered = [0] * n
    non_adj_reordered = [0] * n
    for i, orig_i in enumerate(vertex_order):
        for j, orig_j in enumerate(vertex_order):
            if adj[orig_i] & (1 << orig_j):
                adj_reordered[i] |= (1 << j)
                non_adj_reordered[i] = all_mask ^ adj_reordered[i] ^ (1 << i)

    # Rebuild valid_part for reordered vertices
    # Actually, we can just use the original masks but with reordered vertex indices
    # Let's keep it simple: work in original vertex space but pick vertices in heuristic order
    
    def dfs(unassigned, parts, k_limit):
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
            
        # Pick vertex with highest degree among unassigned (in original ordering)
        # To make this faster, precompute a heuristic
        best_v = -1
        best_score = -1
        temp = unassigned
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            # Score: prefer vertices with high degree in the unassigned set
            # and few options
            score = popcount[adj[v] & unassigned]
            if score > best_score:
                best_score = score
                best_v = v
            temp ^= lsb
        
        v = best_v
        lsb = 1 << v
        new_unassigned = unassigned ^ lsb
        
        # Try adding v to an existing part
        for i in range(len(parts)):
            p_mask = parts[i]
            new_p_mask = p_mask | lsb
            if valid_part[new_p_mask]:
                parts[i] = new_p_mask
                if dfs(new_unassigned, parts, k_limit):
                    return True
                parts[i] = p_mask
                
        # Try creating a new part
        if len(parts) < k_limit:
            parts.append(lsb)
            if dfs(new_unassigned, parts, k_limit):
                return True
            parts.pop()
            
        return False

    # Search for minimum k
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        parts = []
        if dfs(all_mask, parts, k):
            # Convert masks to lists of vertices
            result = []
            for mask in parts:
                vertices = []
                for v in range(n):
                    if mask & (1 << v):
                        vertices.append(v)
                if vertices:
                    result.append(vertices)
            # Ensure types are correct
            return [list(part) for part in result]
            
    # Fallback: Greedy (should not happen)
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i+1:]]
            if all(hits) or not any(hits):
                part.append(v)
                break
        else:
            parts.append([v])
    return [list(part) for part in parts]

# EVOLVE-BLOCK-END
