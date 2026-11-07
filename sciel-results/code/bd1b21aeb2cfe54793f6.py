"""Cochromatic partition solver combining bitmask DP with iterative deepening backtracking."""
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute clique and independent set status for all subsets using DP
    # A subset is a clique if it can be built by adding one vertex at a time
    # where each new vertex is adjacent to all existing vertices
    # A subset is an independent set if no two vertices are adjacent
    
    # Use bitmask DP: for each mask, check if it's a clique or independent set
    # is_clique[mask]: mask is a non-empty clique
    # is_indep[mask]: mask is a non-empty independent set
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    # Single vertices are both cliques and independent sets
    for v in range(n):
        is_clique[1 << v] = True
        is_indep[1 << v] = True
    
    # For larger masks, use the property:
    # A set S is a clique iff for some v in S, S\{v} is a clique and v is adjacent to all in S\{v}
    # A set S is an independent set iff for some v in S, S\{v} is an independent set and v is not adjacent to any in S\{v}
    
    # Iterate by popcount
    for popcount in range(2, n + 1):
        for mask in range(1 << n):
            if bin(mask).count('1') != popcount:
                continue
            if is_clique[mask] or is_indep[mask]:
                continue
            
            # Try removing each vertex
            temp = mask
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                rest = mask ^ lsb
                
                # Check clique: rest is clique AND v adjacent to all in rest
                if is_clique[rest]:
                    # v must be adjacent to all vertices in rest
                    if (rest & ~adj_mask[v]) == 0:
                        is_clique[mask] = True
                        break
                
                # Check independent: rest is independent AND v not adjacent to any in rest
                if not is_clique[mask] and is_indep[rest]:
                    # v must NOT be adjacent to any vertex in rest
                    if (rest & adj_mask[v]) == 0:
                        is_indep[mask] = True
                        break
                
                temp ^= lsb
    
    # Now is_valid_part(mask) = is_clique[mask] or is_indep[mask]
    # Note: empty set is not a valid part (parts must be nonempty)
    
    def is_valid_part(mask):
        return mask != 0 and (is_clique[mask] or is_indep[mask])
    
    # Greedy initial solution
    def greedy_partition():
        parts = []  # list of bitmasks
        for v in range(n):
            vmask = 1 << v
            placed = False
            for i, part in enumerate(parts):
                trial = part | vmask
                if is_valid_part(trial):
                    parts[i] = trial
                    placed = True
                    break
            if not placed:
                parts.append(vmask)
        return parts
    
    initial_parts = greedy_partition()
    best_num_parts = len(initial_parts)
    best_partition = initial_parts[:]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s margin
    
    # Vertex ordering by degree (descending) for better pruning
    vertex_order = sorted(range(n), key=lambda v: -bin(adj_mask[v]).count('1'))
    
    # Backtracking search for cochromatic number
    def backtrack_search(k_max):
        """Try to find a valid partition with at most k_max parts."""
        # parts[i] is a bitmask for part i, or 0 if not yet used
        parts = [0] * k_max
        assigned_mask = 0
        
        # Memoization: key is (idx, tuple of parts sorted) -> but this is expensive
        # Instead, use a simpler memo: for each idx, track which (part_configs) have been tried
        # Given the small n=14, we can afford a more aggressive search
        
        def search(idx):
            nonlocal assigned_mask
            if time.time() > deadline:
                return None
            
            if idx == n:
                # All vertices assigned
                result = []
                for p in parts:
                    if p:
                        verts = []
                        temp = p
                        while temp:
                            lsb = temp & (-temp)
                            verts.append(lsb.bit_length() - 1)
                            temp ^= lsb
                        result.append(verts)
                return result
            
            v = vertex_order[idx]
            vmask = 1 << v
            
            # Try placing v in each existing part
            for i in range(k_max):
                if parts[i] == 0:
                    # Empty part slot - only try the first one to avoid symmetry
                    parts[i] = vmask
                    res = search(idx + 1)
                    if res is not None:
                        return res
                    parts[i] = 0
                else:
                    trial = parts[i] | vmask
                    if is_valid_part(trial):
                        parts[i] = trial
                        res = search(idx + 1)
                        if res is not None:
                            return res
                        parts[i] = trial ^ vmask
            
            # Try creating a new part (use first empty slot to break symmetry)
            for i in range(k_max):
                if parts[i] == 0:
                    parts[i] = vmask
                    res = search(idx + 1)
                    if res is not None:
                        return res
                    parts[i] = 0
                    break
            
            return None
        
        return search(0)
    
    # Try smaller numbers of parts, starting from best_num_parts - 1
    for k in range(best_num_parts - 1, 0, -1):
        if time.time() > deadline:
            break
        result = backtrack_search(k)
        if result is not None:
            best_num_parts = k
            best_partition = result
            break
    
    # Convert best_partition to list of lists of Python ints
    final_parts = []
    for p in best_partition:
        verts = []
        temp = p
        while temp:
            lsb = temp & (-temp)
            verts.append(lsb.bit_length() - 1)
            temp ^= lsb
        final_parts.append(verts)
    
    # Ensure types are correct: Python int, list
    final_result = []
    for part in final_parts:
        final_result.append([int(v) for v in part])
    
    return final_result

# EVOLVE-BLOCK-END
