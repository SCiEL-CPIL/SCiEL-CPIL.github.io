import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all masks
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest]:
            # v must be connected to all in rest
            if (adj[v] & rest) == rest:
                is_clique[mask] = True
            else:
                is_clique[mask] = False
        else:
            is_clique[mask] = False
            
        # Check independent
        if is_ind[rest]:
            # v must NOT be connected to any in rest
            if (adj[v] & rest) == 0:
                is_ind[mask] = True
            else:
                is_ind[mask] = False
        else:
            is_ind[mask] = False

    valid_masks = []
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind[mask]:
            valid_masks.append(mask)
            
    # Sort valid masks by size descending to try larger sets first
    valid_masks.sort(key=lambda x: x.bit_count(), reverse=True)
    
    # Precompute max valid subset size for each mask
    # max_size_in_mask[M] = max(popcount(S) for S in valid_masks if S & M == S)
    max_size_in_mask = [0] * (1 << n)
    for M in range(1, 1 << n):
        max_s = 0
        for S in valid_masks:
            if S & M == S:
                s_size = S.bit_count()
                if s_size > max_s:
                    max_s = s_size
                    if max_s == M.bit_count():
                        break
        max_size_in_mask[M] = max_s

    # Precompute for each vertex v, the list of valid masks containing v
    valid_masks_by_vertex = [[] for _ in range(n)]
    for S in valid_masks:
        m = S
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            valid_masks_by_vertex[v].append(S)
            m ^= lsb

    def solve(rem, parts_left):
        if rem == 0:
            return True
        if parts_left == 0:
            return False
        
        # Lower bound
        if parts_left == 1:
            return is_clique[rem] or is_ind[rem]
            
        # Pick a vertex v in rem
        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        # Candidates: S in valid_masks_by_vertex[v] such that S & rem == S
        candidates = []
        for S in valid_masks_by_vertex[v]:
            if S & rem == S:
                candidates.append(S)
                
        if not candidates:
            return False
            
        # Sort by size descending
        candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        # Additional pruning:
        # If the largest candidate size is S_max, we need at least ceil(popcount(rem)/S_max) parts.
        max_s = candidates[0].bit_count()
        if parts_left < math.ceil(rem.bit_count() / max_s):
            return False
            
        for S in candidates:
            if solve(rem ^ S, parts_left - 1):
                return True
        return False

    # Find minimum k
    best_k = None
    for k in range(1, n + 1):
        if solve(full_mask, k):
            best_k = k
            break
            
    if best_k is None:
        best_k = n # Should not happen
        
    # Reconstruct partition
    def find_partition(rem, parts_left, current_path):
        if rem == 0:
            return current_path
        
        if parts_left == 0:
            return None
            
        if parts_left == 1:
            if is_clique[rem] or is_ind[rem]:
                return current_path + [rem]
            return None

        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        candidates = []
        for S in valid_masks_by_vertex[v]:
            if S & rem == S:
                candidates.append(S)
                
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: x.bit_count(), reverse=True)
        
        max_s = candidates[0].bit_count()
        if parts_left < math.ceil(rem.bit_count() / max_s):
            return None
            
        for S in candidates:
            res = find_partition(rem ^ S, parts_left - 1, current_path + [S])
            if res:
                return res
        return None

    partition_masks = find_partition(full_mask, best_k, [])
    
    # Convert masks to list of vertex lists
    result = []
    for mask in partition_masks:
        verts = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb
        result.append(verts)
        
    return result

if __name__ == "__main__":
    construct()
