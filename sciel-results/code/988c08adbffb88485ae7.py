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
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest]:
            ok = True
            r = rest
            while r:
                lsb_r = r & -r
                if not (adj[v] & lsb_r):
                    ok = False
                    break
                r ^= lsb_r
            is_clique[mask] = ok
        else:
            is_clique[mask] = False
            
        # Check independent
        if is_ind[rest]:
            ok = True
            r = rest
            while r:
                lsb_r = r & -r
                if (adj[v] & lsb_r):
                    ok = False
                    break
                r ^= lsb_r
            is_ind[mask] = ok
        else:
            is_ind[mask] = False

    # Collect valid masks and organize by contained vertex for fast lookup
    valid_masks = []
    valid_by_vertex = [[] for _ in range(n)]  # valid_by_vertex[v] = list of valid masks containing v
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind[mask]:
            valid_masks.append(mask)
            # Add to each vertex in mask
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                valid_by_vertex[v].append(mask)
                m ^= lsb

    # Sort valid masks by size descending for greedy
    valid_masks_sorted = sorted(valid_masks, key=lambda x: x.bit_count(), reverse=True)
    
    # Precompute max_valid_size for all masks
    max_valid_size = [0] * (1 << n)
    for S in valid_masks:
        s_size = S.bit_count()
        free = full_mask ^ S
        sub = free
        while True:
            M = S | sub
            if s_size > max_valid_size[M]:
                max_valid_size[M] = s_size
            if sub == 0:
                break
            sub = (sub - 1) & free
            
    # Greedy Upper Bound
    def greedy_partition():
        parts = []
        rem = full_mask
        while rem:
            best_S = 0
            best_size = 0
            for S in valid_masks_sorted:
                if (S & rem) == S:
                    sz = S.bit_count()
                    if sz > best_size:
                        best_size = sz
                        best_S = S
                    else:
                        break  # Since sorted by size descending, no need to continue
            if best_S == 0:
                v_bit = rem & -rem
                best_S = v_bit
            parts.append(best_S)
            rem ^= best_S
        return parts

    greedy_parts = greedy_partition()
    best_k = len(greedy_parts)
    best_partition = greedy_parts

    # B&B Search for smaller k
    deadline = time.time() + 75.0  # Safety margin before 90s limit
    
    # Precompute for each vertex, the valid masks containing it, sorted by size descending
    valid_by_vertex_sorted = []
    for v in range(n):
        masks = valid_by_vertex[v]
        valid_by_vertex_sorted.append(sorted(masks, key=lambda x: x.bit_count(), reverse=True))

    def solve(rem, parts_left, path):
        """
        Returns True if rem can be partitioned into parts_left valid sets.
        """
        nonlocal best_k, best_partition
        
        if time.time() > deadline:
            return False

        if rem == 0:
            return parts_left == 0

        if parts_left == 0:
            return False

        if parts_left == 1:
            if is_clique[rem] or is_ind[rem]:
                path.append(rem)
                return True
            return False

        # Pruning: Check if remaining vertices can fit in parts_left sets
        if rem.bit_count() > parts_left * max_valid_size[rem]:
            return False

        # Pick a vertex v in rem with minimum number of valid candidates
        # This reduces branching factor
        v_bit = rem & -rem
        v = v_bit.bit_length() - 1
        
        # Try to find a better pivot: vertex with fewest valid subsets in rem
        min_candidates = float('inf')
        best_v = v
        m = rem
        while m:
            lsb = m & -m
            vi = lsb.bit_length() - 1
            # Count how many valid masks containing vi are subsets of rem
            count = 0
            for S in valid_by_vertex_sorted[vi]:
                if (S & rem) == S:
                    count += 1
                    if count >= min_candidates:
                        break
            if count < min_candidates:
                min_candidates = count
                best_v = vi
                if min_candidates == 1:
                    break
            m ^= lsb
        
        v = best_v
        v_bit = 1 << v
        
        # Get candidates: valid masks containing v that are subsets of rem
        candidates = []
        for S in valid_by_vertex_sorted[v]:
            if (S & rem) == S:
                candidates.append(S)
                # Early termination: if we have enough large candidates, stop
                # Actually, we need all for correctness, but we can break early if size is too small
                # to make progress. But let's keep all for now.
                
        if not candidates:
            return False

        for S in candidates:
            # Additional pruning: after removing S, check if remaining can fit
            rem_after = rem ^ S
            if rem_after.bit_count() > (parts_left - 1) * max_valid_size[rem_after]:
                continue
            path.append(S)
            if solve(rem_after, parts_left - 1, path):
                return True
            path.pop()
            
        return False

    # Try to find a partition with fewer parts than greedy
    for k in range(1, best_k):
        path = []
        if solve(full_mask, k, path):
            best_k = k
            best_partition = path
            break

    # Convert best_partition masks to list of vertex lists
    result = []
    for mask in best_partition:
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
