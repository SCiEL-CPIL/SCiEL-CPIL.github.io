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
    
    # Helper to get list of bits in mask
    def get_bits(mask):
        bits = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            bits.append(idx)
            m ^= lsb
        return bits

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

    valid_masks = []
    for mask in range(1, 1 << n):
        if is_clique[mask] or is_ind[mask]:
            valid_masks.append(mask)
            
    # Precompute max_valid_size for all masks
    # max_valid_size[mask] = max(popcount(S) for S in valid_masks if S is subset of mask)
    max_valid_size = [0] * (1 << n)
    # Iterate through valid masks and update their supersets
    # To optimize, we can iterate mask from 1 to 2^n-1 and compute max_valid_size[mask]
    # using DP: max_valid_size[mask] = max(max_valid_size[mask ^ lsb], 
    #                                      max_valid_size_of_valid_subset_containing_lsb_in_mask)
    # The second term is hard to compute directly.
    # Let's stick to the superset update method, but only for valid masks.
    # Number of valid masks is small.
    for S in valid_masks:
        s_size = S.bit_count()
        # Iterate all supersets of S
        # Supersets of S are M such that M & S == S.
        # Let free_bits = full_mask ^ S
        # Subsets of free_bits are the variable parts.
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
            # Find largest valid subset of rem
            best_S = 0
            best_size = 0
            # Iterate valid masks, check if subset of rem
            # Optimization: only check masks with size > best_size
            # valid_masks is not sorted by size, so we might check many.
            # For n=14, this is fast enough.
            for S in valid_masks:
                if (S & rem) == S:
                    sz = S.bit_count()
                    if sz > best_size:
                        best_size = sz
                        best_S = S
            if best_S == 0:
                # Should not happen as single vertices are always valid (clique or ind)
                # But if it does, pick any vertex
                v_bit = rem & -rem
                best_S = v_bit
                best_size = 1
            parts.append(best_S)
            rem ^= best_S
        return parts

    greedy_parts = greedy_partition()
    best_k = len(greedy_parts)
    best_partition = greedy_parts

    # B&B Search for smaller k
    deadline = time.time() + 75.0 # Safety margin
    
    def solve(rem, parts_left, path):
        """
        Returns True if rem can be partitioned into parts_left valid sets.
        Modifies path to store the sets if successful.
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
        
        # Lower bound: greedy packing - repeatedly take largest valid subset
        # If even greedy needs more than parts_left, impossible
        greedy_count = 0
        temp_rem = rem
        while temp_rem and greedy_count < parts_left:
            best_sz = 0
            for S in valid_masks:
                if (S & temp_rem) == S:
                    sz = S.bit_count()
                    if sz > best_sz:
                        best_sz = sz
            if best_sz == 0:
                break
            # Remove a vertex from temp_rem (approximate: just count)
            greedy_count += 1
            # Find and remove the best subset
            for S in valid_masks:
                if (S & temp_rem) == S and S.bit_count() == best_sz:
                    temp_rem ^= S
                    break
        if temp_rem != 0 and greedy_count >= parts_left:
            return False

        # Pick the most constrained vertex: one that appears in the fewest valid subsets of rem
        v_counts = {}
        for S in valid_masks:
            if (S & rem) == S:
                # For each vertex in S, increment count
                m = S
                while m:
                    lsb = m & -m
                    v_counts[lsb] = v_counts.get(lsb, 0) + 1
                    m ^= lsb
        
        # Find vertex with minimum count (most constrained)
        min_count = float('inf')
        v_bit = rem & -rem
        m = rem
        while m:
            lsb = m & -m
            cnt = v_counts.get(lsb, 0)
            if cnt < min_count:
                min_count = cnt
                v_bit = lsb
            m ^= lsb
        
        # Candidates: S in valid_masks such that S contains v_bit and S is subset of rem
        candidates = []
        for S in valid_masks:
            if (S & v_bit) and (S & rem == S):
                candidates.append(S)
                
        if not candidates:
            return False

        # Sort candidates: larger first, then prefer those covering more "hard" vertices
        def candidate_score(S):
            # Size is primary, but also consider coverage of constrained vertices
            return (-S.bit_count(), S)
        
        candidates.sort(key=candidate_score)

        for S in candidates:
            path.append(S)
            if solve(rem ^ S, parts_left - 1, path):
                return True
            path.pop()
            
        return False

    # Try to find a partition with fewer parts than greedy
    for k in range(1, best_k):
        if time.time() > deadline:
            break
        path = []
        if solve(full_mask, k, path):
            best_k = k
            best_partition = path
            break # Found minimum, stop

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
        
    # Ensure types are correct (Python int, list)
    # The vertices are already Python ints from bit_length
    # The result is a list of lists of ints
    
    return result

if __name__ == "__main__":
    construct()
