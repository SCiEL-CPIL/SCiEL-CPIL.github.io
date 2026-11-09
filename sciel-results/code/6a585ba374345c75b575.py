import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety

    def popcount(x):
        return bin(x).count('1')

    def to_result(n, adj_masks):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    edges.append([u, v])
        return {'n': int(n), 'edges': edges}

    def find_violation_bitmask(adj_masks, n, max_k=None):
        """Find a regular induced subgraph of size >= 6 using bitmask iteration.
        Returns (mask, k) or (None, None)."""
        if max_k is None:
            max_k = n
        # Iterate by size, starting from 6
        for k in range(6, min(max_k, n) + 1):
            # Iterate all k-subsets via bitmask
            # Use the "next combination" trick
            mask = (1 << k) - 1  # first k bits set
            while mask < (1 << n):
                # Check if mask has exactly k bits (it should by construction)
                # Check regularity
                temp = mask
                first_deg = -1
                is_reg = True
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    deg = popcount(adj_masks[v] & mask)
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                    temp ^= lsb
                if is_reg:
                    return mask, k
                # Next combination of k bits in n
                # Smallest zero bit to the right of the lowest set bit
                c = mask & -mask
                r = mask + c
                mask = r | (((r ^ mask) >> 2) // c)
                if mask >= (1 << n):
                    break
        return None, None

    def validate_full(adj_masks, n):
        """Check all k >= 6. Returns True if valid."""
        mask, k = find_violation_bitmask(adj_masks, n)
        return mask is None

    def count_violations_k6(adj_masks, n):
        """Count regular induced subgraphs of exactly size 6."""
        count = 0
        mask = (1 << 6) - 1
        while mask < (1 << n):
            temp = mask
            first_deg = -1
            is_reg = True
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = popcount(adj_masks[v] & mask)
                if first_deg == -1:
                    first_deg = deg
                elif deg != first_deg:
                    is_reg = False
                    break
                temp ^= lsb
            if is_reg:
                count += 1
            # Next combination
            c = mask & -mask
            r = mask + c
            mask = r | (((r ^ mask) >> 2) // c)
            if mask >= (1 << n):
                break
        return count

    def local_search(adj_masks, n, max_steps, time_limit):
        """Local search to eliminate regular induced subgraphs."""
        for step in range(max_steps):
            if time.time() > time_limit:
                return adj_masks, False
            
            mask, k = find_violation_bitmask(adj_masks, n, max_k=12)
            if mask is None:
                return adj_masks, True
            
            # Get vertices in mask
            vertices = []
            temp = mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                vertices.append(v)
                temp ^= lsb
            
            # Try flipping a random edge in the violating set
            if len(vertices) >= 2:
                u, v = random.sample(vertices, 2)
                adj_masks[u] ^= (1 << v)
                adj_masks[v] ^= (1 << u)
        
        return adj_masks, False

    def try_n(n, max_restarts=5, max_steps_per_restart=200):
        """Try to find a valid graph on n vertices."""
        if time.time() > deadline - 10:
            return None
        
        for attempt in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random graph
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Local search
            adj_masks, success = local_search(adj_masks, n, max_steps_per_restart, deadline - 15)
            
            if success:
                # Final validation
                if validate_full(adj_masks, n):
                    return adj_masks
        
        return None

    # Fallback: small valid graph
    n_fallback = 5
    adj_fallback = [0] * n_fallback
    best_result = to_result(n_fallback, adj_fallback)
    
    # Try n values in descending order
    for n_target in [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break
        
        adj = try_n(n_target)
        if adj is not None:
            best_result = to_result(n_target, adj)
            break
    
    return best_result
