import random
import time

def _is_c4_free_bitmask(adj_bits, n):
    """Check C4-free using bitmasks. adj_bits[u] is an int bitmask of neighbors."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj_bits[u] & adj_bits[v]
            # Check if common has more than 1 bit set
            if common & (common - 1):
                return False
    return True

def _can_add_edge_bitmask(adj_bits, n, u, v):
    """Check if adding edge (u,v) preserves C4-freeness."""
    if u == v:
        return False
    if (adj_bits[u] >> v) & 1:
        return False
    
    # 1. Check u and v common neighbors
    common_uv = adj_bits[u] & adj_bits[v]
    if common_uv & (common_uv - 1):
        return False

    # 2. Check pairs (u, w) for w in adj[v]
    bits_v = adj_bits[v]
    while bits_v:
        lsb = bits_v & -bits_v
        w = lsb.bit_length() - 1
        bits_v -= lsb
        if w == u: continue
        if adj_bits[u] & adj_bits[w]:
            return False

    # 3. Check pairs (v, w) for w in adj[u]
    bits_u = adj_bits[u]
    while bits_u:
        lsb = bits_u & -bits_u
        w = lsb.bit_length() - 1
        bits_u -= lsb
        if w == v: continue
        if adj_bits[v] & adj_bits[w]:
            return False

    return True

def _has_independent_set_bitmask(adj_bits, n, k):
    """Check if graph has independent set of size k."""
    full_mask = (1 << n) - 1
    non_adj_bits = [0] * n
    for u in range(n):
        mask = full_mask & ~(1 << u) & ~adj_bits[u]
        non_adj_bits[u] = mask
    
    def backtrack(candidates, size):
        if size >= k:
            return True
        if not candidates:
            return False
        if size + candidates.bit_count() < k:
            return False
        
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        
        # Include v
        new_cand = candidates & non_adj_bits[v]
        new_cand &= ~lsb
        if backtrack(new_cand, size + 1):
            return True
        
        # Exclude v
        new_cand2 = candidates & ~lsb
        if backtrack(new_cand2, size):
            return True
        
        return False

    return backtrack(full_mask, 0)

def construct():
    start_time = time.time()
    time_limit = 110
    
    for target_n in range(43, 38, -1):
        if time.time() - start_time > time_limit:
            break
            
        for trial in range(100):
            if time.time() - start_time > time_limit:
                break
            
            random.seed(trial * 1000 + target_n)
            adj_bits = [0] * target_n
            
            edges = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
            random.shuffle(edges)
            
            for u, v in edges:
                if _can_add_edge_bitmask(adj_bits, target_n, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
            
            if not _is_c4_free_bitmask(adj_bits, target_n):
                continue
                
            if not _has_independent_set_bitmask(adj_bits, target_n, 11):
                edge_list = []
                for u in range(target_n):
                    mask = adj_bits[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_list.append([u, v])
                        mask -= lsb
                return {'n': target_n, 'edges': edge_list}
        
    for target_n in range(38, 29, -1):
        if time.time() - start_time > time_limit:
            break
        for trial in range(20):
            if time.time() - start_time > time_limit:
                break
            random.seed(trial * 1000 + target_n + 500)
            adj_bits = [0] * target_n
            edges = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
            random.shuffle(edges)
            for u, v in edges:
                if _can_add_edge_bitmask(adj_bits, target_n, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
            if not _is_c4_free_bitmask(adj_bits, target_n):
                continue
            if not _has_independent_set_bitmask(adj_bits, target_n, 11):
                edge_list = []
                for u in range(target_n):
                    mask = adj_bits[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_list.append([u, v])
                        mask -= lsb
                return {'n': target_n, 'edges': edge_list}
                
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
