"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 55.0
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute all valid parts (cliques and independent sets)
    all_cliques = []
    all_independent = []
    
    # Enumerate all subsets (2^14 = 16384, very fast)
    for mask in range(1, 1 << n):
        is_clique = True
        is_indep = True
        verts = []
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                u, v = verts[i], verts[j]
                if (adj[u] >> v) & 1:
                    is_indep = False
                else:
                    is_clique = False
            if not is_clique and not is_indep:
                break
        
        if is_clique:
            all_cliques.append(mask)
        if is_indep:
            all_independent.append(mask)
    
    # Combine all valid parts
    all_parts = list(set(all_cliques + all_independent))
    # Sort by size descending to try larger parts first
    all_parts.sort(key=lambda m: bin(m).count('1'), reverse=True)
    
    # Precompute for each vertex which parts contain it
    parts_by_vertex = [[] for _ in range(n)]
    for pm in all_parts:
        for v in range(n):
            if (pm >> v) & 1:
                parts_by_vertex[v].append(pm)
    
    # Memoization: minimum parts needed to cover a set
    memo_min = {}
    memo_min[0] = 0
    
    # Lower bound: greedy coloring-like estimate
    def lower_bound(remaining):
        if remaining == 0:
            return 0
        # Simple bound: at least ceil(popcount / max_part_size_in_remaining)
        # But better: use a quick greedy packing
        lb = 0
        rem = remaining
        while rem:
            # Find largest valid part subset of rem
            best_size = 0
            for pm in all_parts:
                if (pm & rem) == pm:
                    sz = bin(pm).count('1')
                    if sz > best_size:
                        best_size = sz
                        if best_size == bin(rem).count('1'):
                            break
            if best_size == 0:
                # Should not happen, but safety
                return lb + bin(rem).count('1')
            lb += 1
            rem &= ~(1 << 0)  # placeholder, we need to remove the part
            # Actually let's do a proper greedy
            break
        # Use a simpler bound: chromatic number of the "conflict" structure
        # For now, use popcount / max_part as rough bound
        max_sz = 1
        for pm in all_parts:
            if (pm & remaining) == pm:
                sz = bin(pm).count('1')
                if sz > max_sz:
                    max_sz = sz
        if max_sz == 0:
            return bin(remaining).count('1')
        return (bin(remaining).count('1') + max_sz - 1) // max_sz
    
    # Branch and bound search
    best_found = [None]
    
    def search(remaining, depth, parts_list):
        if time.time() > deadline:
            return False
        if remaining == 0:
            best_found[0] = list(parts_list)
            return True
        
        # Lower bound pruning
        lb = lower_bound(remaining)
        if depth + lb > len(parts_list) + 1:
            return False
        
        # Memoization check
        if remaining in memo_min:
            if memo_min[remaining] >= depth + 1:
                return False
        
        # Pick the vertex with fewest options to reduce branching
        # Find vertex in remaining with fewest valid parts
        min_options = None
        min_v = -1
        rem = remaining
        while rem:
            lsb = rem & (-rem)
            v = lsb.bit_length() - 1
            rem ^= lsb
            # Count parts containing v that are subsets of remaining
            cnt = 0
            for pm in parts_by_vertex[v]:
                if (pm & remaining) == pm:
                    cnt += 1
            if min_options is None or cnt < min_options:
                min_options = cnt
                min_v = v
                if cnt <= 1:
                    break
        
        if min_options == 0:
            memo_min[remaining] = depth + 1
            return False
        
        # Try parts containing min_v, sorted by size descending
        candidates = [pm for pm in parts_by_vertex[min_v] if (pm & remaining) == pm]
        candidates.sort(key=lambda m: bin(m).count('1'), reverse=True)
        
        for part_mask in candidates:
            new_remaining = remaining & ~part_mask
            if new_remaining == remaining:
                continue
            
            # Prune if we know this subproblem needs too many parts
            if new_remaining in memo_min and memo_min[new_remaining] + 1 > depth:
                continue
            
            parts_list.append([v for v in range(n) if (part_mask >> v) & 1])
            if search(new_remaining, depth + 1, parts_list):
                return True
            parts_list.pop()
        
        memo_min[remaining] = depth + 1
        return False
    
    # Get initial upper bound from greedy
    greedy_parts = []
    used = 0
    for v in range(n):
        if (used >> v) & 1:
            continue
        best_part = 1 << v
        best_sz = 1
        for pm in parts_by_vertex[v]:
            if (pm & ~used) == pm:
                sz = bin(pm).count('1')
                if sz > best_sz:
                    best_sz = sz
                    best_part = pm
        greedy_parts.append([x for x in range(n) if (best_part >> x) & 1])
        used |= best_part
    
    upper_bound = len(greedy_parts)
    
    # Search for optimal
    parts_list = []
    if not search((1 << n) - 1, 0, parts_list):
        # Fallback to greedy
        return [list(p) for p in greedy_parts]
    
    result = [[int(v) for v in part] for part in parts_list]
    return result
# EVOLVE-BLOCK-END
