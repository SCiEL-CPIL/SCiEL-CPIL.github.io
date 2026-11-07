"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        a, b = e
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    adj = [0] * n
    for a, b in edge_set:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
    
    valid = [False] * (1 << n)
    for mask in range(1, 1 << n):
        verts = [i for i in range(n) if (mask >> i) & 1]
        is_cli = True
        is_ind = True
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                if verts[j] in adj[verts[i]]:
                    is_ind = False
                else:
                    is_cli = False
        valid[mask] = is_cli or is_ind
    
    def greedy():
        remaining = (1 << n) - 1
        parts = []
        while remaining:
            best = 0
            best_sz = 0
            for mask in range(1, 1 << n):
                if valid[mask] and (mask & ~remaining) == 0:
                    sz = bin(mask).count('1')
                    if sz > best_sz:
                        best = mask
                        best_sz = sz
            parts.append(best)
            remaining &= ~best
        return parts
    
    greedy_res = greedy()
    best_count = len(greedy_res)
    best_parts = list(greedy_res)
    
    start = time.time()
    deadline = start + 75
    
    def solve(assigned, parts):
        nonlocal best_count, best_parts
        if time.time() > deadline:
            return
        if assigned == (1 << n) - 1:
            if len(parts) < best_count:
                best_count = len(parts)
                best_parts = list(parts)
            return
        
        rem = (1 << n) - 1 - assigned
        rem_cnt = bin(rem).count('1')
        
        # Pruning: need at least ceil(rem_cnt / max_part_size) more parts
        # But max_part_size is at most n, so this is weak. Use len(parts) + 1 as lower bound
        if len(parts) + 1 >= best_count:
            return
        
        # Pick vertex with fewest options
        unassigned = [i for i in range(n) if not (assigned >> i) & 1]
        best_v = unassigned[0]
        best_opts = float('inf')
        for v in unassigned:
            opts = 0
            for p in parts:
                if valid[p | (1 << v)]:
                    opts += 1
            opts += 1  # new part
            if opts < best_opts:
                best_opts = opts
                best_v = v
        
        v = best_v
        vbit = 1 << v
        
        # Try joining existing parts
        for i in range(len(parts)):
            new_p = parts[i] | vbit
            if valid[new_p]:
                parts[i] = new_p
                solve(assigned | vbit, parts)
                parts[i] = new_p
        
        # Try new part
        if len(parts) + 1 < best_count:
            parts.append(vbit)
            solve(assigned | vbit, parts)
            parts.pop()
    
    solve(0, [])
    
    result = []
    for pm in best_parts:
        verts = [i for i in range(n) if (pm >> i) & 1]
        result.append(verts)
    
    all_v = []
    for part in result:
        all_v.extend(part)
    assert sorted(all_v) == list(range(n))
    
    return result

# EVOLVE-BLOCK-END
