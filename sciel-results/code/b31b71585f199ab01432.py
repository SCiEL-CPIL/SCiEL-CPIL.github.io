"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time
import sys

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for (u, v) in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    size = 1 << n
    
    # Precompute is_clique and is_indep using bit-parallel checks
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be adjacent to all in rest
        if is_clique[rest]:
            if (rest & ~adj[v]) == 0:
                is_clique[mask] = True
        
        # Check independent: v must not be adjacent to any in rest
        if is_indep[rest]:
            if (rest & adj[v]) == 0:
                is_indep[mask] = True
    
    valid = [is_clique[m] or is_indep[m] for m in range(size)]
    
    full_mask = size - 1
    
    # Fast check: can we do it in 1 part?
    if valid[full_mask]:
        part = [v for v in range(n)]
        return [part]
    
    # Fast check: can we do it in 2 parts?
    # Only need to check valid masks s where s < full_mask^s to avoid duplicates
    # But we must ensure both parts are non-empty
    two_part_found = None
    for s in range(1, size):
        comp = full_mask ^ s
        if comp == 0:
            continue
        if s < comp:  # only check one side to avoid duplicate work
            if valid[s] and valid[comp]:
                two_part_found = s
                break
    
    if two_part_found is not None:
        s = two_part_found
        part1 = [v for v in range(n) if s & (1 << v)]
        part2 = [v for v in range(n) if (full_mask ^ s) & (1 << v)]
        return [part1, part2]
    
    # DP for minimum number of parts (will be >= 3)
    INF = 10**9
    dp = [INF] * size
    dp[0] = 0
    
    # List of valid masks for faster iteration
    valid_masks = [m for m in range(1, size) if valid[m]]
    
    for mask in range(1, size):
        if time.time() > deadline:
            break
        best = INF
        s = mask
        while s:
            if valid[s]:
                val = 1 + dp[mask ^ s]
                if val < best:
                    best = val
                    if best == 3:  # we know it's at least 3, so 3 is optimal
                        break
            s = (s - 1) & mask
        dp[mask] = best
    
    min_parts = dp[full_mask]
    
    # Reconstruct
    parts = []
    remaining = full_mask
    while remaining:
        s = remaining
        found = False
        while s:
            if valid[s] and dp[remaining ^ s] == dp[remaining] - 1:
                part = [v for v in range(n) if s & (1 << v)]
                parts.append(part)
                remaining ^= s
                found = True
                break
            s = (s - 1) & remaining
        if not found:
            break
    
    # Verify and fallback if needed
    all_vertices = set()
    for p in parts:
        for v in p:
            all_vertices.add(v)
    if all_vertices != set(range(n)) or len(parts) == 0:
        # Fallback: greedy assignment
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                can_clique = all((v, u) in edge_set for u in part)
                can_indep = all((v, u) not in edge_set for u in part)
                if can_clique or can_indep:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
    
    # Ensure all parts are nonempty and valid
    final_parts = [p for p in parts if len(p) > 0]
    return final_parts
# EVOLVE-BLOCK-END
