"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute is_clique and is_ind for all subsets
    # is_clique[mask]: True if mask induces a clique
    # is_ind[mask]: True if mask induces an independent set
    is_clique = [False] * (1 << n)
    is_ind = [False] * (1 << n)
    is_clique[0] = True
    is_ind[0] = True
    
    for mask in range(1, 1 << n):
        # Get least significant bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be connected to all in rest, and rest must be clique
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = True
            
        # Check independent set: v must not be connected to any in rest, and rest must be independent
        if is_ind[rest] and (adj[v] & rest) == 0:
            is_ind[mask] = True
            
    is_valid = [is_clique[m] or is_ind[m] for m in range(1 << n)]
    
    # Memoization for the search: min_parts[rem] = minimum number of parts needed to cover 'rem'
    # We initialize with a large value
    INF = n + 1
    min_parts = [INF] * (1 << n)
    min_parts[0] = 0
    
    # Also store the actual partition for reconstruction
    # solution[rem] = (first_part_mask, remaining_mask) where first_part_mask is a valid subset of rem
    # and remaining_mask = rem ^ first_part_mask
    solution = [None] * (1 << n)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0
    
    # We compute min_parts[rem] for all rem in increasing order of popcount?
    # No, the recurrence is min_parts[rem] = 1 + min(min_parts[rem ^ sub] for sub in valid_subsets(rem))
    # This is not strictly increasing in rem value, but we can use memoized recursion or iterative DP.
    # Since n=14, 2^14 = 16384 states.
    # For each state rem, we iterate all submasks sub of rem.
    # Total complexity is sum_{rem} 2^{popcount(rem)} = 3^n.
    # 3^14 = 4,782,969. This is very fast.
    
    # Iterative DP
    # We need to iterate rem such that we have computed min_parts[rem ^ sub] before min_parts[rem].
    # This is tricky because rem ^ sub can be any subset of rem.
    # Actually, min_parts[rem] depends on min_parts[smaller subsets].
    # So we can iterate rem by popcount? Or just iterate rem from 0 to 2^n - 1?
    # If we iterate rem from 0 to 2^n - 1, rem ^ sub is always a subset of rem, so its value is <= rem.
    # Wait, if rem = 1010 (10), sub = 0010 (2), rem ^ sub = 1000 (8). 8 < 10.
    # If rem = 1001 (9), sub = 0001 (1), rem ^ sub = 1000 (8). 8 < 9.
    # Generally, sub is a non-empty subset of rem, so rem ^ sub is a proper subset of rem.
    # The numeric value of a proper subset is not always smaller (e.g., 1000 (8) is subset of 1001 (9), 8<9. But 0111 (7) is subset of 1111 (15), 7<15. What about 1010 (10) and 0101 (5)? 5 is not a subset of 10.
    # A subset S of R implies S & ~R = 0.
    # Is S < R always? No. 10 (1010) and 9 (1001) are not subsets of each other.
    # But if S is a subset of R, then S = R & S.
    # Consider R = 1011 (11). Subset S = 1001 (9). 9 < 11.
    # Consider R = 1100 (12). Subset S = 0100 (4). 4 < 12.
    # It turns out that for any proper subset S of R, S < R is NOT necessarily true?
    # Let R = 2^k. Only subsets are 0 and 2^k.
    # Let R = 2^k + 2^j.
    # Actually, if S is a subset of R, then S <= R is true?
    # Yes, because S = R & S <= R.
    # So if we iterate rem from 0 to 2^n - 1, all subsets (including rem^sub) are smaller than rem?
    # Wait, rem^sub is a subset of rem. So rem^sub <= rem.
    # If rem^sub == rem, then sub=0, which we exclude (sub must be non-empty).
    # So rem^sub < rem?
    # If S is a proper subset of R, is S < R?
    # Yes. Because R = S | T where T is non-empty and disjoint from S.
    # R = S + T (since disjoint). T > 0. So R > S.
    # So iterating rem from 0 to 2^n - 1 works.
    
    for rem in range(1, 1 << n):
        if time.time() > deadline:
            break
            
        # Find the best submask
        # We can iterate all submasks of rem
        sub = rem
        best_val = INF
        best_sub = None
        
        while sub:
            if is_valid[sub]:
                rest = rem ^ sub
                if min_parts[rest] + 1 < best_val:
                    best_val = min_parts[rest] + 1
                    best_sub = sub
                    # Optional pruning: if best_val == 1, we can't do better
                    if best_val == 1:
                        break
            sub = (sub - 1) & rem
            
        min_parts[rem] = best_val
        solution[rem] = best_sub
    
    # Reconstruct the solution from full_mask
    if min_parts[full_mask] < INF:
        parts = []
        rem = full_mask
        while rem:
            sub = solution[rem]
            if sub is None:
                # Fallback
                break
            # Convert mask to list of vertices
            verts = []
            temp = sub
            while temp:
                b = temp & -temp
                idx = b.bit_length() - 1
                verts.append(idx)
                temp ^= b
            parts.append(verts)
            rem ^= sub
        
        # Verify
        if len(parts) == min_parts[full_mask]:
            all_verts = []
            for part in parts:
                all_verts.extend(part)
            if sorted(all_verts) == list(range(n)):
                return parts
    
    # Fallback greedy solution
    def greedy_solution():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                # Check if trial is clique or independent
                is_c = True
                is_i = True
                for i in range(len(trial)):
                    for j in range(i+1, len(trial)):
                        if adj[trial[i]] & (1 << trial[j]):
                            is_i = False
                        else:
                            is_c = False
                        if not is_c and not is_i:
                            break
                    if not is_c and not is_i:
                        break
                if is_c or is_i:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts

    return greedy_solution()
# EVOLVE-BLOCK-END
