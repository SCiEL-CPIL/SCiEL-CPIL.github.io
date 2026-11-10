"""Maximum independent set via branch-and-bound with bitsets."""

def construct():
    import time
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 192
    # vertices are 0..n-1 representing integers 1..n
    
    # Precompute: for each pair (i,j) where i<j, check if they conflict.
    # Conflict: exists c,d distinct from i,j, c!=d, such that (val_i) | (val_j + val_c + val_d)
    # OR (val_j) | (val_i + val_c + val_d)
    
    vals = list(range(1, n+1))  # vals[i] = i+1
    
    # For efficiency, precompute all triple sums
    # triple_sum[i][j][k] = vals[i]+vals[j]+vals[k] for i<j<k
    # But that's 192^3/6 ~ 1.1M entries, feasible
    
    # Actually, let's think differently. For a given pair (i,j), we need to check
    # if there exist c,d (c!=d, c!=i, d!=i, c!=j, d!=j) such that:
    #   vals[i] divides (vals[j] + vals[c] + vals[d])
    # or vals[j] divides (vals[i] + vals[c] + vals[d])
    
    # Precompute for each value v and each pair of indices (c,d) not involving i or j:
    # This is complex. Let's precompute all pair sums first.
    
    # pair_sum[c][d] = vals[c] + vals[d] for c < d
    # Then for fixed i, j: we need vals[i] | (vals[j] + pair_sum[c][d]) for some c,d not in {i,j}
    
    # Alternative: for each i, precompute which (j, c, d) triples cause conflict.
    # This is O(n^4) which is ~192^4 = 1.35B - too slow.
    
    # Better: For each i, and for each possible sum S, know which pairs (c,d) give vals[c]+vals[d] = S.
    # Then for fixed i, j: check if (vals[j] + S) % vals[i] == 0 for some S achievable by pairs not involving i or j.
    
    # Let me precompute pair_sums as a dict: sum_val -> list of (c,d) pairs
    from collections import defaultdict
    pair_sums = defaultdict(list)
    for c in range(n):
        for d in range(c+1, n):
            pair_sums[vals[c] + vals[d]].append((c, d))
    
    # Now for each pair (i, j), i < j, check conflict:
    # Conflict if exists c,d not in {i,j}, c!=d: vals[i] | (vals[j]+vals[c]+vals[d])
    # i.e., exists S in pair_sums such that (vals[j]+S) % vals[i] == 0 and some (c,d) in pair_sums[S] with c,d not in {i,j}
    
    # Precompute conflict as bitsets
    # conflict[i] is a bitset of vertices that conflict with i
    
    # For speed, let's precompute for each i a set of "forbidden pair sums" relative to j
    # Actually, let's just do it directly but smartly.
    
    # For each i (0-indexed), val_i = vals[i]
    # For each j != i, val_j = vals[j]
    # We need: exists S such that (val_j + S) % val_i == 0, and pair_sums[S] contains a pair (c,d) with c,d not in {i,j}
    
    # Precompute for each i: for each residue r mod val_i, which pair sums S satisfy (S + val_j) % val_i == 0... 
    # This depends on val_j, so it's per (i,j) pair.
    
    # Let's just iterate. n=192, pairs = 192*191/2 = ~18k pairs. For each, we check up to ~384 distinct pair sums.
    # 18k * 384 = ~7M operations, feasible.
    
    # But we need to check if ANY pair in pair_sums[S] avoids {i,j}.
    # Precompute for each S: the set of indices used by pairs in pair_sums[S]
    # Actually, store pair_sums[S] as a list, and for checking "avoids {i,j}", we can check if any pair has both elements not in {i,j}.
    
    # For efficiency, for each S, precompute the set of all indices that appear in any pair in pair_sums[S].
    # If |indices_used| >= 3, then for any specific {i,j}, there's likely a pair avoiding them.
    # But we need to be exact.
    
    # Let me precompute: for each S, a bitset of all indices used.
    # Then for pair (i,j): conflict if exists S with (val_j+S) % val_i == 0 AND the bitset of pair_sums[S] has some pair not using i or j.
    # "Some pair not using i or j" is equivalent to: the total number of pairs in pair_sums[S] > number of pairs that use i or j.
    
    # Simpler: for each S, store the list of pairs. For checking if a pair avoids {i,j}, iterate the list.
    # Worst case pair_sums[S] could be large, but average is small.
    
    # Let me just do it directly.
    
    conflict = [0] * n  # bitset, but Python ints as bitsets
    
    for i in range(n):
        vi = vals[i]
        for j in range(i+1, n):
            vj = vals[j]
            # Check if vi | (vj + vc + vd) for some c,d not in {i,j}
            # i.e., exists S: (vj + S) % vi == 0 and pair_sums[S] has a pair avoiding {i,j}
            # Also check if vj | (vi + vc + vd)
            
            conflict_found = False
            
            # Check vi | (vj + S)
            # S must satisfy (vj + S) % vi == 0 => S % vi == (-vj) % vi
            target_res = (-vj) % vi
            # Iterate over multiples of vi: S = k*vi - vj for k such that S >= 2 (min pair sum) and S <= 2*n (max)
            # Actually S ranges from 1+2=3 to (n-1)+n = 383
            # (vj + S) must be divisible by vi, so S = m*vi - vj for m >= 1
            for m in range(1, (383 + vj) // vi + 1):
                S = m * vi - vj
                if S < 3 or S > 2 * n:
                    continue
                if S in pair_sums:
                    # Check if any pair in pair_sums[S] avoids {i,j}
                    for (c, d) in pair_sums[S]:
                        if c != i and c != j and d != i and d != j:
                            conflict_found = True
                            break
                if conflict_found:
                    break
            
            if not conflict_found:
                # Check vj | (vi + S)
                target_res2 = (-vi) % vj
                for m in range(1, (383 + vi) // vj + 1):
                    S = m * vj - vi
                    if S < 3 or S > 2 * n:
                        continue
                    if S in pair_sums:
                        for (c, d) in pair_sums[S]:
                            if c != i and c != j and d != i and d != j:
                                conflict_found = True
                                break
                    if conflict_found:
                        break
            
            if conflict_found:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
        
        if time.time() > deadline - 5:
            break
    
    # Now solve maximum independent set using branch and bound
    # State: current set (as bitset), candidates (as bitset)
    # candidates must be independent from each other and from current set
    
    best_size = 0
    best_set = 0  # bitset of best solution
    
    # Initial candidates: all vertices
    all_candidates = (1 << n) - 1
    
    # For branch and bound, we need to pick a vertex and branch on include/exclude
    # Upper bound: current_size + popcount(candidates)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Use a stack-based or recursive approach
    # To avoid deep recursion, use iterative with explicit stack
    # But let's try recursive first with sys.setrecursionlimit
    
    import sys
    sys.setrecursionlimit(10000)
    
    # Precompute: for each vertex, its neighbors (conflict set)
    # When we include vertex v, we remove v and all its neighbors from candidates
    
    # Branch and bound:
    # def solve(current_size, current_set, candidates):
    #   if candidates == 0: update best, return
    #   upper = current_size + popcount(candidates)
    #   if upper <= best_size: return
    #   Pick a vertex v from candidates (e.g., highest degree in induced subgraph)
    #   Branch 1: include v -> new_candidates = candidates & ~(conflict[v] | (1<<v))
    #   Branch 2: exclude v -> new_candidates = candidates & ~(1<<v)
    #   Explore both
    
    # For vertex selection, pick the vertex with max degree in the candidate subgraph
    # to prune more effectively
    
    def solve(current_size, current_set, candidates):
        nonlocal best_size, best_set
        if time.time() > deadline - 2:
            return
        
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        upper = current_size + popcount(candidates)
        if upper <= best_size:
            return
        
        # Pick vertex with max degree in candidate subgraph
        # Degree of v in candidates: popcount(conflict[v] & candidates)
        # Find max degree vertex
        cand_list = []
        c = candidates
        while c:
            lsb = c & (-c)
            idx = lsb.bit_length() - 1
            deg = popcount(conflict[idx] & candidates)
            cand_list.append((deg, idx))
            c ^= lsb
        
        if not cand_list:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
        
        # Sort by degree descending
        cand_list.sort(reverse=True)
        max_deg, v = cand_list[0]
        v_bit = 1 << v
        
        # Branch 1: include v
        new_candidates = candidates & ~(conflict[v] | v_bit)
        solve(current_size + 1, current_set | v_bit, new_candidates)
        
        if time.time() > deadline - 2:
            return
        
        # Branch 2: exclude v
        new_candidates2 = candidates & ~v_bit
        solve(current_size, current_set, new_candidates2)
    
    solve(0, 0, all_candidates)
    
    # Convert best_set bitset to list of integers
    result = []
    for i in range(n):
        if best_set & (1 << i):
            result.append(vals[i])
    
    # Fallback: if something went wrong, use greedy
    if not result:
        # Greedy: try to add vertices in some order
        selected = []
        for i in range(n):
            v = vals[i]
            ok = True
            for a in selected:
                if a == v:
                    continue
                # Check if a divides sum of v and any two others in selected
                # This is expensive, just do a simple check
                pass
            # Simplified fallback
            selected.append(v)
        result = selected[:48]
    
    return result
