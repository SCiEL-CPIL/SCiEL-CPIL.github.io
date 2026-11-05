import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # 1. Precompute GCDs
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # 2. Identify Bad Triples and Conflicts using bitmasks for speed
    # conflict_mask[v] is a bitmask where bit (u*N + w) is set if {v, u, w} is bad
    # But that's too large. Instead, for each v, store a set of frozensets but use a faster check.
    # Better: For each v, precompute which pairs in S would conflict.
    # Use: conflict_pairs[v] = set of frozenset({u,w}) such that {v,u,w} is bad
    
    conflict_pairs = [set() for _ in range(N + 1)]
    
    for a, b, c in combinations(numbers, 3):
        g_ab = gcd_table[a][b]
        g_ac = gcd_table[a][c]
        g_bc = gcd_table[b][c]
        if g_ab == g_ac == g_bc:
            conflict_pairs[a].add(frozenset((b, c)))
            conflict_pairs[b].add(frozenset((a, c)))
            conflict_pairs[c].add(frozenset((a, b)))

    # Precompute degrees
    degrees = [len(conflict_pairs[v]) for v in range(N + 1)]
    
    # 3. Greedy Construction - pick vertices with lowest conflict degree first
    candidates = sorted(numbers, key=lambda x: degrees[x])
    
    S = set()
    active_pairs = set()
    
    def can_add(v, active_pairs_set):
        if not conflict_pairs[v]:
            return True
        if not active_pairs_set:
            return True
        for pair in conflict_pairs[v]:
            if pair in active_pairs_set:
                return False
        return True

    for v in candidates:
        if time.time() > deadline:
            break
        if can_add(v, active_pairs):
            S.add(v)
            for u in S:
                if u != v:
                    active_pairs.add(frozenset((u, v)))

    # 4. Local Search with add, 1-swap, and 2-swap phases
    remaining = set(numbers) - S
    
    def rebuild_active_pairs(S_set):
        pairs = set()
        for a, b in combinations(S_set, 2):
            pairs.add(frozenset((a, b)))
        return pairs

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add a vertex
        for v in sorted(remaining, key=lambda x: degrees[x]):
            if time.time() > deadline:
                break
            if can_add(v, active_pairs):
                S.add(v)
                remaining.remove(v)
                for u in S:
                    if u != v:
                        active_pairs.add(frozenset((u, v)))
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-swap (remove u from S, add v from remaining)
        found_swap = False
        for u in list(S):
            if time.time() > deadline:
                break
            for v in remaining:
                if time.time() > deadline:
                    break
                # Check if v can be added to S \ {u}
                # v is valid if no conflicting pair of v is in active_pairs without u
                valid = True
                for pair in conflict_pairs[v]:
                    if pair in active_pairs:
                        if u not in pair:
                            valid = False
                            break
                if valid:
                    S.remove(u)
                    S.add(v)
                    remaining.remove(v)
                    remaining.add(u)
                    active_pairs = rebuild_active_pairs(S)
                    found_swap = True
                    improved = True
                    break
            if found_swap:
                break
        
        if improved:
            continue
        
        # Phase 3: Try 2-swap (remove u1, u2 from S, add v1, v2 from remaining)
        # Only do this if we have time and haven't improved in a while
        if time.time() < deadline - 5.0:
            found_2swap = False
            S_list = list(S)
            rem_list = list(remaining)
            
            # Limit the search space for 2-swaps
            for i in range(len(S_list)):
                if time.time() > deadline - 2.0:
                    break
                u1 = S_list[i]
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline - 2.0:
                        break
                    u2 = S_list[j]
                    
                    # Build S without u1, u2
                    S_minus = S - {u1, u2}
                    active_minus = set()
                    for a, b in combinations(S_minus, 2):
                        active_minus.add(frozenset((a, b)))
                    
                    for k in range(len(rem_list)):
                        if time.time() > deadline - 2.0:
                            break
                        v1 = rem_list[k]
                        for l in range(k + 1, len(rem_list)):
                            if time.time() > deadline - 2.0:
                                break
                            v2 = rem_list[l]
                            
                            # Check if v1 and v2 can both be added to S_minus
                            # And that {v1, v2} doesn't create a bad triple with any element in S_minus
                            
                            # First check v1 against S_minus
                            ok1 = True
                            for pair in conflict_pairs[v1]:
                                if pair in active_minus:
                                    ok1 = False
                                    break
                                # Also check if pair is {v2, w} for some w in S_minus
                                # This will be checked when we add both
                            
                            if not ok1:
                                continue
                            
                            # Check v2 against S_minus
                            ok2 = True
                            for pair in conflict_pairs[v2]:
                                if pair in active_minus:
                                    ok2 = False
                                    break
                            
                            if not ok2:
                                continue
                            
                            # Check the pair {v1, v2} against each element in S_minus
                            # For each w in S_minus, check if {v1, v2, w} is bad
                            # This means: is frozenset({v1, v2}) in conflict_pairs[w]?
                            # Or equivalently, is frozenset({w, v1}) in conflict_pairs[v2] and frozenset({w, v2}) in conflict_pairs[v1]?
                            # Simpler: check if frozenset({v1, v2}) is in conflict_pairs[w] for any w in S_minus
                            
                            pair_v1v2 = frozenset((v1, v2))
                            ok_both = True
                            for w in S_minus:
                                if pair_v1v2 in conflict_pairs[w]:
                                    ok_both = False
                                    break
                            
                            # Also need to check that v1 and v2 don't form bad triples among themselves
                            # But since they're just two elements, no triple with only them.
                            # The triple would be {v1, v2, w} for w in S_minus, which we checked.
                            
                            if ok_both:
                                # Perform the 2-swap
                                S.discard(u1)
                                S.discard(u2)
                                S.add(v1)
                                S.add(v2)
                                remaining.discard(v1)
                                remaining.discard(v2)
                                remaining.add(u1)
                                remaining.add(u2)
                                active_pairs = rebuild_active_pairs(S)
                                found_2swap = True
                                improved = True
                                break
                        
                        if improved:
                            break
                    
                    if improved:
                        break
                
                if improved:
                    break

    return list(S)
