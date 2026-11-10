"""Combined program: bitmask state + backtracking with branch-and-bound + local search polish."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 82.0  # safety margin before 90s kill

    n = 256
    recip = [0.0] + [1.0 / i for i in range(1, n + 1)]

    # Precompute for each x, the set of pairs (a, c) such that (a, x, c) is a 3-AP
    # i.e., a + c = 2*x, with a < x < c (or a > x > c, but we'll handle both)
    # Actually, let's precompute conflicts: for each x, list of (a, c) where {a, x, c} forms a 3-AP
    # and a != x != c != a.
    # For checking if adding x to S creates a 3-AP, we need to check if there exist a, c in S
    # such that a + c = 2*x (x is middle), or x + c = 2*a (a is middle, so c = 2*a - x),
    # or a + x = 2*c (c is middle, so c = (a+x)/2).
    
    # More efficient: for each x, precompute the list of pairs (a, c) such that a + c = 2*x, a != x, c != x.
    # Then x can be added iff no such (a, c) has both a and c in S.
    # Also need to check cases where x is an endpoint.
    # Let's think differently.
    # A 3-AP is (a, b, c) with a + c = 2*b.
    # When adding x to S, a new 3-AP is created iff there exist a, c in S (a != c, a != x, c != x) such that:
    #   Case 1: x is the middle: a + c = 2*x
    #   Case 2: x is the left endpoint: x + c = 2*a => a = (x + c)/2, so c must be in S and (x+c) even, and a in S
    #   Case 3: x is the right endpoint: a + x = 2*c => c = (a+x)/2, so a must be in S and (a+x) even, and c in S
    
    # For efficiency with bitmasks, let's precompute for each x:
    #   middle_pairs[x] = list of (a, c) with a + c = 2*x, a < x < c
    #   left_pairs[x] = list of (a, c) where x is left endpoint: a = (x+c)/2, so for each c in S with c > x and (x+c) even, check if (x+c)//2 in S
    #   right_pairs[x] = list of (a, c) where x is right endpoint: c = (a+x)/2, so for each a in S with a < x and (a+x) even, check if (a+x)//2 in S
    
    # Actually, let's use a different approach for the backtracking.
    # We'll use a recursive backtracking that builds the set incrementally.
    # At each step, we decide to include or exclude the next number.
    # We use branch-and-bound: upper bound on remaining reciprocal sum.
    
    # Precompute suffix sums of reciprocals for upper bound
    suffix_recip = [0.0] * (n + 2)
    for i in range(n, 0, -1):
        suffix_recip[i] = suffix_recip[i + 1] + recip[i]

    # Precompute conflict information
    # For each x, list of (a, c) such that {a, x, c} is a 3-AP
    conflicts = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = 2 * b - a
            if c > n or c <= b:
                continue
            # (a, b, c) is a 3-AP
            conflicts[a].append((b, c))
            conflicts[b].append((a, c))
            conflicts[c].append((a, b))

    # For fast checking, let's precompute for each x, the list of pairs (y, z) such that {x, y, z} is a 3-AP
    # and y, z != x.
    # Actually, let's just use a set-based approach for the backtracking.
    
    # Let's use a different strategy: 
    # 1. Use greedy to get a good initial solution
    # 2. Use local search (swaps) to improve
    # 3. Use backtracking with branch-and-bound on small neighborhoods to find better solutions
    
    # Greedy: add numbers in ascending order (smallest first = highest reciprocal)
    # Check if adding x creates a 3-AP with existing elements
    
    def can_add_greedy(x, selected):
        """Check if x can be added to selected without creating a 3-AP."""
        for a in selected:
            # Case 1: x is middle: a + c = 2*x => c = 2*x - a
            c = 2 * x - a
            if c in selected and c != x:
                return False
            # Case 2: x is left endpoint: x + c = 2*a => c = 2*a - x
            c = 2 * a - x
            if c in selected and c != x and c != a:
                return False
            # Case 3: a is middle: x + c = 2*a => c = 2*a - x
            # Same as Case 2
        return True

    # Actually, the above is not quite right. Let me redo.
    # A 3-AP (a, b, c) with a < b < c means a + c = 2*b.
    # When adding x to S, we check if there exist a, c in S such that {a, x, c} forms a 3-AP.
    # This happens if:
    #   (i) x is the middle: a + c = 2*x, i.e., c = 2*x - a
    #   (ii) x is the smallest: x + c = 2*a, i.e., c = 2*a - x (and a is in S, c > a > x)
    #   (iii) x is the largest: a + x = 2*c, i.e., c = (a + x) / 2 (and a < c < x)
    
    def can_add(x, S):
        """Check if x can be added to set S without creating a 3-AP."""
        for a in S:
            # Case (i): x is middle, a is one endpoint
            c = 2 * x - a
            if c in S and c != x and c != a:
                return False
            # Case (ii): x is smallest, a is middle
            # x + c = 2*a => c = 2*a - x
            c = 2 * a - x
            if c in S and c != x and c != a and c > a:
                return False
            # Case (iii): x is largest, a is smallest
            # a + x = 2*c => c = (a + x) / 2
            if (a + x) % 2 == 0:
                c = (a + x) // 2
                if c in S and c != x and c != a and a < c < x:
                    return False
        return True

    # Greedy construction
    S = set()
    for x in range(1, n + 1):
        if can_add(x, S):
            S.add(x)

    def reciprocal_sum(S):
        return sum(1.0 / x for x in S)

    best_set = set(S)
    best_score = reciprocal_sum(best_set)

    # Local search: try swaps
    def is_3ap_free(S):
        """Verify S has no 3-term AP."""
        S_list = sorted(S)
        S_set = set(S_list)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                c = 2 * b - a
                if c in S_set and c > b:
                    return False
        return True

    # Verify
    assert is_3ap_free(best_set), "Initial solution has a 3-AP!"

    # Local search with swaps
    def local_search(S, score, deadline):
        """Perform local search on S, trying to improve score."""
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            S_list = sorted(S)
            not_S = [x for x in range(1, n + 1) if x not in S]
            
            # Try all single swaps: remove r, add a
            # Prioritize adding small a (high reciprocal) and removing large r
            best_gain = 0
            best_a = -1
            best_r = -1
            
            for a in not_S:
                for r in S_list:
                    gain = recip[a] - recip[r]
                    if gain <= 1e-15:
                        break
                    # Try swap
                    new_S = set(S)
                    new_S.discard(r)
                    new_S.add(a)
                    if is_3ap_free(new_S):
                        if gain > best_gain:
                            best_gain = gain
                            best_a = a
                            best_r = r
                        break  # Only need to check the first valid r (largest r) for this a
            
            if best_a != -1:
                S.discard(best_r)
                S.add(best_a)
                score += best_gain
                improved = True
        
        return S, score

    best_set, best_score = local_search(best_set, best_score, DEADLINE - 5.0)

    # Backtracking with branch-and-bound on a neighborhood
    # We'll try to improve by considering re-adding some excluded small numbers
    # and removing some included large numbers, using a more thorough search
    
    if (time.time() - start_time) < DEADLINE - 10.0:
        # Try a more aggressive local search with 2-swaps
        def local_search_2swap(S, score, deadline):
            """Try 2-swaps: remove r1, r2, add a1, a2."""
            improved = True
            while improved and (time.time() - start_time) < deadline:
                improved = False
                S_list = sorted(S)
                not_S = [x for x in range(1, n + 1) if x not in S]
                
                # Limit the search to a reasonable number of combinations
                # Try adding small numbers and removing large ones
                small_not_S = [x for x in not_S if x <= 64]
                large_S = [x for x in S_list if x > 128]
                
                if len(small_not_S) < 2 or len(large_S) < 2:
                    break
                
                # Try all pairs
                for i in range(len(small_not_S)):
                    if (time.time() - start_time) > deadline:
                        break
                    a1 = small_not_S[i]
                    for j in range(i + 1, len(small_not_S)):
                        a2 = small_not_S[j]
                        for k in range(len(large_S)):
                            r1 = large_S[k]
                            for l in range(k + 1, len(large_S)):
                                r2 = large_S[l]
                                gain = recip[a1] + recip[a2] - recip[r1] - recip[r2]
                                if gain <= 1e-15:
                                    continue
                                new_S = set(S)
                                new_S.discard(r1)
                                new_S.discard(r2)
                                new_S.add(a1)
                                new_S.add(a2)
                                if is_3ap_free(new_S):
                                    if gain > 1e-15:
                                        S = new_S
                                        score += gain
                                        improved = True
                                        break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            return S, score

        best_set, best_score = local_search_2swap(best_set, best_score, DEADLINE - 5.0)

    # Final verification
    assert is_3ap_free(best_set), "Final solution has a 3-AP!"
    assert len(best_set) > 0, "Solution is empty!"

    return list(best_set)
# EVOLVE-BLOCK-END
