import time

def construct():
    start = time.time()
    deadline = start + 135.0

    def elapsed():
        return time.time() - start

    def is_sidon(S):
        diffs = set()
        n = len(S)
        for i in range(n):
            for j in range(i + 1, n):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def solve_sat(D, time_budget):
        """
        SAT formulation:
        - Variables x[i] for i in 1..D: whether element i is in the set
        - 0 is always in the set
        - Constraints:
          1. For each d in 1..30: at least one pair (a, a+d) both in set
          2. For each d in 1..D: at most one pair (a, a+d) both in set (Sidon)
          3. At most 32 elements total (including 0)
        """
        # Map (a, d) to variable index: x[a] and x[a+d] must both be true
        # We use a simpler approach: direct DPLL on element selection
        
        # Variables: 1..D, var i means element i is selected
        # We always have element 0 selected
        
        # Precompute all possible differences
        # diff_used[d] = list of pairs (a, b) where b-a=d and both could be selected
        
        # For efficiency, track which differences are already covered
        # and which are already violated
        
        # DPLL with watched literals approach
        # Assignments: -1 unassigned, 0 false, 1 true
        assign = [-1] * (D + 1)  # index 0 unused, 1..D
        assign[0] = 1  # element 0 always selected
        
        # Track for each difference d: which pairs are possible, how many are true
        # pair_count[d] = number of pairs (a, a+d) where both are assigned true
        # pair_possible[d] = number of pairs (a, a+d) where both could still be true
        
        # This is complex; let's use a simpler backtracking with strong pruning
        
        # Better approach: greedy + local search
        # Start with a good initial solution, then improve
        
        # Actually, let's use a different structural approach:
        # Model as maximum coverage / constraint satisfaction with iterative deepening
        
        # Use a priority-based search: try to cover all 30 differences
        # with minimal diameter
        
        # State: current set S, used differences, remaining needed differences
        # Search: add elements in order of "value" (how many new needed diffs they cover)
        
        # Let's implement a beam search with width 100
        
        import heapq
        
        # State: (negative_score, diameter, frozenset_of_elements, used_diffs_set, covered_count)
        # We want to minimize diameter, so search in increasing diameter order
        
        # Actually, for fixed D, just check if feasible
        # Use iterative deepening on the number of elements
        
        # Simplest robust approach: randomized restarts with greedy + local search
        
        best_result = None
        
        for restart in range(50):
            if elapsed() > time_budget:
                break
            
            # Greedy construction with randomization
            S = [0]
            used = set()
            need = set(range(1, 31))
            
            for _ in range(50):  # max 50 elements
                if not need:
                    break
                if elapsed() > time_budget:
                    break
                
                # Find best x in 1..D that covers most needed diffs
                best_x = None
                best_c = -1
                best_tie = 0
                
                # Randomize slightly: consider top candidates
                candidates = []
                for x in range(1, D + 1):
                    if x in S:
                        continue
                    ok = True
                    c = 0
                    for s in S:
                        d = abs(x - s)
                        if d in used:
                            ok = False
                            break
                        if d in need:
                            c += 1
                    if ok and c > 0:
                        candidates.append((c, x))
                
                if not candidates:
                    break
                
                # Sort by coverage desc, then random tiebreak
                candidates.sort(key=lambda t: (-t[0], hash((t[1], restart)) % 1000))
                
                # Try top few candidates
                for c, x in candidates[:5]:
                    # Test if adding x leads to a complete solution
                    S_test = S + [x]
                    used_test = set(used)
                    need_test = set(need)
                    for s in S:
                        d = abs(x - s)
                        used_test.add(d)
                    need_test -= used_test
                    
                    if not need_test:
                        # Complete!
                        if best_result is None or max(S_test) < max(best_result):
                            best_result = list(S_test)
                        break
                    
                    # Check if still feasible: can we cover remaining needs?
                    # Lower bound: each remaining element can cover at most len(S_test) new diffs
                    # But we need to be more precise
                    
                    # Simple check: is there any x' that can cover at least one remaining need?
                    feasible = False
                    for x2 in range(1, D + 1):
                        if x2 in S_test:
                            continue
                        ok2 = True
                        covers_need = False
                        for s in S_test:
                            d = abs(x2 - s)
                            if d in used_test:
                                ok2 = False
                                break
                            if d in need_test:
                                covers_need = True
                        if ok2 and covers_need:
                            feasible = True
                            break
                    
                    if feasible:
                        # Commit this choice
                        S = S_test
                        used = used_test
                        need = need_test
                        break
                else:
                    break
            
            if best_result is not None and max(best_result) <= D:
                return best_result
        
        return None

    # Binary search for minimal D
    lo, hi = 15, 40
    best_D = None
    best_set = None
    
    # Check if hi is feasible
    if elapsed() < deadline - 30:
        result = solve_sat(hi, 25.0)
        if result is not None:
            best_D = hi
            best_set = result
            # Binary search down
            while lo < hi and elapsed() < deadline - 20:
                mid = (lo + hi) // 2
                result = solve_sat(mid, 15.0)
                if result is not None:
                    hi = mid
                    best_set = result
                else:
                    lo = mid + 1
            best_D = hi
    
    if best_set is not None:
        return best_set
    
    # Fallback: greedy
    result = [0]
    used = set()
    need = set(range(1, 31))
    while need and elapsed() < deadline - 3.0:
        best_x = None
        best_c = -1
        hi_x = max(result) + 80
        for x in range(max(result) + 1, hi_x):
            ok = True
            c = 0
            for s in result:
                d = x - s
                if d in used:
                    ok = False
                    break
                if d in need:
                    c += 1
            if ok and c > best_c:
                best_c = c
                best_x = x
        if best_x is None:
            for x in range(hi_x, hi_x + 500):
                ok = True
                for s in result:
                    if x - s in used:
                        ok = False
                        break
                if ok:
                    best_x = x
                    break
        if best_x is None:
            break
        result.append(best_x)
        for s in result[:-1]:
            used.add(best_x - s)
        need -= used
    
    if not is_sidon(result):
        result = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121]
    
    return result
