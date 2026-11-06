import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    N_DIFFS = 30
    FULL_MASK = (1 << N_DIFFS) - 1

    def time_ok():
        return time.time() < deadline

    # --- Phase 1: Greedy construction to get a valid starting set ---
    def greedy_build():
        a = [0]
        covered = set()
        for _ in range(32):
            missing = [d for d in range(1, N_DIFFS + 1) if d not in covered]
            if not missing:
                break
            best_x = None
            best_score = -1
            max_a = max(a)
            # Try candidate x values
            for x in range(max_a + 1, max_a + 500):
                new_diffs = [x - y for y in a]
                # Check Sidon: all diffs distinct
                all_diffs = covered | set(new_diffs)
                if len(all_diffs) != len(covered) + len(new_diffs):
                    continue
                score = sum(1 for d in new_diffs if d in missing)
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x is not None:
                a.append(best_x)
                for d in [best_x - y for y in a[:-1]]:
                    covered.add(d)
            else:
                # Add large element
                x = 2 * max_a + 100
                while True:
                    new_diffs = [x - y for y in a]
                    all_diffs = covered | set(new_diffs)
                    if len(all_diffs) == len(covered) + len(new_diffs):
                        a.append(x)
                        for d in new_diffs:
                            covered.add(d)
                        break
                    x += 1
        return a

    # --- Phase 2: Local search to minimize max element ---
    def is_valid_set(S):
        if 0 not in S:
            return False
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = S[j] - S[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, N_DIFFS + 1):
            if d not in diffs:
                return False
        return True

    def local_search(S, time_limit):
        S = sorted(S)
        best = S[:]
        best_max = max(S)
        start = time.time()
        
        while time.time() < time_limit:
            improved = False
            # Try replacing each element (except 0) with a smaller value
            for i in range(1, len(S)):
                orig = S[i]
                # Try values from 1 to orig-1
                for x in range(1, orig):
                    if x in S:
                        continue
                    new_S = sorted(S[:i] + [x] + S[i+1:])
                    if is_valid_set(new_S):
                        if max(new_S) < best_max:
                            best_max = max(new_S)
                            best = new_S[:]
                            S = new_S
                            improved = True
                            break
            # Try removing an element (if we have >9 elements)
            if len(S) > 9:
                for i in range(1, len(S)):
                    new_S = S[:i] + S[i+1:]
                    if is_valid_set(new_S):
                        if max(new_S) < best_max:
                            best_max = max(new_S)
                            best = new_S[:]
                            S = new_S
                            improved = True
                            break
            if not improved:
                break
        return best

    # --- Phase 3: CSP backtracking with MRV on differences ---
    def csp_search():
        """
        CSP approach: assign elements to cover differences.
        We use the fact that for a Sidon set, each difference appears at most once.
        Strategy: build set element by element, but choose next element based on
        which uncovered difference is most constrained (MRV).
        """
        best_solution = [None]
        best_max_val = [float('inf')]
        
        def dfs(S, used_diffs_set, covered_mask):
            if not time_ok():
                return False
            
            if covered_mask == FULL_MASK:
                if S[-1] < best_max_val[0]:
                    best_max_val[0] = S[-1]
                    best_solution[0] = S[:]
                return True
            
            # MRV: find the uncovered difference with fewest valid placements
            uncovered = [d for d in range(1, N_DIFFS + 1) if not (covered_mask >> (d-1)) & 1]
            if not uncovered:
                return True
            
            # For each uncovered diff, count valid (x,y) pairs where x-y=d, x not in S, 
            # and adding x doesn't create conflicts
            # We want to pick the diff with fewest options
            
            best_diff = None
            best_options = None
            best_options_count = float('inf')
            
            for d in uncovered:
                options = []
                for y in S:
                    x = y + d
                    if x in S or x <= 0:
                        continue
                    # Check if adding x creates any duplicate difference
                    valid = True
                    for y2 in S:
                        if y2 == y:
                            continue
                        diff2 = x - y2
                        if diff2 <= 0:
                            continue
                        if diff2 in used_diffs_set:
                            valid = False
                            break
                    if valid:
                        # Also check no two new diffs are equal (they can't be since x-y is fixed)
                        options.append(x)
                if len(options) < best_options_count:
                    best_options_count = len(options)
                    best_diff = d
                    best_options = options
                    if best_options_count == 0:
                        return False
            
            if best_options is None or not best_options:
                return False
            
            # Try each option for the most constrained difference
            # Sort by resulting max value (prefer smaller)
            best_options.sort()
            
            for x in best_options:
                new_S = S + [x]
                new_diffs_set = used_diffs_set.copy()
                new_mask = covered_mask
                for y in S:
                    diff = x - y
                    new_diffs_set.add(diff)
                    if 1 <= diff <= N_DIFFS:
                        new_mask |= (1 << (diff - 1))
                
                if dfs(new_S, new_diffs_set, new_mask):
                    return True
            
            return False
        
        dfs([0], set(), 0)
        return best_solution[0]

    # --- Main execution ---
    # Start with greedy
    S = greedy_build()
    
    # Local search
    S = local_search(S, deadline - 10.0)
    
    # Try CSP to potentially find better
    csp_result = csp_search()
    if csp_result is not None and max(csp_result) < max(S):
        S = csp_result
    
    # Final local search
    S = local_search(S, deadline - 2.0)
    
    # Verify
    if not is_valid_set(S):
        # Fallback to greedy
        S = greedy_build()
    
    return S
