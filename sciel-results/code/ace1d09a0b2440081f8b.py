# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    def compute_score_fast(U):
        """Compute score using vectorized NumPy operations for speed."""
        U = np.array(sorted(set(U)), dtype=np.int64)
        if len(U) == 0 or U[0] != 0:
            return -1
        if len(U) == 1:
            return 0.0
            
        max_u = U[-1]
        if max_u == 0:
            return 0.0
            
        # Vectorized computation of sumset and diffset sizes
        # This is O(n^2) but very fast in C/NumPy
        sums = np.add.outer(U, U)
        n_sum = len(np.unique(sums))
        
        diffs = np.subtract.outer(U, U)
        n_diff = len(np.unique(diffs))
        
        if n_sum == 0:
            return -1
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
            
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
            
        score = 1 + math.log(ratio) / math.log(base)
        return score

    best_score = 0
    best_set = [0, 1, 3]
    best_score = compute_score_fast([0, 1, 3])
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score_fast(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Exhaustive search for small sets
    # Increased max_u to 30 to capture optimal Golomb rulers up to N=7
    max_u_limit_small = 30
    for max_u in range(1, max_u_limit_small + 1):
        if time.time() > deadline - 10:
            break
        elements = list(range(1, max_u + 1))
        # Limit size to 7 to keep combinatorial explosion manageable within time
        for size in range(3, 8):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)

    # Phase 2: Generate Optimal Golomb Rulers via Backtracking
    def find_optimal_golomb_ruler(n):
        """Find the Golomb ruler of length n with minimum span."""
        if n <= 1:
            return [0]
        
        best_ruler = None
        best_len = float('inf')
        
        def backtrack(current, used_diffs, last):
            nonlocal best_ruler, best_len
            if len(current) == n:
                span = current[-1]
                if span < best_len:
                    best_len = span
                    best_ruler = current[:]
                return
            
            # Heuristic: try to place next mark as close to last as possible,
            # but ensure no difference collisions.
            # We can prune if current span already exceeds best_len
            # (though we don't know final span yet, we can bound it)
            
            start = last + 1
            # Upper bound for next mark: 
            # If we have k marks left, and min gap is 1, 
            # max possible span increase is roughly (n - len(current)) * (best_len / n)
            # Simpler: just iterate and prune if last + (n - len(current)) > best_len
            # Actually, if we place x, the remaining marks must fit in x + (n - len(current) - 1) * max_gap?
            # Let's just iterate up to best_len.
            
            for x in range(start, best_len + 1):
                # Check differences with all previous marks
                valid = True
                new_diffs = []
                for u in current:
                    d = x - u
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Pruning: If x is too large, we might not beat best_len
                # The remaining marks (n - len(current) - 1) must be placed > x.
                # Min span for remaining is x + (n - len(current) - 1).
                if x + (n - len(current) - 1) >= best_len:
                    break
                
                current.append(x)
                used_diffs.update(new_diffs)
                backtrack(current, used_diffs, x)
                current.pop()
                for d in new_diffs:
                    used_diffs.discard(d)

        backtrack([0], {0}, 0)
        return best_ruler

    for n in range(3, 11):
        if time.time() > deadline - 10:
            break
        gr = find_optimal_golomb_ruler(n)
        if gr:
            try_set(gr)

    # Phase 3: Random Sparse Sets
    for _ in range(500):
        if time.time() > deadline - 10:
            break
        n = random.randint(3, 15)
        max_u = random.randint(50, 500)
        points = [0]
        for _ in range(n - 1):
            points.append(random.randint(1, max_u))
        points = sorted(set(points))
        try_set(points)

    # Phase 4: Local Search with Fast Evaluator
    if time.time() < deadline - 5:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            improved = False
            
            # Try adding an element
            if len(current) < 30:
                max_cur = max(current)
                candidates = set()
                for x in current:
                    for dx in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                        candidates.add(x + dx)
                candidates.update(range(max_cur + 1, max_cur + 20))
                candidates.discard(0)
                
                for new_el in sorted(candidates):
                    if new_el in current or new_el < 0:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score_fast(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(max(1, current[i] - 20), current[i] + 21):
                        if new_el in current or new_el < 0:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score_fast(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score_fast(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    return list(best_set)
# EVOLVE-BLOCK-END
