import time
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s

    total_words = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7, radius=1

    # Helper: int <-> base-q tuple
    def int_to_tuple(x, n, q):
        t = [0] * n
        for i in range(n):
            t[i] = x % q
            x //= q
        return t

    def tuple_to_int(t, n, q):
        x = 0
        for k in range(n - 1, -1, -1):
            x = x * q + t[k]
        return x

    # Precompute balls (list of 15 ints each)
    balls = []
    for i in range(total_words):
        t_i = int_to_tuple(i, n, q)
        ball = [i]
        for pos in range(n):
            orig_val = t_i[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                t_new = t_i[:]
                t_new[pos] = new_val
                x_new = tuple_to_int(t_new, n, q)
                ball.append(x_new)
        balls.append(ball)

    # Precompute ball membership: ball_members[w] = list of codewords whose ball contains w
    # This is 2187 lists, each of size 15.
    ball_members = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in balls[c]:
            ball_members[w].append(c)

    # Precompute complement constraints: for each codeword c, the set of words NOT in balls[c]
    # Each codeword "forbids" these words from being the unique uncovered witness.
    # We'll represent forbidden sets as bitmasks over the 144 tetracodes.
    
    # Tetracodes: all 2-digit base-3 pairs
    tetracodes = []
    for a in range(3):
        for b in range(3):
            tetracodes.append((a, b))
    tc_idx = {tc: i for i, tc in enumerate(tetracodes)}
    tc_mask_all = (1 << 144) - 1

    # For each codeword c, compute the bitmask of tetracodes NOT in balls[c]
    # A tetracode (a,b) is in balls[c] if there exists a word w in balls[c] with w[0]=a, w[1]=b
    forbidden_masks = []
    for c in range(total_words):
        covered_tcs = set()
        for w in balls[c]:
            tw = int_to_tuple(w, n, q)
            covered_tcs.add((tw[0], tw[1]))
        mask = 0
        for i, tc in enumerate(tetracodes):
            if tc not in covered_tcs:
                mask |= (1 << i)
        forbidden_masks.append(mask)

    # Precompute: for each tetracode index, which codewords do NOT cover it (i.e., it's in their forbidden set)
    tc_forbidden_by = [[] for _ in range(144)]
    for c in range(total_words):
        mask = forbidden_masks[c]
        for i in range(144):
            if mask & (1 << i):
                tc_forbidden_by[i].append(c)

    # Phase 0: Greedy set cover initialization
    covered = np.zeros(total_words, dtype=bool)
    initial_codewords = []
    
    # Greedy: pick codeword covering most uncovered words
    uncovered_count = total_words
    while uncovered_count > 0:
        if time.time() > deadline - 80:
            break
        best_idx = -1
        best_cnt = -1
        for c in range(total_words):
            cnt = 0
            for w in balls[c]:
                if not covered[w]:
                    cnt += 1
            if cnt > best_cnt:
                best_cnt = cnt
                best_idx = c
        if best_cnt == 0:
            break
        initial_codewords.append(best_idx)
        for w in balls[best_idx]:
            if not covered[w]:
                covered[w] = True
                uncovered_count -= 1

    # Ensure validity
    final_covered = np.zeros(total_words, dtype=bool)
    for c in initial_codewords:
        for w in balls[c]:
            final_covered[w] = True
    if not np.all(final_covered):
        for u in range(total_words):
            if not final_covered[u]:
                initial_codewords.append(u)
                for w in balls[u]:
                    final_covered[w] = True

    best_solution = list(initial_codewords)
    best_size = len(best_solution)

    # Phase 1: Branch-and-bound exact search to find minimal hitting set
    # Variables: 144 tetracodes. Assign each to a codeword that covers it.
    # Constraint: codeword c can be assigned to tetracode i only if c covers some word with that tetracode
    # i.e., (a,b) is NOT in forbidden_masks[c].
    
    # Precompute allowed codewords per tetracode
    allowed_by_tc = []
    for i in range(144):
        allowed = []
        for c in range(total_words):
            if not (forbidden_masks[c] & (1 << i)):
                allowed.append(c)
        allowed_by_tc.append(allowed)
    
    # Dynamic variable ordering: pick tetracode with fewest remaining allowed codewords
    # State: assignment array (144 entries, -1 = unassigned), selected codewords set
    
    def branch_and_bound(assignments, selected_set, depth):
        nonlocal best_solution, best_size
        
        if time.time() > deadline - 5:
            return False  # timeout
        
        # Check if all assigned
        if depth == 144:
            sol = list(selected_set)
            if len(sol) < best_size:
                best_solution = sol
                best_size = len(sol)
            return True
        
        # Lower bound: at least len(selected_set) codewords needed
        # Fractional LB: for each unassigned tetracode, count how many selected codewords could cover it
        # Actually simpler: each unassigned tetracode needs at least one codeword.
        # The number of unassigned tetracodes is (144 - depth).
        # But multiple tetracodes can be covered by the same codeword.
        # Tighter LB: max over unassigned tetracodes of (1 / max fraction of unassigned tcs one codeword can cover)
        # Simpler: if len(selected_set) + (144 - depth) <= best_size, prune (each unassigned needs at least 1 new)
        # Even simpler: if len(selected_set) >= best_size, prune
        if len(selected_set) >= best_size:
            return False
        
        # Find unassigned tetracode with fewest remaining allowed options
        min_remaining = 145
        min_tc = -1
        for i in range(144):
            if assignments[i] == -1:
                # Count allowed codewords not yet "excluded" by other assignments
                # Actually, all allowed codewords are still available
                cnt = len(allowed_by_tc[i])
                if cnt < min_remaining:
                    min_remaining = cnt
                    min_tc = i
                    if cnt == 1:
                        break
        if min_tc == -1:
            return True  # all assigned
        
        # Try each allowed codeword for this tetracode
        # Order: prefer codewords that cover more unassigned tetracodes
        candidates = allowed_by_tc[min_tc]
        # Sort by how many unassigned tetracodes they can cover
        def score_c(c):
            cnt = 0
            for i in range(144):
                if assignments[i] == -1 and not (forbidden_masks[c] & (1 << i)):
                    cnt += 1
            return -cnt
        candidates.sort(key=score_c)
        
        for c in candidates:
            if time.time() > deadline - 5:
                return False
            if c in selected_set:
                # Already selected, just assign
                assignments[min_tc] = c
                if branch_and_bound(assignments, selected_set, depth + 1):
                    assignments[min_tc] = -1
                    return True
                assignments[min_tc] = -1
            else:
                # Add to selected set
                if len(selected_set) + 1 >= best_size:
                    continue
                selected_set.add(c)
                assignments[min_tc] = c
                if branch_and_bound(assignments, selected_set, depth + 1):
                    assignments[min_tc] = -1
                    selected_set.discard(c)
                    return True
                assignments[min_tc] = -1
                selected_set.discard(c)
        
        return False
    
    # Run B&B with time limit
    assignments = [-1] * 144
    selected_set = set()
    
    # Use iterative deepening: try to find solution of size <= best_size - 1, then -2, etc.
    # Actually, just run B&B and let it find improvements.
    # But B&B on 144 variables is huge. Let's use a heuristic: 
    # Start with current best, try to remove codewords via B&B on the constraint that 
    # every tetracode must be covered by remaining codewords.
    
    # Alternative: Use the B&B to verify if a smaller solution exists.
    # For now, just try a few random restarts with perturbation + greedy removal.
    
    rng = np.random.default_rng(123)
    
    # Phase 2: Iterated local search with smarter perturbation
    current = list(best_solution)
    
    def is_valid_sol(sol):
        cover = np.zeros(total_words, dtype=np.uint8)
        for c in sol:
            for w in balls[c]:
                cover[w] += 1
        return np.all(cover > 0)
    
    def greedy_remove(sol):
        """Remove redundant codewords."""
        sol = list(sol)
        cover = np.zeros(total_words, dtype=np.uint8)
        for c in sol:
            for w in balls[c]:
                cover[w] += 1
        
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(sol) - 1, -1, -1):
                c = sol[i]
                redundant = True
                for w in balls[c]:
                    if cover[w] == 1:
                        redundant = False
                        break
                if redundant:
                    sol.pop(i)
                    for w in balls[c]:
                        cover[w] -= 1
                    changed = True
        return sol
    
    # Try multiple perturbations
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        # Perturb: swap 2-3 codewords
        perturbed = list(current)
        n_swaps = rng.integers(2, 5)
        for _ in range(n_swaps):
            if len(perturbed) == 0:
                break
            idx = rng.integers(len(perturbed))
            c_old = perturbed[idx]
            c_new = rng.integers(total_words)
            while c_new in perturbed:
                c_new = rng.integers(total_words)
            perturbed[idx] = c_new
        
        # Repair if invalid
        if not is_valid_sol(perturbed):
            cover = np.zeros(total_words, dtype=np.uint8)
            for c in perturbed:
                for w in balls[c]:
                    cover[w] += 1
            uncovered = np.where(cover == 0)[0]
            for u in uncovered:
                if u not in perturbed:
                    perturbed.append(u)
        
        # Greedy remove
        perturbed = greedy_remove(perturbed)
        
        if len(perturbed) < best_size:
            best_size = len(perturbed)
            best_solution = list(perturbed)
            current = list(perturbed)
        elif len(perturbed) == best_size and rng.random() < 0.1:
            current = list(perturbed)
    
    # Final verification
    if not is_valid_sol(best_solution):
        cover = np.zeros(total_words, dtype=np.uint8)
        for c in best_solution:
            for w in balls[c]:
                cover[w] += 1
        uncovered = np.where(cover == 0)[0]
        for u in uncovered:
            if u not in best_solution:
                best_solution.append(u)
        best_solution = greedy_remove(best_solution)
    
    # Convert to result format
    result = []
    for c in best_solution:
        result.append(int_to_tuple(c, n, q))
    return result
