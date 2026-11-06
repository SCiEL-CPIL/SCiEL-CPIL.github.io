import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)

    smooth_set = set(smooth)
    m = len(smooth)

    # Precompute conflict bitmask: conflict[i] is a bitmask of indices j
    # such that smooth[i] and smooth[j] are NOT in an antichain relation
    # (i.e., one divides the other). We use this for fast local checks.
    # Note: smooth[i] != smooth[j] always since they are distinct values.
    conflict = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            a, b = smooth[i], smooth[j]
            if b % a == 0 or a % b == 0:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    # Precompute valid antichain pairs and store in dictionary by sum
    # Optimization: we store the first valid pair found for each sum.
    pair_solutions = {}
    for i in range(m):
        a = smooth[i]
        for j in range(i + 1, m):
            b = smooth[j]
            s = a + b
            if s > 128:
                break
            # Check antichain: a does not divide b and b does not divide a
            # Since a < b, we only need to check b % a != 0
            if b % a != 0:
                if s not in pair_solutions:
                    pair_solutions[s] = (a, b)

    # Precompute valid antichain triples
    triple_solutions = {}
    for i in range(m):
        a = smooth[i]
        for j in range(i + 1, m):
            b = smooth[j]
            if b % a == 0:
                continue
            for k in range(j + 1, m):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    break
                # Check antichain for (a, b, c)
                # a < b < c
                # Check pairs: (a,b) already ok, (a,c), (b,c)
                if c % a != 0 and c % b != 0:
                    if s not in triple_solutions:
                        triple_solutions[s] = (a, b, c)

    # Precompute valid antichain quadruples
    quad_solutions = {}
    for i in range(m):
        a = smooth[i]
        for j in range(i + 1, m):
            b = smooth[j]
            if b % a == 0:
                continue
            for k in range(j + 1, m):
                c = smooth[k]
                if c % a == 0 or c % b == 0:
                    continue
                base_sum = a + b + c
                if base_sum > 128:
                    continue
                for l in range(k + 1, m):
                    d = smooth[l]
                    s = base_sum + d
                    if s > 128:
                        break
                    # Check antichain for new pairs involving d: (a,d), (b,d), (c,d)
                    if d % a != 0 and d % b != 0 and d % c != 0:
                        if s not in quad_solutions:
                            quad_solutions[s] = (a, b, c, d)

    # Precompute valid antichain quintuples (only if needed, but let's precompute for speed)
    # C(50,5) is ~2M, might be slow in pure Python loops if not careful.
    # Let's try to be smart: only compute if we expect to need it.
    # But for safety and simplicity, let's do a targeted search for k=5+ if needed.
    
    answers = []

    # Helper for backtracking search for k >= 5
    # Uses bitmask for conflict checking
    def search_k(n, k, start_idx, current_sum, current_mask):
        if time.time() > deadline:
            return None
        if k == 0:
            return [] if current_sum == n else None
        if k == 1:
            rem = n - current_sum
            if rem <= 0 or rem > 128:
                return None
            # Find if rem is in smooth and not conflicting
            # We need rem to be in smooth_set
            if rem not in smooth_set:
                return None
            # Find index of rem
            # Since smooth is sorted, we can binary search or just check
            # But we need to ensure it's > smooth[start_idx-1] to maintain order?
            # Actually, the order in the list doesn't matter for the final set,
            # but for the search tree structure we assume increasing indices.
            # So we need to find an index idx >= start_idx such that smooth[idx] == rem.
            # Since smooth is sorted, if rem > smooth[start_idx-1], it might be at some idx >= start_idx.
            # But if rem < smooth[start_idx], we can't pick it because we're restricting to indices >= start_idx.
            # Wait, the search is for *distinct* terms. The order in the list is just for iteration.
            # If we pick rem, its index must be >= start_idx to avoid duplicates and maintain the "increasing index" invariant of the search.
            # So we need to check if there exists an index idx >= start_idx with smooth[idx] == rem.
            # Since smooth is strictly increasing, there is at most one such index.
            # We can precompute a dict value -> index for smooth numbers.
            pass # We'll handle this in the main loop below for efficiency

        # Pruning
        # Min possible sum with k items from start_idx
        # If current_sum + sum of k smallest available > n, prune
        # If current_sum + sum of k largest available < n, prune (but largest might be too big)
        
        for i in range(start_idx, m - k + 1):
            a = smooth[i]
            new_sum = current_sum + a
            if new_sum > n:
                break
            
            # Check conflict with current_mask
            # current_mask has bits set for chosen indices.
            # We need to check if any bit in current_mask is set in conflict[i].
            # i.e., (current_mask & conflict[i]) == 0
            if current_mask & conflict[i]:
                continue
            
            new_mask = current_mask | (1 << i)
            
            if k == 1:
                if new_sum == n:
                    return [a]
                # If k==1 and new_sum != n, no solution
                # (since we break on new_sum > n, and smooth is increasing, no other option)
                # Wait, if k==1, we are picking the *last* element.
                # So if new_sum == n, we found it. If new_sum < n, we can't pick more.
                # So if new_sum != n, this branch fails.
                continue
            
            res = search_k(n, k - 1, i + 1, new_sum, new_mask)
            if res is not None:
                return [a] + res
        return None

    # Precompute value -> index map for quick lookup in k=1 case if needed
    val_to_idx = {v: i for i, v in enumerate(smooth)}

    # Refine search_k for k=1 to use val_to_idx
    def search_k_optimized(n, k, start_idx, current_sum, current_mask):
        if time.time() > deadline:
            return None
        if k == 0:
            return [] if current_sum == n else None
        
        if k == 1:
            rem = n - current_sum
            if rem <= 0 or rem > 128:
                return None
            if rem not in val_to_idx:
                return None
            idx = val_to_idx[rem]
            if idx < start_idx:
                return None
            if current_mask & (1 << idx):
                return None
            if current_mask & conflict[idx]:
                return None
            return [rem]

        for i in range(start_idx, m - k + 1):
            a = smooth[i]
            new_sum = current_sum + a
            if new_sum > n:
                break
            
            if current_mask & conflict[i]:
                continue
            
            new_mask = current_mask | (1 << i)
            res = search_k_optimized(n, k - 1, i + 1, new_sum, new_mask)
            if res is not None:
                return [a] + res
        return None

    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback: try to fill remaining with any valid representation
            # or just break and hope we have enough? No, we need 128 rows.
            # If we run out of time, we must return *something*.
            # But the loop is fast for k<=4. k>=5 is the risk.
            pass

        found = None

        # Try k=1
        if n in smooth_set:
            found = [n]
        
        # Try k=2
        if found is None:
            if n in pair_solutions:
                found = list(pair_solutions[n])
        
        # Try k=3
        if found is None:
            if n in triple_solutions:
                found = list(triple_solutions[n])
        
        # Try k=4
        if found is None:
            if n in quad_solutions:
                found = list(quad_solutions[n])
        
        # Try k=5
        if found is None:
            found = search_k_optimized(n, 5, 0, 0, 0)

        # Try k=6 (if 5 failed)
        if found is None:
            found = search_k_optimized(n, 6, 0, 0, 0)

        # Try k=7 (unlikely)
        if found is None:
            found = search_k_optimized(n, 7, 0, 0, 0)

        if found is None:
            # Should not happen, but fallback
            found = []

        answers.append(found)

    return answers
