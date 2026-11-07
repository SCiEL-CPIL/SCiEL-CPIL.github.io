<<<<<<< SEARCH
    # Phase 2: swap-based local search on the best found (multiple rounds)
    if time.time() < deadline - 5:
        for _ in range(3):  # up to 3 rounds of local search
            if time.time() > deadline - 5:
                break
            before = len(best)
            best = _swap_search(best, deadline - 5)
            if len(best) <= before:
                break
            best_len = len(best)

    # Phase 3: final repeated-pass extension of best with all candidates
    if time.time() < deadline - 2:
        best = _grow(best, candidates_all, deadline - 1)
        best_len = len(best)

    # Phase 4: one more local search round if time permits
    if time.time() < deadline - 1:
        best = _swap_search(best, deadline - 0.5)
        best_len = len(best)
=======
    # Phase 2: swap-based local search on the best found (multiple rounds)
    if time.time() < deadline - 5:
        for _ in range(5):  # up to 5 rounds of local search
            if time.time() > deadline - 5:
                break
            before = len(best)
            best = _swap_search(best, deadline - 5)
            if len(best) <= before:
                break
            best_len = len(best)

    # Phase 3: final repeated-pass extension of best with all candidates
    if time.time() < deadline - 2:
        best = _grow(best, candidates_all, deadline - 1)
        best_len = len(best)

    # Phase 4: one more local search round if time permits
    if time.time() < deadline - 1:
        best = _swap_search(best, deadline - 0.5)
        best_len = len(best)

    # Phase 5: final polish - try adding single elements with different orderings
    if time.time() < deadline - 0.5:
        # Try ascending order
        cur = list(best)
        cur_set = set(cur)
        dp = _compute_dp(cur)
        for x in range(1, 257):
            if time.time() > deadline - 0.3:
                break
            if x in cur_set:
                continue
            ok, new_dp = _try_add_fast(cur, x, dp)
            if ok:
                cur.append(x)
                cur_set.add(x)
                dp = new_dp
        if len(cur) > best_len:
            best = cur
            best_len = len(cur)
        
        # Try descending order
        cur = list(best)
        cur_set = set(cur)
        dp = _compute_dp(cur)
        for x in range(256, 0, -1):
            if time.time() > deadline - 0.2:
                break
            if x in cur_set:
                continue
            ok, new_dp = _try_add_fast(cur, x, dp)
            if ok:
                cur.append(x)
                cur_set.add(x)
                dp = new_dp
        if len(cur) > best_len:
            best = cur
            best_len = len(cur)
>>>>>>> REPLACE
