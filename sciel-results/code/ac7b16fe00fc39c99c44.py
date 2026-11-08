<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63

    # Guaranteed valid fallback.
    fallback = _schoolbook()

    best = fallback
    best_rank = schoolbook_rank

    # Attempt greedy search for lower ranks.
    # Search from a reasonable lower bound up to just below schoolbook.
    # Lower bound for matmul rank: at least max(n*m, m*p, p*n) is NOT valid,
    # but a weak lower bound is max(n, m, p) * min(n, m, p) ... not tight.
    # We'll search ranks from 20 up to 62, with limited trials each.
    min_rank = 20
    max_rank = schoolbook_rank - 1  # 62

    # Allocate time budget across ranks.
    total_budget = deadline - start_time
    ranks_to_try = list(range(min_rank, max_rank + 1))
    # Prioritize lower ranks.
    per_rank_time = max(1.0, total_budget / max(1, len(ranks_to_try)) * 0.8)

    for rank in ranks_to_try:
        if time.time() > deadline - 5:
            break
        # Scale trials by time budget.
        rank_start = time.time()
        max_trials = int(500 * (per_rank_time / 1.0))
        max_trials = max(50, min(max_trials, 20000))
        result = _greedy_search(rank, max_trials, min(deadline - 5, time.time() + per_rank_time))
        if result is not None:
            # Double-check verification.
            if _verify(result["U"], result["V"], result["W"]):
                best = result
                best_rank = rank
                # Found a valid decomposition; keep searching for even lower.
                # But since we go in increasing order, this is the lowest found so far.
                # Continue to try lower? No, we're going ascending, so this is
                # the first (lowest) rank that worked. We can stop early if we
                # want the minimal, but let's continue a bit to be safe.
                # Actually ascending means we find the smallest rank first.
                # We can break here since any higher rank is worse.
                # But to be safe against a fluke, we accept it and break.
                break

    return best
=======
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63

    # Guaranteed valid fallback.
    fallback = _schoolbook()

    best = fallback
    best_rank = schoolbook_rank

    # Try the structured Strassen-based decomposition (should give rank 21).
    structured = _build_strassen_3x3_3x7()
    if structured is not None:
        best = structured
        best_rank = len(structured["U"][0])

    # Attempt further improvement via search if time remains.
    if time.time() < deadline - 10 and best_rank > 21:
        # Try to find a lower rank decomposition via random search
        for target_rank in range(15, best_rank):
            if time.time() > deadline - 5:
                break
            # Limited trials
            max_trials = 100
            for trial in range(max_trials):
                if time.time() > deadline - 5:
                    break
                seed = trial * 7919 + 13 + target_rank * 31
                rng = random.Random(seed)
                vals = [-2, -1, 0, 1, 2]
                U = [[0] * target_rank for _ in range(n * m)]
                V = [[0] * target_rank for _ in range(m * p)]
                W = [[0] * target_rank for _ in range(p * n)]
                density = 0.3
                for c in range(target_rank):
                    for idx in range(n * m):
                        if rng.random() < density:
                            U[idx][c] = rng.choice(vals)
                    for idx in range(m * p):
                        if rng.random() < density:
                            V[idx][c] = rng.choice(vals)
                    for idx in range(p * n):
                        if rng.random() < density:
                            W[idx][c] = rng.choice(vals)
                if _verify(U, V, W):
                    best = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    break
            if best_rank == target_rank:
                break

    return best
>>>>>>> REPLACE
