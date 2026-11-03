<<<<<<< SEARCH
def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses a bitset for fast difference tracking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    used_bits = 0
    rows = []
    k_plus_1 = k + 1
    # Precompute: for a given row prefix, the set of diffs introduced by adding x
    # is {x - a for a in row_prefix}. We check each against the bitset.

    def build_row(pos, current_mark, row_so_far):
        nonlocal used_bits
        if time.time() > deadline:
            return None

        if pos == k_plus_1:
            return list(row_so_far)

        # Find valid candidates
        valid_candidates = []
        # Prune upper bound: we need at least (k_plus_1 - pos - 1) more elements,
        # each at least 1 greater than previous, so max possible x is:
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None

        for x in range(current_mark + 1, max_x + 1):
            # Check all new diffs are positive and unused
            ok = True
            for a in row_so_far:
                d = x - a
                if (used_bits >> d) & 1:
                    ok = False
                    break
            if ok:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        # Randomize the order of candidates to escape local minima
        # But bias towards smaller values to keep scope small
        n_smallest = min(10, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]

        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(4, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest

        rng.shuffle(candidates_to_try)

        for x in candidates_to_try:
            # Mark diffs
            mask = 0
            for a in row_so_far:
                d = x - a
                mask |= (1 << d)
            used_bits |= mask
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            used_bits &= ~mask

        return None

    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all diffs for this row in the global bitset
        for i2 in range(k_plus_1):
            for j2 in range(i2 + 1, k_plus_1):
                used_bits |= (1 << (row[j2] - row[i2]))

    return rows
=======
def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search with MRV heuristic: choose the position with fewest
    valid candidates first, and use forward checking to prune early.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    used_bits = 0
    rows = []
    k_plus_1 = k + 1

    def count_valid_x(x, row_so_far):
        """Count how many unused differences x would introduce relative to row_so_far."""
        count = 0
        for a in row_so_far:
            d = x - a
            if not ((used_bits >> d) & 1):
                count += 1
        return count

    def build_row(row_so_far, assigned_mask):
        """
        row_so_far: list of values already placed (length = pos)
        assigned_mask: bitmask of which positions in row_so_far are 'fixed' (always all)
        We build the row left-to-right but at each step pick the next value
        that minimizes the number of future conflicts (MRV-like).
        """
        if time.time() > deadline:
            return None

        pos = len(row_so_far)
        if pos == k_plus_1:
            return list(row_so_far)

        # Determine valid next values
        prev_val = row_so_far[-1] if row_so_far else 0
        # We need k_plus_1 - pos - 1 more elements after this one,
        # so x <= max_scope - (k_plus_1 - pos - 1)
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= prev_val:
            return None

        valid_candidates = []
        for x in range(prev_val + 1, max_x + 1):
            ok = True
            for a in row_so_far:
                d = x - a
                if (used_bits >> d) & 1:
                    ok = False
                    break
            if ok:
                valid_candidates.append(x)

        if not valid_candidates:
            return None

        # MRV heuristic: prefer candidates that leave more room / fewer conflicts for remaining positions
        # Approximate: prefer smaller x (tighter packing) but also check forward feasibility
        # Score each candidate by how many future positions it blocks
        scored = []
        for x in valid_candidates:
            # Estimate: for each remaining position, how many valid options would be left?
            # Cheap proxy: number of unused diffs in [x+1, max_scope] that don't conflict
            # We just bias toward smaller x with some randomization
            score = x + rng.random() * 2  # slight randomization
            scored.append((score, x))
        
        scored.sort()
        # Take top candidates, shuffle within top few for diversity
        n_top = min(15, len(scored))
        top_candidates = [x for _, x in scored[:n_top]]
        rng.shuffle(top_candidates)

        for x in top_candidates:
            # Forward check: after placing x, can we still complete the row?
            # Quick check: at least one valid value exists for next position
            next_max = max_scope - (k_plus_1 - pos - 2) if pos < k_plus_1 - 1 else max_scope
            if pos < k_plus_1 - 1:
                has_next = False
                for x2 in range(x + 1, next_max + 1):
                    ok2 = True
                    for a in row_so_far + [x]:
                        d = x2 - a
                        if (used_bits >> d) & 1:
                            ok2 = False
                            break
                    if ok2:
                        has_next = True
                        break
                if not has_next:
                    continue

            # Mark diffs
            mask = 0
            for a in row_so_far:
                d = x - a
                mask |= (1 << d)
            used_bits |= mask
            res = build_row(row_so_far + [x], assigned_mask | (1 << pos))
            if res is not None:
                return res
            used_bits &= ~mask

        return None

    for i in range(n_rows):
        row = build_row([0], 1)
        if row is None:
            return None
        rows.append(row)
        # Mark all diffs for this row in the global bitset
        for i2 in range(k_plus_1):
            for j2 in range(i2 + 1, k_plus_1):
                used_bits |= (1 << (row[j2] - row[i2]))

    return rows
>>>>>>> REPLACE
