<<<<<<< SEARCH
    best_candidate = base
    best_rank = 36

    # ---- Try to reduce rank via random column elimination ----
    # Each term c defines a vector in R^{n*m} x R^{m*p} x R^{p*n} (outer product of three vectors).
    # We want a subset of terms whose sum of outer products equals the target tensor.
    # Heuristic: try removing columns one at a time and solving for the residual using the remaining columns.

    # For efficiency, we work with the tensor entries: T[i,a,k] = sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c]
    # Target: T[i,a,k] = 1 for all i,a,k.

    # Try greedy elimination: for each column, check if it can be expressed as a linear combination
    # of the others in the tensor space. This is expensive, so we limit the search.

    def _try_reduce():
        """Attempt to find a smaller rank decomposition by eliminating columns."""
        nonlocal best_candidate, best_rank
        from fractions import Fraction

        # Represent each column c as a 3D tensor (flattened)
        num_terms = n * m * p  # 48
        def _col_vector(U, V, W, c):
            vec = [0] * num_terms
            for i in range(n):
                for a in range(m):
                    u_val = U[i * m + a][c]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[a * p + k][c]
                        if v_val == 0:
                            continue
                        w_val = W[k * n + i][c]
                        if w_val == 0:
                            continue
                        vec[i * m * p + a * p + k] = u_val * v_val * w_val
            return vec

        U = base["U"]
        V = base["V"]
        W = base["W"]
        rank0 = len(U[0])  # 36

        cols = []
        for c in range(rank0):
            cols.append(_col_vector(U, V, W, c))

        target = [1] * num_terms

        # Check that base is valid
        total = [0] * num_terms
        for cv in cols:
            for idx in range(num_terms):
                total[idx] += cv[idx]
        for idx in range(num_terms):
            if total[idx] != 1:
                return  # base invalid, shouldn't happen

        # Try to find a subset of columns that sums to target
        # Greedy: start with all 44, try removing each one and see if the rest can be scaled/adjusted
        # Actually, we can try: for each column c, remove it and solve for weights on the remaining
        # columns such that sum w_i * col_i = target. This is a linear system.

        # Since the space is 48-dimensional and we have 44 columns, the system is underdetermined.
        # We look for integer/half-integer solutions.

        # Simplified approach: try removing pairs of columns and see if the residual can be handled
        # by scaling the remaining columns (i.e., all remaining columns get the same multiplier,
        # and the removed columns' contribution can be absorbed). This is too restrictive.

        # Better: try removing one column at a time. If removing column c, the residual is
        # target - sum(other cols) = -col_c. We need to express col_c as a linear combination of others.
        # This requires solving a linear system of size 44 x 48 (more variables than equations won't work).
        # Actually 48 equations, 43 unknowns (after fixing one scale). Overdetermined.

        # Given the complexity, we'll try a simpler heuristic: check if any column is zero or
        # if any two columns are identical (then one can be removed by combining).

        # Check for duplicate columns
        seen = {}
        removable = []
        for c in range(rank0):
            key = tuple(cols[c])
            if key in seen:
                removable.append(c)
            else:
                seen[key] = c

        if removable:
            # Try removing one duplicate
            for c_rem in removable:
                new_rank = rank0 - 1
                new_U = [[0] * new_rank for _ in range(n * m)]
                new_V = [[0] * new_rank for _ in range(m * p)]
                new_W = [[0] * new_rank for _ in range(p * n)]
                # Copy all columns except c_rem
                nc = 0
                for c in range(rank0):
                    if c == c_rem:
                        continue
                    for i in range(n * m):
                        new_U[i][nc] = U[i][c]
                    for i in range(m * p):
                        new_V[i][nc] = V[i][c]
                    for i in range(p * n):
                        new_W[i][nc] = W[i][c]
                    nc += 1

                cand = {"U": new_U, "V": new_V, "W": new_W}
                if _validate(cand):
                    best_candidate = cand
                    best_rank = new_rank
                    return

        # No duplicates found. Try a random linear combination approach:
        # Generate random linear combinations of columns and see if we can represent the target
        # with fewer columns. This is very expensive, so we limit the search.

        if not _elapsed_ok():
            return

        # Try random subsets of columns with size < 44
        # For small reductions, try removing 1-3 columns at a time
        max_remove = 3
        for remove_count in range(1, max_remove + 1):
            if not _elapsed_ok():
                return
            # Try random subsets of columns to remove
            for _attempt in range(200):
                if not _elapsed_ok():
                    return
                removed_set = set(random.sample(range(rank0), remove_count))
                kept = [c for c in range(rank0) if c not in removed_set]
                new_rank = len(kept)

                # The residual that the removed columns contributed is:
                # residual = target - sum(kept cols)
                # We need to express residual as a sum of outer products using new columns.
                # This is a hard problem. Skip for now.
                pass

        # If we get here, no improvement found
        return

    if _elapsed_ok():
        _try_reduce()

    if best_rank < 36:
        return best_candidate

    return base
=======
    # Return the static rank-36 decomposition directly to avoid potential bugs in the random search
    return base
>>>>>>> REPLACE
