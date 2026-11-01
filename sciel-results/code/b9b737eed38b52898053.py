"""Combined: Strassen block structure + local-search merge refinement."""
import time
import random

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s wall-clock limit
    rng = random.Random(12345)

    n, m, p = DIMENSIONS  # n=2, m=2, p=8
    nm, mp, pn = n * m, m * p, p * n

    # Build the full tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i, else 0.
    T = [[[0] * pn for _ in range(mp)] for _ in range(nm)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j][j * p + k][k * n + i] = 1

    def verify(U, V, W, rank):
        """Check exact decomposition: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]."""
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != T[a][b][c]:
                        return False
        return True

    # Schoolbook decomposition of rank n*m*p (guaranteed valid baseline).
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return U, V, W

    # Strassen decomposition for a single 2x2 * 2x2 block.
    def strassen_block(k0, k1):
        """
        Build the 7-term Strassen decomposition for the 2x2 block at columns k0, k1.
        Returns U_block (4x7), V_block (16x7), W_block (16x7).
        """
        U_b = [[0] * 7 for _ in range(nm)]
        V_b = [[0] * 7 for _ in range(mp)]
        W_b = [[0] * 7 for _ in range(pn)]

        # A indices: a = i*m+j = i*2+j
        # B indices: b = j*p+k = j*8+k
        # C indices: c = k*n+i = k*2+i

        # M1 = (A00+A11)*(B00+B11)
        U_b[0][0] += 1  # A[0,0]
        U_b[3][0] += 1  # A[1,1]
        V_b[0 * p + k0][0] += 1  # B[0,k0]
        V_b[1 * p + k1][0] += 1  # B[1,k1]

        # M2 = (A10+A11)*B00
        U_b[2][1] += 1  # A[1,0]
        U_b[3][1] += 1  # A[1,1]
        V_b[0 * p + k0][1] += 1  # B[0,k0]

        # M3 = A00*(B01-B11)
        U_b[0][2] += 1  # A[0,0]
        V_b[0 * p + k1][2] += 1  # B[0,k1]
        V_b[1 * p + k1][2] -= 1  # B[1,k1]

        # M4 = A11*(B10-B00)
        U_b[3][3] += 1  # A[1,1]
        V_b[1 * p + k0][3] += 1  # B[1,k0]
        V_b[0 * p + k0][3] -= 1  # B[0,k0]

        # M5 = (A00+A01)*B11
        U_b[0][4] += 1  # A[0,0]
        U_b[1][4] += 1  # A[0,1]
        V_b[1 * p + k1][4] += 1  # B[1,k1]

        # M6 = (A10-A00)*(B00+B01)
        U_b[2][5] += 1  # A[1,0]
        U_b[0][5] -= 1  # A[0,0]
        V_b[0 * p + k0][5] += 1  # B[0,k0]
        V_b[0 * p + k1][5] += 1  # B[0,k1]

        # M7 = (A01-A11)*(B10+B11)
        U_b[1][6] += 1  # A[0,1]
        U_b[3][6] -= 1  # A[1,1]
        V_b[1 * p + k0][6] += 1  # B[1,k0]
        V_b[1 * p + k1][6] += 1  # B[1,k1]

        # C formulas:
        # C[k0,0] = M1 + M4 - M5 + M7
        W_b[k0 * n + 0][0] += 1
        W_b[k0 * n + 0][3] += 1
        W_b[k0 * n + 0][4] -= 1
        W_b[k0 * n + 0][6] += 1

        # C[k1,0] = M3 + M5
        W_b[k1 * n + 0][2] += 1
        W_b[k1 * n + 0][4] += 1

        # C[k0,1] = M2 + M4
        W_b[k0 * n + 1][1] += 1
        W_b[k0 * n + 1][3] += 1

        # C[k1,1] = M1 - M2 + M3 + M6
        W_b[k1 * n + 1][0] += 1
        W_b[k1 * n + 1][1] -= 1
        W_b[k1 * n + 1][2] += 1
        W_b[k1 * n + 1][5] += 1

        return U_b, V_b, W_b

    # Build the full Strassen-based decomposition.
    def strassen_decomposition():
        pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        rank = 7 * len(pairs)  # 28
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]

        for block_idx, (k0, k1) in enumerate(pairs):
            U_b, V_b, W_b = strassen_block(k0, k1)
            offset = block_idx * 7
            for r in range(7):
                for a in range(nm):
                    U[a][offset + r] += U_b[a][r]
                for b in range(mp):
                    V[b][offset + r] += V_b[b][r]
                for c in range(pn):
                    W[c][offset + r] += W_b[c][r]

        return U, V, W

    # Start with the best initial decomposition.
    U_s, V_s, W_s = schoolbook()
    U_str, V_str, W_str = strassen_decomposition()

    # Verify Strassen decomposition.
    strassen_valid = verify(U_str, V_str, W_str, 28)

    if strassen_valid:
        U, V, W = U_str, V_str, W_str
        rank = 28
    else:
        U, V, W = U_s, V_s, W_s
        rank = 32

    # Local-search merge: try to merge pairs of rank-1 components.
    # Work with deep copies for safe rollback.
    def deep_copy(U, V, W, rank):
        return ([row[:] for row in U], [row[:] for row in V], [row[:] for row in W], rank)

    def try_merge(U, V, W, rank, col_a, col_b):
        """
        Try to merge two rank-1 components (columns col_a and col_b) into one.
        Returns (new_U, new_V, new_W, new_rank) on success, None on failure.
        """
        # Get the two rank-1 vectors
        u1 = [U[a][col_a] for a in range(nm)]
        v1 = [V[b][col_a] for b in range(mp)]
        w1 = [W[c][col_a] for c in range(pn)]
        u2 = [U[a][col_b] for a in range(nm)]
        v2 = [V[b][col_b] for b in range(mp)]
        w2 = [W[c][col_b] for c in range(pn)]

        # Compute the sum tensor
        sum_tensor = {}
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = u1[a] * v1[b] * w1[c] + u2[a] * v2[b] * w2[c]
                    if abs(val) > 1e-12:
                        sum_tensor[(a, b, c)] = val

        if len(sum_tensor) == 0:
            # Both terms cancel everywhere
            keep_cols = [c for c in range(rank) if c not in (col_a, col_b)]
            new_rank = len(keep_cols)
            new_U = [[U[a][c] for c in keep_cols] for a in range(nm)]
            new_V = [[V[b][c] for c in keep_cols] for b in range(mp)]
            new_W = [[W[c][c2] for c2 in keep_cols] for c in range(pn)]
            return new_U, new_V, new_W, new_rank

        # Try to find a rank-1 decomposition u⊗v⊗w = sum_tensor
        # Fix gauge: u[a0]=1, v[b0]=1 where (a0,b0,c0) is in sum_tensor.
        entries = list(sum_tensor.items())
        a0, b0, c0 = entries[0][0]
        v0 = entries[0][1]

        # Find a1 with (a1,b0,c0) in sum_tensor, a1 != a0
        a1 = None
        for (aa, bb, cc), val in entries:
            if aa != a0 and bb == b0 and cc == c0:
                a1 = aa
                break

        # Find b1 with (a0,b1,c0) in sum_tensor, b1 != b0
        b1 = None
        for (aa, bb, cc), val in entries:
            if aa == a0 and bb != b0 and cc == c0:
                b1 = bb
                break

        # Find c1 with (a0,b0,c1) in sum_tensor, c1 != c0
        c1 = None
        for (aa, bb, cc), val in entries:
            if aa == a0 and bb == b0 and cc != c0:
                c1 = cc
                break

        if a1 is None or b1 is None or c1 is None:
            # Try alternative: use different reference points
            # If we can't find the cross pattern, try another approach
            # Use the fact that for rank-1, T[a,b,c]/T[a0,b0,c] = (u[a]/u[a0])*(v[b]/v[b0])*(w[c]/w[c0])
            # We need to verify consistency.
            # Pick any 4 entries and try to solve.
            if len(entries) < 4:
                return None

            # Try all combinations of 4 entries
            import itertools
            for combo in itertools.combinations(entries, 4):
                (aa0, bb0, cc0), val0 = combo[0]
                (aa1, bb1, cc1), val1 = combo[1]
                (aa2, bb2, cc2), val2 = combo[2]
                (aa3, bb3, cc3), val3 = combo[3]

                # Fix gauge: u[aa0]=1, v[bb0]=1
                # w[cc0] = val0
                # u[aa1]*v[bb1]*w[cc1] = val1
                # u[aa2]*v[bb2]*w[cc2] = val2
                # u[aa3]*v[bb3]*w[cc3] = val3

                # This is still underdetermined. Skip for now.
                pass
            return None

        # Solve with the cross pattern
        u_sol = [0.0] * nm
        v_sol = [0.0] * mp
        w_sol = [0.0] * pn

        u_sol[a0] = 1.0
        v_sol[b0] = 1.0
        w_sol[c0] = v0

        # From (a1,b0,c0): u[a1]*1*w[c0] = val1 -> u[a1] = val1/w[c0]
        val_a1 = sum_tensor.get((a1, b0, c0), 0)
        if abs(w_sol[c0]) < 1e-12:
            return None
        u_sol[a1] = val_a1 / w_sol[c0]

        # From (a0,b1,c0): 1*v[b1]*w[c0] = val2 -> v[b1] = val2/w[c0]
        val_b1 = sum_tensor.get((a0, b1, c0), 0)
        v_sol[b1] = val_b1 / w_sol[c0]

        # From (a0,b0,c1): 1*1*w[c1] = val3 -> w[c1] = val3
        val_c1 = sum_tensor.get((a0, b0, c1), 0)
        w_sol[c1] = val_c1

        # Verify against all entries in sum_tensor
        for (aa, bb, cc), val in sum_tensor.items():
            predicted = u_sol[aa] * v_sol[bb] * w_sol[cc]
            if abs(predicted - val) > 1e-9:
                return None

        # Verify that all zero entries remain zero
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a, b, c) not in sum_tensor:
                        if abs(u_sol[a] * v_sol[b] * w_sol[c]) > 1e-9:
                            return None

        # Build the new matrices
        # Replace col_a with the new vector, remove col_b
        keep_cols = [c for c in range(rank) if c not in (col_a, col_b)]
        # Insert new column at position of col_a
        new_rank = rank - 1
        new_U = [[0.0] * new_rank for _ in range(nm)]
        new_V = [[0.0] * new_rank for _ in range(mp)]
        new_W = [[0.0] * new_rank for _ in range(pn)]

        for idx, c in enumerate(keep_cols):
            if c > col_a:
                # Shift: original col c becomes col c-1 in new matrix
                pass
            # We'll build by iterating over new column indices

        # Simpler: create new matrices by copying all columns except col_b,
        # then set the col_a position to the new values.
        # New column order: all original columns except col_b, with col_a's position
        # now holding the merged values.

        new_col_map = []
        for c in range(rank):
            if c == col_b:
                continue
            new_col_map.append(c)

        # Now col_a might be at a different position. Find its new index.
        new_a_idx = new_col_map.index(col_a) if col_a in new_col_map else -1

        new_U = [[0.0] * new_rank for _ in range(nm)]
        new_V = [[0.0] * new_rank for _ in range(mp)]
        new_W = [[0.0] * new_rank for _ in range(pn)]

        for new_idx, old_idx in enumerate(new_col_map):
            for a in range(nm):
                new_U[a][new_idx] = U[a][old_idx]
            for b in range(mp):
                new_V[b][new_idx] = V[b][old_idx]
            for c in range(pn):
                new_W[c][new_idx] = W[c][old_idx]

        # Set the merged column
        if new_a_idx >= 0:
            for a in range(nm):
                new_U[a][new_a_idx] = u_sol[a]
            for b in range(mp):
                new_V[b][new_a_idx] = v_sol[b]
            for c in range(pn):
                new_W[c][new_a_idx] = w_sol[c]

        return new_U, new_V, new_W, new_rank

    # Run local search
    best_rank = rank
    best_state = deep_copy(U, V, W, rank)

    for iteration in range(100):
        if time.time() > deadline:
            break

        merged = False
        for col_a in range(rank):
            if time.time() > deadline:
                break
            for col_b in range(col_a + 1, rank):
                if time.time() > deadline:
                    break

                result = try_merge(U, V, W, rank, col_a, col_b)
                if result is not None:
                    new_U, new_V, new_W, new_rank = result
                    # Verify the result
                    if verify(new_U, new_V, new_W, new_rank):
                        U, V, W, rank = new_U, new_V, new_W, new_rank
                        if rank < best_rank:
                            best_rank = rank
                            best_state = deep_copy(U, V, W, rank)
                        merged = True
                        break
            if merged:
                break

        if not merged:
            break  # No more merges possible

    # Return the best state found
    U, V, W, rank = best_state
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
