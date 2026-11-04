import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add Strassen 2x2*2x2 block (rank 7) at given offsets."""
        nonlocal col
        if time.time() > deadline:
            return
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1

        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1

        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 1x2*2x2 block (rank 4) at given offsets."""
        nonlocal col
        if time.time() > deadline:
            return
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1

        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1

        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r

        # M1: A0*B00
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2: A1*B10
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3: A0*B01
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4: A1*B11
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1

    # Build baseline rank-66 decomposition
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)

    # Convert to numpy for fast verification
    U_np = np.array(U, dtype=np.int64)
    V_np = np.array(V, dtype=np.int64)
    W_np = np.array(W, dtype=np.int64)

    def verify(U_arr, V_arr, W_arr):
        """Check if decomposition is exact: sum_k U[:,k] outer V[:,k] outer W[:,k] == T"""
        # T[i,j,k] = delta_{i,k} * delta_{j,i+m} for proper indices
        # Actually T[a,b,c] = 1 if a==c and b==a+m (i.e., C[a,b] gets A[a,b-m]*B[m,b])
        # Let's use: reconstruction R = sum_k U[:,k] * V[:,k] * W[:,k] (outer products)
        # R[a,b,c] should equal T[a,b,c]
        # Efficient: R = sum_k np.einsum('i,j,k->ijk', U[:,k], V[:,k], W[:,k])
        # But that's slow for large rank. Use tensor contraction.
        # T[a,b,c] = 1 iff a==c and b-m==a (i.e., b = a+m)
        # Actually for matmul: C = AB, C[i,j] = sum_k A[i,k]*B[k,j]
        # Tensor T[i,k,j] = A[i,k]*B[k,j]*C[i,j] coefficient
        # T[a,b,c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n)
        # T[a,b,c] = 1 iff (a//m == c%n) and (a%m == b//p) and (b%p == c//n)
        # Wait: A is n x m, so A[i,j] with i in [0,n), j in [0,m), index = i*m+j
        # B is m x p, so B[k,l] with k in [0,m), l in [0,p), index = k*p+l
        # C is n x p... wait the problem says C is p x n? Let me re-read.
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So C has p*n entries, indexed as k*n+i where k is column (0..n-1) and i is row (0..p-1)?
        # Actually for 3x4 * 4x6, C is 3x6, so p*n = 6*3 = 18.
        # W[k*n+i, r]: k in [0,n), i in [0,p)? That gives n*p = 18.
        # Hmm, let me think: C[i,j] = sum_k A[i,k]*B[k,j], i in [0,3), j in [0,6)
        # The trace convention W[k*n+i,r] suggests index = k*n + i where k is col index (0..p-1=5) and i is row (0..n-1=2)
        # So C index = col * n + row = j * 3 + i
        # T[a,b,c]: a = i*m + k (A index), b = k*p + j (B index), c = j*n + i (C index)
        # T[a,b,c] = 1 iff a%m == b//p and a//m == c%n and b%p == c//n
        # Let's verify: a//m = i, a%m = k, b//p = k, b%p = j, c//n = j, c%n = i
        # So T[a,b,c] = 1 iff a%m == b//p and a//m == c%n and b%p == c//n

        # Build true tensor
        T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    a = i * m + k
                    b = k * p + j
                    c = j * n + i
                    T[a, b, c] = 1

        # Compute reconstruction
        R = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        rk = U_arr.shape[1]
        for k in range(rk):
            R += np.einsum('i,j,l->ijl', U_arr[:, k], V_arr[:, k], W_arr[:, k])

        return np.array_equal(R, T)

    # Greedy drop/merge search
    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

    if verify(U_np, V_np, W_np):
        best_rank = rank

    # Try dropping individual columns
    cur_U, cur_V, cur_W = U_np.copy(), V_np.copy(), W_np.copy()
    cur_rank = rank

    # Precompute true tensor for repeated use
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                T[a, b, c] = 1

    def fast_verify(U_arr, V_arr, W_arr, T_arr):
        rk = U_arr.shape[1]
        R = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for k in range(rk):
            R += np.einsum('i,j,l->ijl', U_arr[:, k], V_arr[:, k], W_arr[:, k])
        return np.array_equal(R, T_arr)

    # Try dropping each column one at a time
    for trial in range(cur_rank):
        if time.time() > deadline - 5:
            break
        if cur_U[:, trial].sum() == 0:
            continue
        # Try dropping column trial
        new_U = np.delete(cur_U, trial, axis=1)
        new_V = np.delete(cur_V, trial, axis=1)
        new_W = np.delete(cur_W, trial, axis=1)
        if fast_verify(new_U, new_V, new_W, T):
            cur_U, cur_V, cur_W = new_U, new_V, new_W
            cur_rank -= 1
            best_rank = cur_rank
            best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
            trial -= 1  # re-check same index

    # Try merging pairs of columns (linear combination)
    for _round in range(3):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(cur_rank - 1):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, cur_rank):
                if time.time() > deadline - 5:
                    break
                # Try replacing columns i,j with a single column that is their sum
                # New column: U_i + U_j, V_i + V_j, W_i + W_j
                # This works if the cross terms cancel, which they won't in general
                # Instead, try: can we express col_i as a linear combo of others?
                # Simpler: try dropping i, then dropping j from the result
                pass

        # Try: for each pair (i,j), check if U_i+U_j, V_i+V_j, W_i+W_j combined with remaining works
        # This is equivalent to checking if the sum of rank-1 terms i and j can be replaced
        # by a single rank-1 term. This only works in special cases.
        # Alternative: try replacing two columns with one column that is a linear combination
        # with coefficient 1 on one and 0 on the other (i.e., just dropping)
        # Or try: new_col = U_i - U_j, V_i - V_j, W_i - W_j (if that helps)

    # Local search: try perturbing a column and re-optimizing
    # For each column, try small integer perturbations to other columns
    if time.time() < deadline - 10:
        for _iter in range(50):
            if time.time() > deadline - 10:
                break
            # Pick a random column to try modifying
            ci = random.randint(0, cur_rank - 1)
            cj = random.randint(0, cur_rank - 1)
            if ci == cj:
                continue
            # Try: U[:,ci] += U[:,cj] * alpha for alpha in {-1, 1}
            for alpha in [-1, 1]:
                new_U = cur_U.copy()
                new_U[:, ci] += alpha * cur_U[:, cj]
                # Check if we can drop column cj
                if np.all(new_U[:, cj] == 0) and np.all(cur_V[:, cj] == 0) and np.all(cur_W[:, cj] == 0):
                    pass  # already zero
                # This approach is getting complex. Let's try a different strategy.
                break

    # Final: try all pairs of columns for linear dependence
    if time.time() < deadline - 10:
        for i in range(cur_rank):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, cur_rank):
                if time.time() > deadline - 10:
                    break
                # Check if column j is a scalar multiple of column i in all three matrices
                # If U[:,j] = c*U[:,i], V[:,j] = c*V[:,i], W[:,j] = c*W[:,i], we can drop j
                Ui = cur_U[:, i]
                Uj = cur_V[:, j]  # typo check
                Uj = cur_U[:, j]
                Vi = cur_V[:, i]
                Vj = cur_V[:, j]
                Wi = cur_W[:, i]
                Wj = cur_W[:, j]

                # Check if they are proportional
                # Find non-zero entries
                nz_u = Ui != 0
                if not np.any(nz_u):
                    continue
                # Try c = 1 and c = -1
                for c in [1, -1]:
                    if np.array_equal(Uj, c * Ui) and np.array_equal(Vj, c * Vi) and np.array_equal(Wj, c * Wi):
                        # Drop column j
                        cur_U = np.delete(cur_U, j, axis=1)
                        cur_V = np.delete(cur_V, j, axis=1)
                        cur_W = np.delete(cur_W, j, axis=1)
                        cur_rank -= 1
                        best_rank = cur_rank
                        best_U, best_V, best_W = cur_U.copy(), cur_V.copy(), cur_W.copy()
                        break

    # Convert back to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
