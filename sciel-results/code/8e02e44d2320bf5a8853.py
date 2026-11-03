import time
import numpy as np
from itertools import combinations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 135

    def time_left():
        return deadline - time.time()

    def build_constraint_matrix(U, V, rank):
        """Build the constraint matrix A such that A @ W_flat = 1 for all tensor entries.
        A has shape (n*m*p, p*n*rank).
        A[i*m*p + j*p + k, (k*n+i)*rank + r] = U[i*m+j, r] * V[j*p+k, r]
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        num_eqs = n * m * p
        num_unk = p * n * rank
        A = np.zeros((num_eqs, num_unk))
        eq = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        A[eq, (k * n + i) * rank + r] = U_arr[i * m + j, r] * V_arr[j * p + k, r]
                    eq += 1
        return A

    def verify_with_matrix(A, W_flat, tol=1e-8):
        """Check if A @ W_flat = 1 within tolerance."""
        return np.max(np.abs(A @ W_flat - 1.0)) < tol

    def verify_full(U, V, W, rank, tol=1e-8):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition for given j-pairs and k-pairs."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]

        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # Strassen 7 terms for 2x2x2 block
                # m1 = (A11+A22)(B11+B22)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # m2 = (A21+A22)B11
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # m3 = A11(B21-B22)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # m4 = A22(B12-B11)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # m5 = (A11+A12)B22
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # m6 = (A21-A11)(B11+B12)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # m7 = (A12-A22)(B21+B22)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1

        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1

        return U, V, W, rank

    def try_eliminate_terms_fast(U, V, W, rank):
        """Try to eliminate redundant terms using constraint matrix."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        changed = True
        while changed and time_left() > 8:
            changed = False
            for term_idx in range(rank - 1, -1, -1):
                if time_left() < 8:
                    break
                # Build constraint matrix without this term
                U_sub = np.delete(U_arr, term_idx, axis=1)
                V_sub = np.delete(V_arr, term_idx, axis=1)
                W_sub = np.delete(W_arr, term_idx, axis=1)
                new_rank = rank - 1

                if new_rank == 0:
                    continue

                A = build_constraint_matrix(U_sub, V_sub, new_rank)
                b = np.ones(n * m * p)

                try:
                    W_flat, residuals, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
                    if np.max(np.abs(A @ W_flat - b)) < 1e-8:
                        W_snapped = np.round(W_flat * 2) / 2
                        if np.max(np.abs(A @ W_snapped - b)) < 1e-8:
                            U_arr = U_sub
                            V_arr = V_sub
                            W_arr = W_snapped.reshape(p * n, new_rank)
                            rank = new_rank
                            changed = True
                            break
                except:
                    pass

        return U_arr, V_arr, W_arr, rank

    def try_pairwise_elimination_fast(U, V, W, rank):
        """Try to eliminate pairs of terms."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        improved = True
        while improved and time_left() > 15:
            improved = False
            for a in range(rank - 1, -1, -1):
                if time_left() < 15:
                    break
                for b in range(a - 1, -1, -1):
                    if time_left() < 15:
                        break
                    keep = [j for j in range(rank) if j != a and j != b]
                    U_sub = U_arr[:, keep]
                    V_sub = V_arr[:, keep]
                    new_rank = rank - 2

                    if new_rank == 0:
                        continue

                    A = build_constraint_matrix(U_sub, V_sub, new_rank)
                    b_vec = np.ones(n * m * p)

                    try:
                        W_flat, residuals, rnk, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                        if np.max(np.abs(A @ W_flat - b_vec)) < 1e-8:
                            W_snapped = np.round(W_flat * 2) / 2
                            if np.max(np.abs(A @ W_snapped - b_vec)) < 1e-8:
                                U_arr = U_sub
                                V_arr = V_sub
                                W_arr = W_snapped.reshape(p * n, new_rank)
                                rank = new_rank
                                improved = True
                                break
                    except:
                        pass
                if improved:
                    break

        return U_arr, V_arr, W_arr, rank

    def solve_for_W(U, V, rank):
        """Given U and V, solve for W."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        A = build_constraint_matrix(U_arr, V_arr, rank)
        b = np.ones(n * m * p)
        try:
            W_flat, residuals, rnk, sv = np.linalg.lstsq(A, b, rcond=None)
            if np.max(np.abs(A @ W_flat - b)) > 1e-8:
                return None
            W_arr = W_flat.reshape(p * n, rank)
            W_snapped = np.round(W_arr * 2) / 2
            if np.max(np.abs(A @ W_snapped.flatten() - b)) > 1e-8:
                # Try integer snap
                W_int = np.round(W_arr)
                if np.max(np.abs(A @ W_int.flatten() - b)) < 1e-8:
                    return W_int.tolist()
                return None
            return W_snapped.tolist()
        except:
            return None

    def build_strassen_and_optimize(j_pairs, k_pairs):
        """Build Strassen decomposition and try to reduce rank."""
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)

        if not verify_full(U.tolist(), V.tolist(), W.tolist(), rank):
            return None, None, None, rank

        U, V, W, rank = try_eliminate_terms_fast(U, V, W, rank)
        U, V, W, rank = try_pairwise_elimination_fast(U, V, W, rank)

        if not verify_full(U.tolist(), V.tolist(), W.tolist(), rank):
            return None, None, None, rank

        return U.tolist(), V.tolist(), W.tolist(), rank

    # Generate configurations
    configurations = []

    # 2 j-pairs, 3 k-pairs: rank = 7*6 + 2*1*6 = 54
    for jp1 in combinations(range(5), 2):
        for jp2 in combinations(range(5), 2):
            if set(jp1) & set(jp2):
                continue
            if jp1 > jp2:
                continue
            for kp1 in combinations(range(6), 2):
                for kp2 in combinations(range(6), 2):
                    if set(kp1) & set(kp2):
                        continue
                    if kp1 > kp2:
                        continue
                    for kp3 in combinations(range(6), 2):
                        if set(kp1) & set(kp3) or set(kp2) & set(kp3):
                            continue
                        if kp1 > kp2 or kp2 > kp3:
                            continue
                        configurations.append(([list(jp1), list(jp2)], [list(kp1), list(kp2), list(kp3)]))

    # 2 j-pairs, 2 k-pairs: rank = 7*4 + 2*1*6 = 40
    for jp1 in combinations(range(5), 2):
        for jp2 in combinations(range(5), 2):
            if set(jp1) & set(jp2):
                continue
            if jp1 > jp2:
                continue
            for kp1 in combinations(range(6), 2):
                for kp2 in combinations(range(6), 2):
                    if set(kp1) & set(kp2):
                        continue
                    if kp1 > kp2:
                        continue
                    configurations.append(([list(jp1), list(jp2)], [list(kp1), list(kp2)]))

    # 1 j-pair, 3 k-pairs: rank = 7*3 + 2*3*6 = 51
    for jp1 in combinations(range(5), 2):
        for kp1 in combinations(range(6), 2):
            for kp2 in combinations(range(6), 2):
                if set(kp1) & set(kp2):
                    continue
                if kp1 > kp2:
                    continue
                for kp3 in combinations(range(6), 2):
                    if set(kp1) & set(kp3) or set(kp2) & set(kp3):
                        continue
                    if kp1 > kp2 or kp2 > kp3:
                        continue
                    configurations.append(([list(jp1)], [list(kp1), list(kp2), list(kp3)]))

    # 1 j-pair, 2 k-pairs: rank = 7*2 + 2*3*6 = 50
    for jp1 in combinations(range(5), 2):
        for kp1 in combinations(range(6), 2):
            for kp2 in combinations(range(6), 2):
                if set(kp1) & set(kp2):
                    continue
                if kp1 > kp2:
                    continue
                configurations.append(([list(jp1)], [list(kp1), list(kp2)]))

    def expected_rank(jp, kp):
        num_blocks = len(jp) * len(kp)
        used_j = set()
        for j0, j1 in jp:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = m - len(used_j)
        return 7 * num_blocks + 2 * remaining_j * p

    configurations.sort(key=lambda x: expected_rank(x[0], x[1]))

    best_result = None
    best_rank = float('inf')

    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U_list, V_list, W_list, rank = build_strassen_and_optimize(j_pairs, k_pairs)

        if U_list is not None and rank < best_rank:
            best_rank = rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}

    # Try random structured approaches for lower rank
    if time_left() > 25:
        np.random.seed(123)
        for attempt in range(50):
            if time_left() < 20:
                break

            target_rank = max(30, best_rank - 8)
            for test_rank in range(target_rank, best_rank):
                if time_left() < 15:
                    break

                U_test = np.zeros((n * m, test_rank))
                V_test = np.zeros((m * p, test_rank))

                # Sparse random structure
                for i in range(n):
                    for j in range(m):
                        num_nonzero = np.random.randint(2, 4)
                        cols = np.random.choice(test_rank, size=num_nonzero, replace=False)
                        for c in cols:
                            U_test[i * m + j, c] = np.random.choice([1, -1])

                for j in range(m):
                    for k in range(p):
                        num_nonzero = np.random.randint(2, 4)
                        cols = np.random.choice(test_rank, size=num_nonzero, replace=False)
                        for c in cols:
                            V_test[j * p + k, c] = np.random.choice([1, -1])

                W_test = solve_for_W(U_test, V_test, test_rank)
                if W_test is not None:
                    if verify_full(U_test.tolist(), V_test.tolist(), W_test, test_rank):
                        if test_rank < best_rank:
                            best_rank = test_rank
                            best_result = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test}
                        break

    # Fallback to schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
