import time
import numpy as np
from fractions import Fraction
from itertools import combinations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 145

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U_arr, V_arr, W_arr, rank):
        """Vectorized verification."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]

        rank = 7 * len(j_pairs) * len(k_pairs) + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))

        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
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

    def tensor_vector(U, V, W, t):
        """Compute the tensor vector for term t."""
        total_dim = n * m * p
        vec = np.zeros(total_dim, dtype=float)
        for i in range(n):
            for j in range(m):
                u_val = U[i*m+j, t]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j*p+k, t]
                    w_val = W[k*n+i, t]
                    if v_val == 0 or w_val == 0:
                        continue
                    vec[i*m*p + j*p + k] = u_val * v_val * w_val
        return vec

    def aggressive_reduction(U, V, W, rank):
        """Aggressive rank reduction using tensor vector linear dependencies."""
        if time_left() < 10:
            return U, V, W, rank

        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        total_dim = n * m * p
        max_rounds = 50

        for round_num in range(max_rounds):
            if time_left() < 10:
                break

            # Compute all tensor vectors
            vecs = np.zeros((rank, total_dim), dtype=float)
            for t in range(rank):
                vecs[t] = tensor_vector(U_arr, V_arr, W_arr, t)

            # Try to find a term whose vector is in the span of others
            eliminated = False
            for t in range(rank):
                if time_left() < 10:
                    break

                # Check if vecs[t] is in span of vecs[0:t] + vecs[t+1:rank]
                others = np.delete(vecs, t, axis=0)
                if len(others) == 0:
                    continue

                # Use least squares to check if vecs[t] is in span
                # Solve: minimize ||others^T * x - vecs[t]||
                # Use SVD for numerical stability
                try:
                    U_svd, s_vals, Vt = np.linalg.svd(others, full_matrices=False)
                    # Effective rank
                    eff_rank = np.sum(s_vals > 1e-10)
                    if eff_rank == 0:
                        continue

                    # Project vecs[t] onto span
                    proj = U_svd[:, :eff_rank] @ (U_svd[:, :eff_rank].T @ vecs[t])
                    residual = vecs[t] - proj
                    if np.linalg.norm(residual) < 1e-8 * max(1.0, np.linalg.norm(vecs[t])):
                        # vecs[t] is in span - eliminate it
                        U_arr[:, t] = 0
                        V_arr[:, t] = 0
                        W_arr[:, t] = 0
                        eliminated = True
                        break
                except:
                    continue

            if not eliminated:
                break

            # Remove the eliminated column
            # Find which column was zeroed
            zero_cols = [c for c in range(rank) if np.all(U_arr[:, c] == 0) and np.all(V_arr[:, c] == 0) and np.all(W_arr[:, c] == 0)]
            if zero_cols:
                keep = [c for c in range(rank) if c not in zero_cols]
                U_arr = U_arr[:, keep]
                V_arr = V_arr[:, keep]
                W_arr = W_arr[:, keep]
                rank = len(keep)

        return U_arr, V_arr, W_arr, rank

    def single_term_elimination(U, V, W, rank):
        """Try to eliminate individual terms."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        for _ in range(30):
            if time_left() < 5:
                break
            progress = False
            for t in range(rank):
                if time_left() < 5:
                    break
                U_test = U_arr.copy()
                V_test = V_arr.copy()
                W_test = W_arr.copy()
                U_test[:, t] = 0
                V_test[:, t] = 0
                W_test[:, t] = 0

                if verify_fast(U_test, V_test, W_test, rank):
                    keep = [c for c in range(rank) if c != t]
                    U_arr = U_arr[:, keep]
                    V_arr = V_arr[:, keep]
                    W_arr = W_arr[:, keep]
                    rank = len(keep)
                    progress = True
                    break
            if not progress:
                break

        return U_arr, V_arr, W_arr, rank

    def pairwise_elimination(U, V, W, rank):
        """Try to eliminate pairs of terms."""
        if time_left() < 15:
            return U, V, W, rank

        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        max_pairs = min(5000, rank * (rank - 1) // 2)
        pairs_checked = 0

        for idx1 in range(rank):
            if time_left() < 10 or pairs_checked >= max_pairs:
                break
            for idx2 in range(idx1 + 1, rank):
                if time_left() < 10 or pairs_checked >= max_pairs:
                    break
                pairs_checked += 1

                U_test = U_arr.copy()
                V_test = V_arr.copy()
                W_test = W_arr.copy()
                U_test[:, idx1] = 0
                U_test[:, idx2] = 0
                V_test[:, idx1] = 0
                V_test[:, idx2] = 0
                W_test[:, idx1] = 0
                W_test[:, idx2] = 0

                if verify_fast(U_test, V_test, W_test, rank):
                    keep = [c for c in range(rank) if c != idx1 and c != idx2]
                    U_arr = U_arr[:, keep]
                    V_arr = V_arr[:, keep]
                    W_arr = W_arr[:, keep]
                    rank = len(keep)
                    return pairwise_elimination(U_arr, V_arr, W_arr, rank)

        return U_arr, V_arr, W_arr, rank

    # Try different configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,2)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(3,5),(2,4)]),
        ([(0,1),(2,3)], [(1,2),(3,4),(0,5)]),
        ([(0,1),(2,3)], [(0,2),(3,5),(1,4)]),
        ([(0,1),(2,3)], [(0,1),(2,5),(3,4)]),
        ([(0,1),(2,3)], [(0,4),(1,2),(3,5)]),
        ([(0,2),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,3),(1,4)], [(0,1),(2,3),(4,5)]),
        ([(0,4),(1,2)], [(0,1),(2,3),(4,5)]),
    ]

    best_result = None
    best_rank = float('inf')

    for j_pairs, k_pairs in configurations:
        if time_left() < 20:
            break
        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)

        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue

        # Single term elimination
        U, V, W, rank = single_term_elimination(U, V, W, rank)

        # Aggressive SVD-based reduction
        if time_left() > 40:
            U, V, W, rank = aggressive_reduction(U, V, W, rank)

        # More single term elimination after aggressive reduction
        if time_left() > 30:
            U, V, W, rank = single_term_elimination(U, V, W, rank)

        # Pairwise elimination
        if time_left() > 20:
            U, V, W, rank = pairwise_elimination(U, V, W, rank)

        if rank < best_rank:
            best_rank = rank
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook if nothing worked
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
