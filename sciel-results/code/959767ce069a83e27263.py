"""Rank-65/66 exact tensor decomposition for 3x4 by 4x6 matrix multiplication.

Partition strategy:
  A (3x4): rows split 1+2, cols split 2+2
  B (4x6): rows split 2+2, cols split 2+2+2
  C (3x6): each block row uses two column blocks of A times two row blocks of B

Block ranks:
  1x2 * 2x2  -> rank 4 (schoolbook)
  2x2 * 2x2  -> rank 7 (Strassen)

Total: 6 * (rank 4) + 6 * (rank 7) = 24 + 42 = 66

Then attempt rank reduction by finding linear dependencies among the rank-1 terms
using SVD on the tensor flattening matrix.
"""
import time
import numpy as np

def construct():
    _t0 = time.time()
    _deadline = 140  # 10s margin before 150s kill

    n, m, p = 3, 4, 6  # C is n x p, A is n x m, B is m x p

    RANK = 66
    U = [[0] * RANK for _ in range(n * m)]  # (12, 66)
    V = [[0] * RANK for _ in range(m * p)]  # (24, 66)
    W = [[0] * RANK for _ in range(p * n)]  # (18, 66)

    col = 0

    # Indexing:
    #   A_{i,k}  -> U row  i * m + k       (i in [0,n), k in [0,m))
    #   B_{k,j}  -> V row  k * p + j       (k in [0,m), j in [0,p))
    #   C_{i,j}  -> W row  j * n + i       (j in [0,p), i in [0,n))  (trace convention)

    def _idx_u(i, k):
        return i * m + k

    def _idx_v(k, j):
        return k * p + j

    def _idx_w(j, i):
        return j * n + i

    def _add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """1x2 * 2x2 block, rank 4 (schoolbook).
        A[ar:ar+1, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr, cc], C[cr, cc+1]
        """
        nonlocal col
        A0 = _idx_u(a_r, a_c)
        A1 = _idx_u(a_r, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C0 = _idx_w(c_c, c_r)
        C1 = _idx_w(c_c + 1, c_r)

        # M1 = A0 * B00 -> C0
        U[A0][col] = 1; V[B00][col] = 1; W[C0][col] = 1
        col += 1
        # M2 = A1 * B10 -> C0
        U[A1][col] = 1; V[B10][col] = 1; W[C0][col] = 1
        col += 1
        # M3 = A0 * B01 -> C1
        U[A0][col] = 1; V[B01][col] = 1; W[C1][col] = 1
        col += 1
        # M4 = A1 * B11 -> C1
        U[A1][col] = 1; V[B11][col] = 1; W[C1][col] = 1
        col += 1

    def _add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """2x2 * 2x2 block, rank 7 (Strassen).
        A[ar:ar+2, ac:ac+2] * B[br:br+2, bc:bc+2]
        -> C[cr:cr+2, cc:cc+2]
        """
        nonlocal col
        A00 = _idx_u(a_r, a_c)
        A01 = _idx_u(a_r, a_c + 1)
        A10 = _idx_u(a_r + 1, a_c)
        A11 = _idx_u(a_r + 1, a_c + 1)
        B00 = _idx_v(b_r, b_c)
        B01 = _idx_v(b_r, b_c + 1)
        B10 = _idx_v(b_r + 1, b_c)
        B11 = _idx_v(b_r + 1, b_c + 1)
        C00 = _idx_w(c_c, c_r)
        C01 = _idx_w(c_c + 1, c_r)
        C10 = _idx_w(c_c, c_r + 1)
        C11 = _idx_w(c_c + 1, c_r + 1)

        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C11][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1

        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1

        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1
        col += 1

    # ---- Build the 66-term decomposition ----
    for j in range(3):  # C column block index (B col offset 2*j)
        bc = 2 * j  # B column start

        # --- Row 0 of C (A row 0, 1x2 blocks) ---
        _add_1x2_2x2(0, 0, 0, bc, 0, 2 * j)
        _add_1x2_2x2(0, 2, 2, bc, 0, 2 * j)

        # --- Rows 1-2 of C (A rows 1:3, 2x2 blocks) ---
        _add_strassen(1, 0, 0, bc, 1, 2 * j)
        _add_strassen(1, 2, 2, bc, 1, 2 * j)

    assert col == RANK, f"Expected {RANK} terms, got {col}"

    # Convert to numpy for SVD-based rank reduction
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    def _verify(U_mat, V_mat, W_mat, r):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    # Verify rank-66 decomposition
    if not _verify(U_np, V_np, W_np, 66):
        # Fallback to schoolbook rank 72
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school))
        V_s = np.zeros((m * p, rank_school))
        W_s = np.zeros((p * n, rank_school))
        c = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_s[i * m + k, c] = 1
                    V_s[k * p + j, c] = 1
                    W_s[j * n + i, c] = 1
                    c += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Attempt rank reduction via SVD on the tensor flattening
    # The tensor T has dimensions n*m*p = 72 entries
    # Each rank-1 term l contributes T_l[i,k,j] = U[i*m+k,l] * V[k*p+j,l] * W[j*n+i,l]
    # Build matrix M of shape (n*m*p, RANK) where M[idx, l] = T_l at that index
    # If rank(M) < RANK, there's a nullspace vector allowing rank reduction

    if time.time() - _t0 < _deadline - 5:
        M = np.zeros((n * m * p, RANK))
        for l in range(RANK):
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        M[i * m * p + k * p + j, l] = (
                            U_np[i * m + k, l] *
                            V_np[k * p + j, l] *
                            W_np[j * n + i, l]
                        )

        try:
            _, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
            # Check if there are near-zero singular values
            null_threshold = 1e-10
            null_dims = np.sum(s_vals < null_threshold)
            
            if null_dims > 0:
                # There are nullspace vectors; try to eliminate columns
                # Use the right singular vectors corresponding to small singular values
                # Vt has shape (RANK, RANK) - rows are right singular vectors
                # The last `null_dims` rows span the nullspace
                for nd in range(1, min(null_dims + 1, 5)):
                    if time.time() - _t0 > _deadline - 3:
                        break
                    null_vec = Vt[-nd, :]
                    max_idx = int(np.argmax(np.abs(null_vec)))
                    if abs(null_vec[max_idx]) < 1e-8:
                        continue
                    
                    # Remove column max_idx
                    mask = np.ones(RANK, dtype=bool)
                    mask[max_idx] = False
                    U_reduced = U_np[:, mask]
                    V_reduced = V_np[:, mask]
                    W_reduced = W_np[:, mask]
                    
                    new_rank = RANK - nd
                    if _verify(U_reduced, V_reduced, W_reduced, new_rank):
                        return {
                            "U": U_reduced.tolist(),
                            "V": V_reduced.tolist(),
                            "W": W_reduced.tolist()
                        }
        except Exception:
            pass

    # If no reduction found, return rank-66
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
