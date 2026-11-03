import time
import numpy as np

DIMENSIONS = [2, 4, 6]
DEADLINE_S = 140.0  # safety margin below the 150s evaluator kill

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for u_idx in range(n * m):
        for v_idx in range(m * p):
            for w_idx in range(p * n):
                val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
                i = u_idx // m
                j = u_idx % m
                j2 = v_idx // p
                k = v_idx % p
                k2 = w_idx // n
                i2 = w_idx % n
                if i == i2 and j == j2 and k == k2:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block definitions
    # (A_start_row, A_start_col, B_start_row, B_start_col)
    # A is 2x4. B is 4x6.
    # A_left: rows 0-1, cols 0-1
    # A_right: rows 0-1, cols 2-3
    # B_top: rows 0-1
    # B_bot: rows 2-3
    
    blocks = [
        # (a_rows, a_cols, b_rows, b_cols)
        # A indices in U: i*m + j. i in a_rows, j in a_cols.
        # B indices in V: j*p + k. j in b_rows, k in b_cols.
        # W indices: k*n + i. k in b_cols, i in a_rows.
        
        # Block 0: A_left * B_top_left (cols 0-1 of B)
        ( [0, 1], [0, 1], [0, 1], [0, 1] ),
        # Block 1: A_left * B_top_mid (cols 2-3 of B)
        ( [0, 1], [0, 1], [0, 1], [2, 3] ),
        # Block 2: A_left * B_top_right (cols 4-5 of B)
        ( [0, 1], [0, 1], [0, 1], [4, 5] ),
        # Block 3: A_right * B_bot_left (cols 0-1 of B)
        ( [0, 1], [2, 3], [2, 3], [0, 1] ),
        # Block 4: A_right * B_bot_mid (cols 2-3 of B)
        ( [0, 1], [2, 3], [2, 3], [2, 3] ),
        # Block 5: A_right * B_bot_right (cols 4-5 of B)
        ( [0, 1], [2, 3], [2, 3], [4, 5] ),
    ]
    
    # Strassen components
    # Each component: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (local_a_idx, coeff)
    # v_coeffs: list of (local_b_idx, coeff)
    # w_coeffs: list of (local_z_idx, coeff)
    # local_a_idx: 0->A11, 1->A12, 2->A21, 3->A22
    # local_b_idx: 0->B11, 1->B12, 2->B21, 3->B22
    # local_z_idx: 0->Z11, 1->Z12, 2->Z21, 3->Z22
    
    strassen = [
        # M1
        ( [(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)] ),
        # M2
        ( [(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)] ),
        # M3
        ( [(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)] ),
        # M4
        ( [(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)] ),
        # M5
        ( [(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)] ),
        # M6
        ( [(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)] ),
        # M7
        ( [(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)] ),
    ]
    
    col = 0
    for block in blocks:
        a_rows, a_cols, b_rows, b_cols = block
        
        # Map local indices to global indices
        # Local A: 0->(a_rows[0], a_cols[0]), 1->(a_rows[0], a_cols[1]), 2->(a_rows[1], a_cols[0]), 3->(a_rows[1], a_cols[1])
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        
        # Local B: 0->(b_rows[0], b_cols[0]), 1->(b_rows[0], b_cols[1]), 2->(b_rows[1], b_cols[0]), 3->(b_rows[1], b_cols[1])
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        
        # Local Z: 0->(b_cols[0], a_rows[0]), 1->(b_cols[1], a_rows[0]), 2->(b_cols[0], a_rows[1]), 3->(b_cols[1], a_rows[1])
        # W index: k * n + i
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            for local_idx, coeff in u_coeffs:
                global_idx = a_map[local_idx]
                U[global_idx][col] += coeff
            
            for local_idx, coeff in v_coeffs:
                global_idx = b_map[local_idx]
                V[global_idx][col] += coeff
            
            for local_idx, coeff in w_coeffs:
                global_idx = w_map[local_idx]
                W[global_idx][col] += coeff
            
            col += 1
            
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition")

    # --- Rank-reduction pass: merge linearly dependent rank-1 terms ---
    # Each rank-1 term r contributes a Kronecker product U_r ⊗ V_r ⊗ W_r
    # to the multiplication tensor. If some subset of these vectors is
    # linearly dependent, the dependent terms can be removed (or the
    # decomposition re-expressed with fewer terms).
    #
    # We build the tensor as a 3D array T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # and check whether the rank-1 terms {u_r ⊗ v_r ⊗ w_r} are linearly
    # independent by forming a matrix whose columns are the vectorized
    # Kronecker products and checking its rank.
    #
    # For 42 terms the tensor has 2*4*6 = 48 entries, so the 3D tensor
    # lives in a 48-dim space. We can have at most 48 linearly independent
    # rank-1 terms. If rank < 42, we can in principle reduce.
    #
    # In practice, we check if any column of the Kronecker matrix is a
    # linear combination of others via QR with column pivoting, and if so,
    # remove the dependent column. The remaining columns still span the
    # same tensor, so the decomposition remains exact.

    t_start = time.time()
    n_terms = len(U[0]) if U else 0
    tensor_dim = n * m * p

    if n_terms > 1 and tensor_dim < n_terms:
        # Build Kronecker matrix: columns are vec(u_r ⊗ v_r ⊗ w_r)
        # For each term r, the contribution to T[i,j,k] is U[i,r]*V[j,r]*W[k,r]
        # The vectorized Kronecker product is a tensor_dim vector.
        K = np.zeros((tensor_dim, n_terms), dtype=float)
        for r in range(n_terms):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * (m * p) + j * p + k
                        K[idx, r] = U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]

        # QR with column pivoting to find independent columns
        # We want to find the largest subset of linearly independent columns
        try:
            # Use SVD for numerical rank detection
            U_svd, s_vals, Vt = np.linalg.svd(K, full_matrices=False)
            tol = 1e-10 * max(K.shape)
            num_indep = sum(1 for s in s_vals if s > tol)

            if num_indep < n_terms:
                # Find which columns are independent via QR with pivoting
                # Use scipy if available, otherwise numpy QR
                from scipy.linalg import qr
                Q, P = qr(K, pivoting=True)
                # P is the permutation matrix; the first num_indep columns
                # (in permuted order) are independent
                perm = np.argmax(P, axis=0)  # column permutation
                indep_cols = perm[:num_indep]

                # Remove dependent columns
                keep = set(indep_cols.tolist())
                new_U = []
                new_V = []
                new_W = []
                for r in range(n_terms):
                    if r in keep:
                        new_U.append(U[r])
                        new_V.append(V[r])
                        new_W.append(W[r])

                # Verify the reduced decomposition
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    U, V, W = new_U, new_V, new_W
        except Exception:
            pass  # keep the original 42-term decomposition

    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Invalid decomposition after reduction")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
