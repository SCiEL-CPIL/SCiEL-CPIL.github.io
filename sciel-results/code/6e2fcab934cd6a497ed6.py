"""Matrix multiplication 2x4x6 rank-minimal decomposition via Strassen block structure."""
import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    t0 = time.time()
    deadline = t0 + 110  # safety margin before 120s kill

    # ---------- Direct Strassen-based rank-42 decomposition ----------
    # Decompose 2x4 * 4x6 into 6 blocks of 2x2 * 2x2 multiplications.
    # A (2x4) = [A_L | A_R], each 2x2 (cols 0-1, cols 2-3)
    # B (4x6) split into 3 blocks of 4x2: B_0, B_1, B_2 (cols 0-1, 2-3, 4-5)
    # Each B_b = [B_{b,top}; B_{b,bot}], each 2x2.
    # C = A_L * B_{b,top} + A_R * B_{b,bot} for each block b, summed.
    # Total: 3 blocks * 2 (top/bot) * 7 (Strassen) = 42 rank terms.

    rank = 42
    U = np.zeros((n * m, rank))   # 8 x 42
    V = np.zeros((m * p, rank))   # 24 x 42
    W = np.zeros((p * n, rank))   # 12 x 42

    # Strassen coefficients for 2x2x2 matrix multiplication.
    # A, B, C are 2x2 with indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    U_strassen = [
        [(0, 1), (3, 1)],       # P1: A00, A11
        [(2, 1), (3, 1)],       # P2: A10, A11
        [(0, 1)],               # P3: A00
        [(3, 1)],               # P4: A11
        [(0, 1), (1, 1)],       # P5: A00, A01
        [(2, 1), (0, -1)],      # P6: A10, -A00
        [(1, 1), (3, -1)]       # P7: A01, -A11
    ]

    V_strassen = [
        [(0, 1), (3, 1)],       # P1: B00, B11
        [(0, 1)],               # P2: B00
        [(1, 1), (3, -1)],      # P3: B01, -B11
        [(2, 1), (0, -1)],      # P4: B10, -B00
        [(3, 1)],               # P5: B11
        [(0, 1), (1, 1)],       # P6: B00, B01
        [(2, 1), (3, 1)]        # P7: B10, B11
    ]

    W_strassen = [
        [(0, 1), (3, 1)],       # P1: C00, C11
        [(2, 1), (3, -1)],      # P2: C10, -C11
        [(1, 1), (3, 1)],       # P3: C01, C11
        [(0, 1), (2, 1)],       # P4: C00, C10
        [(0, -1), (1, 1)],      # P5: -C00, C01
        [(3, 1)],               # P6: C11
        [(0, 1)]                # P7: C00
    ]

    col = 0
    for b in range(3):        # 3 column blocks of B (each 2 cols)
        for a in range(2):    # a=0: top (A_L, B_top), a=1: bottom (A_R, B_bot)
            for s in range(7):
                # Update U (A matrix, 2x4, index i*m + j)
                for (idx, coeff) in U_strassen[s]:
                    i_loc = idx // 2
                    j_loc = idx % 2
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx, col] += coeff

                # Update V (B matrix, 4x6, index j*p + k)
                for (idx, coeff) in V_strassen[s]:
                    j_loc = idx // 2
                    k_loc = idx % 2
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx, col] += coeff

                # Update W (Output C, 2x6, index k*n + i per trace convention W[k*n+i, r])
                for (idx, coeff) in W_strassen[s]:
                    i_loc = idx // 2
                    k_loc = idx % 2
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx, col] += coeff

                col += 1

    # ---------- Verification ----------
    def target_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i_A in range(n):
            for j in range(m):
                for k_B in range(p):
                    i = i_A * m + j
                    j_idx = j * p + k_B
                    k = k_B * n + i_A
                    T[i, j_idx, k] = 1.0
        return T

    T_target = target_tensor()
    T_comp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T_comp, T_target, atol=1e-9):
        # Fallback: schoolbook rank-48
        rank0 = n * m * p
        U = np.zeros((n * m, rank0))
        V = np.zeros((m * p, rank0))
        W = np.zeros((p * n, rank0))
        r = 0
        for i_A in range(n):
            for j in range(m):
                for k_B in range(p):
                    i = i_A * m + j
                    j_idx = j * p + k_B
                    k = k_B * n + i_A
                    U[i, r] = 1.0
                    V[j_idx, r] = 1.0
                    W[k, r] = 1.0
                    r += 1

    # ---------- Rank reduction: try to remove redundant columns ----------
    # For each column r, check if its tensor contribution (outer product of
    # U[:,r], V[:,r], W[:,r]) can be expressed as a linear combination of
    # the other columns' contributions. If so, remove it.
    def tensor_of_column(U, V, W, r):
        return np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])

    def full_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def verify(U, V, W, T_target, atol=1e-9):
        T_comp = full_tensor(U, V, W)
        return np.allclose(T_comp, T_target, atol=atol)

    def try_remove_column(U, V, W, r, T_target):
        """Try to remove column r. Return (success, new_U, new_V, new_W)."""
        nr = U.shape[1]
        if nr <= 1:
            return False, U, V, W
        # Build the "basis" matrix from all other columns
        keep = [c for c in range(nr) if c != r]
        # For each other column c, compute its 3D tensor as a flat vector
        # Then solve for coefficients such that sum_c alpha_c * T_c = T_r
        # T_c = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        # This is a linear system: M @ alpha = T_r_flat
        # where M[:,c] = T_c_flat
        # But this is expensive. Instead, use a cheaper approach:
        # Check if the column is in the span by verifying that
        # the 3D tensor of column r is in the span of the others.
        # We can do this by checking if the residual after projecting
        # T_r onto span{others} is zero.
        
        # Build matrix A where each column is the flattened tensor of column c
        dim = U.shape[0] * V.shape[0] * W.shape[0]
        A = np.zeros((dim, len(keep)))
        for ci, c in enumerate(keep):
            A[:, ci] = tensor_of_column(U, V, W, c).ravel()
        
        b = tensor_of_column(U, V, W, r).ravel()
        
        # Solve least squares
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except np.linalg.LinAlgError:
            return False, U, V, W
        
        # Check if the reconstruction is close enough
        recon = A @ alpha
        if np.allclose(recon, b, atol=1e-9):
            # Column r is redundant; remove it
            new_U = np.delete(U, r, axis=1)
            new_V = np.delete(V, r, axis=1)
            new_W = np.delete(W, r, axis=1)
            # Verify
            if verify(new_U, new_V, new_W, T_target):
                return True, new_U, new_V, new_W
        return False, U, V, W

    # Iteratively try to remove columns
    changed = True
    while changed:
        changed = False
        nr = U.shape[1]
        for r in range(nr):
            if time.time() > deadline:
                break
            success, U, V, W = try_remove_column(U, V, W, r, T_target)
            if success:
                changed = True
                break  # Restart from 0 since indices shifted
        if time.time() > deadline:
            break

    # Round to nearest half-integer for cleanliness
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2

    # Final verification after rounding
    T_comp = full_tensor(U, V, W)
    if not np.allclose(T_comp, T_target, atol=1e-9):
        # Unrounded Strassen (rebuild from scratch)
        U = np.zeros((n * m, 42))
        V = np.zeros((m * p, 42))
        W = np.zeros((p * n, 42))
        col = 0
        for b in range(3):
            for a in range(2):
                for s in range(7):
                    for (idx, coeff) in U_strassen[s]:
                        i_loc = idx // 2
                        j_loc = idx % 2
                        U[i_loc * m + a * 2 + j_loc, col] += coeff
                    for (idx, coeff) in V_strassen[s]:
                        j_loc = idx // 2
                        k_loc = idx % 2
                        V[(a * 2 + j_loc) * p + b * 2 + k_loc, col] += coeff
                    for (idx, coeff) in W_strassen[s]:
                        i_loc = idx // 2
                        k_loc = idx % 2
                        W[(b * 2 + k_loc) * n + i_loc, col] += coeff
                    col += 1
        # Try rank reduction again on unrounded version
        changed = True
        while changed:
            changed = False
            nr = U.shape[1]
            for r in range(nr):
                if time.time() > deadline:
                    break
                success, U, V, W = try_remove_column(U, V, W, r, T_target)
                if success:
                    changed = True
                    break
            if time.time() > deadline:
                break

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
