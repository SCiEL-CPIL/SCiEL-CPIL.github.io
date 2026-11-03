"""Strassen-based rank reduction for 4x4 * 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140  # safety margin before 150s kill


def _build_strassen_4x4x4():
    """
    Build a rank-49 exact tensor decomposition for 4x4 * 4x4 matrix multiplication
    using nested Strassen (Strassen at both block levels).
    
    Returns U, V, W as numpy arrays of shape (16, 49), (16, 49), (16, 49) respectively.
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Top-level Strassen terms for 4x4*4x4
    # Each term t has X (2x2) and Y (2x2) matrices formed from A and B blocks
    # X[i,j] is a dict: {(a_row, a_col): coeff} for A entries
    # Y[i,j] is a dict: {(b_row, b_col): coeff} for B entries
    # c_contrib: dict mapping C_block (0,1,2,3) to coefficient
    #   Block 0 = C11 (rows 0-1, cols 0-1)
    #   Block 1 = C12 (rows 0-1, cols 2-3)
    #   Block 2 = C21 (rows 2-3, cols 0-1)
    #   Block 3 = C22 (rows 2-3, cols 2-3)
    
    # Define A block entries: A_block(r, c) where r,c in 0,1
    # A11: (0,0),(0,1),(1,0),(1,1)
    # A12: (0,2),(0,3),(1,2),(1,3)
    # A21: (2,0),(2,1),(3,0),(3,1)
    # A22: (2,2),(2,3),(3,2),(3,3)
    
    # Define B block entries: B_block(r, c) where r,c in 0,1
    # B11: (0,0),(0,1),(1,0),(1,1)
    # B12: (0,2),(0,3),(1,2),(1,3)
    # B21: (2,0),(2,1),(3,0),(3,1)
    # B22: (2,2),(2,3),(3,2),(3,3)
    
    def a_entry(block_r, block_c, local_r, local_c):
        """Map A block (block_r, block_c) local (local_r, local_c) to global (i, j)"""
        i = 2 * block_r + local_r
        j = 2 * block_c + local_c
        return (i, j)
    
    def b_entry(block_r, block_c, local_r, local_c):
        """Map B block (block_r, block_c) local (local_r, local_c) to global (j, k)"""
        j = 2 * block_r + local_r
        k = 2 * block_c + local_c
        return (j, k)
    
    # Top-level Strassen terms: (X_coeffs, Y_coeffs, C_contrib)
    # X_coeffs: 2x2 list of dicts, each dict maps (i,j) -> coeff
    # Y_coeffs: 2x2 list of dicts, each dict maps (j,k) -> coeff
    # C_contrib: dict mapping block_idx -> coeff
    
    terms = []
    
    # T0: X = A11+A22, Y = B11+B22, C11:+1, C22:+1
    X0 = [[{} for _ in range(2)] for _ in range(2)]
    Y0 = [[{} for _ in range(2)] for _ in range(2)]
    for lr in range(2):
        for lc in range(2):
            i, j = a_entry(0, 0, lr, lc)
            X0[lr][lc][(i, j)] = 1
            i, j = a_entry(1, 1, lr, lc)
            X0[lr][lc][(i, j)] = 1
            j, k = b_entry(0, 0, lr, lc)
            Y0[lr][lc][(j, k)] = 1
            j, k = b_entry(1, 1, lr, lc)
            Y0[lr][lc][(j, k)] = 1
    terms.append((X0, Y0, {0: 1, 3: 1}))
    
    # T1: X = A22, Y = B11+B22, C11:-1, C21:+1
    X1 = [[{} for _ in range(2)] for _ in range(2)]
    Y1 = [[{} for _ in range(2)] for _ in range(2)]
    for lr in range(2):
        for lc in range(2):
            i, j = a_entry(1, 1, lr, lc)
            X1[lr][lc][(i, j)] = 1
            j, k = b_entry(0, 0, lr, lc)
            Y1[lr][lc][(j, k)] = 1
            j, k = b_entry(1, 1, lr, lc)
            Y1[lr][lc][(j, k)] = 1
    terms.append((X1, Y1, {0: -1, 2: 1}))
    
    # T2: X = A11-A22, Y = B21+B22, C12:+1, C22:+1
    X2 = [[{} for _ in range(2)] for _ in range(2)]
    Y2 = [[{} for _ in range(2)] for _ in range(2)]
    for lr in range(2):
        for lc in range(2):
            i, j = a_entry(0, 0, lr, lc)
            X2[lr][lc][(i, j)] = 1
            i, j = a_entry(1, 1, lr, lc)
            X2[lr][lc][(i, j)] = -1
            j, k = b_entry(1, 0, lr, lc)
            Y2[lr][lc][(j, k)] = 1
            j, k = b_entry(1, 1, lr, lc)
            Y2[lr][lc][(j, k)] = 1
    terms.append((X2, Y2, {1: 1, 3: 1}))
    
    # T3: X = A11, Y = B11-B21, C11:+1, C21:+1
    X3 = [[{} for _ in range(2)] for _ in range(2)]
    Y3 = [[{} for _ in range(2)] for _ in range(2)]
    for lr in range(2):
        for lc in range(2):
            i, j = a_entry(0, 0, lr, lc)
            X3[lr][lc][(i, j)] = 1
            j, k = b_entry(0, 0, lr, lc)
            Y3[lr][lc][(j, k)] = 1
            j, k = b_entry(1, 0, lr, lc)
            Y3[lr][lc][(j, k)] = -1
    terms.append((X3, Y3, {0: 1, 2: 1}))
    
    # T4: X = A11+A12, Y = B22, C12:+1, C22:-1
    X4 = [[{} for _ in range(2)] for _ in range(2)]
    Y4 = [[{} for _ in range(2)] for _ in range(2)]
    for lr in range(2):
        for lc in range(2):
            i, j = a_entry(0, 0, lr, lc)
            X4[lr][lc][(i, j)] = 1
            i, j = a_entry(0, 1, lr, lc)
            X4[lr][lc][(i, j)] = 1
            j, k = b_entry(1, 1, lr, lc)
            Y4[lr][lc][(j, k)] = 1
    terms.append((X4, Y4, {1: 1, 3: -1}))
    
    # T5: X = A21+A22, Y = B11-B12, C11:+1
    X5 = [[{} for _ in range(2)] for _ in range(2)]
    Y5 = [[{} for _ in range(2)] for _ in range(2)]
    for lr in range(2):
        for lc in range(2):
            i, j = a_entry(1, 0, lr, lc)
            X5[lr][lc][(i, j)] = 1
            i, j = a_entry(1, 1, lr, lc)
            X5[lr][lc][(i, j)] = 1
            j, k = b_entry(0, 0, lr, lc)
            Y5[lr][lc][(j, k)] = 1
            j, k = b_entry(0, 1, lr, lc)
            Y5[lr][lc][(j, k)] = -1
    terms.append((X5, Y5, {0: 1}))
    
    # T6: X = A12-A22, Y = B21+B22, C22:+1
    X6 = [[{} for _ in range(2)] for _ in range(2)]
    Y6 = [[{} for _ in range(2)] for _ in range(2)]
    for lr in range(2):
        for lc in range(2):
            i, j = a_entry(0, 1, lr, lc)
            X6[lr][lc][(i, j)] = 1
            i, j = a_entry(1, 1, lr, lc)
            X6[lr][lc][(i, j)] = -1
            j, k = b_entry(1, 0, lr, lc)
            Y6[lr][lc][(j, k)] = 1
            j, k = b_entry(1, 1, lr, lc)
            Y6[lr][lc][(j, k)] = 1
    terms.append((X6, Y6, {3: 1}))
    
    # Now for each top-level term, apply Strassen 2x2*2x2 to get 7 sub-terms.
    # X_t = [[x00, x01], [x10, x11]], Y_t = [[y00, y01], [y10, y11]]
    # Strassen sub-terms:
    # S1 = (x00+x11)(y00+y11) -> C[0,0] += S1, C[1,1] += S1
    # S2 = x11(y00+y11)       -> C[0,0] -= S2, C[1,0] += S2
    # S3 = (x00-x11)(y10+y11) -> C[0,1] += S3, C[1,1] += S3
    # S4 = x00(y00-y10)       -> C[0,0] += S4, C[1,0] += S4
    # S5 = (x00+x01)y11       -> C[0,1] += S5, C[1,1] -= S5
    # S6 = (x10+x11)(y00-y01) -> C[0,0] += S6
    # S7 = (x01-x11)(y10+y11) -> C[1,1] += S7
    
    # Each St is a product of two linear combinations.
    # For a product P = (sum_a ca * A[ia,ja]) * (sum_b cb * B[jb,kb]):
    # This is a rank-1 term where:
    #   U contribution: for each (ia,ja) with coeff ca, U[ia*4+ja, r] += ca
    #   V contribution: for each (jb,kb) with coeff cb, V[jb*4+kb, r] += cb
    #   W contribution: for each C entry that this product contributes to, W[k*4+i, r] += coeff
    
    # The C entries that (X_t*Y_t)[p,q] contributes to:
    # p,q in 0,1 -> local position in the 2x2 block
    # If Tt contributes to C_block with coefficient c, then (X_t*Y_t)[p,q]
    # contributes to C[2*block_row+p, 2*block_col+q] with coefficient c.
    
    # Block 0: C11 -> rows 0-1, cols 0-1 -> C[0,0], C[0,1], C[1,0], C[1,1]
    # Block 1: C12 -> rows 0-1, cols 2-3 -> C[0,2], C[0,3], C[1,2], C[1,3]
    # Block 2: C21 -> rows 2-3, cols 0-1 -> C[2,0], C[2,1], C[3,0], C[3,1]
    # Block 3: C22 -> rows 2-3, cols 2-3 -> C[2,2], C[2,3], C[3,2], C[3,3]
    
    # For C_block b with local position (p,q):
    #   block_row = b // 2, block_col = b % 2
    #   global_i = 2 * block_row + p
    #   global_k = 2 * block_col + q
    #   W index: k * 4 + i = global_k * 4 + global_i
    
    r_idx = 0
    for t_idx in range(7):
        X_t, Y_t, c_contrib = terms[t_idx]
        x00, x01 = X_t[0], X_t[0]  # x00 is dict, x01 is dict
        x00, x01 = X_t[0], X_t[0]
        x00, x01 = X_t[0], X_t[0]
        # Fix: X_t is 2x2, so X_t[0][0]=x00, X_t[0][1]=x01, X_t[1][0]=x10, X_t[1][1]=x11
        x00, x01 = X_t[0], X_t[0]  # This is wrong, let me fix
        x00 = X_t[0][0]
        x01 = X_t[0][1]
        x10 = X_t[1][0]
        x11 = X_t[1][1]
        y00 = Y_t[0][0]
        y01 = Y_t[0][1]
        y10 = Y_t[1][0]
        y11 = Y_t[1][1]
        
        # Define the 7 Strassen sub-terms as (A_coeffs, B_coeffs, C_local_contribs)
        # C_local_contribs: dict mapping (p,q) -> coeff where (X*Y)[p,q] gets this coeff
        
        # S1 = (x00+x11)(y00+y11)
        # A part: x00 + x11, B part: y00 + y11
        # C contrib: (0,0): +1, (1,1): +1
        # S2 = x11(y00+y11)
        # A part: x11, B part: y00 + y11
        # C contrib: (0,0): -1, (1,0): +1
        # S3 = (x00-x11)(y10+y11)
        # A part: x00 - x11, B part: y10 + y11
        # C contrib: (0,1): +1, (1,1): +1
        # S4 = x00(y00-y10)
        # A part: x00, B part: y00 - y10
        # C contrib: (0,0): +1, (1,0): +1
        # S5 = (x00+x01)y11
        # A part: x00 + x01, B part: y11
        # C contrib: (0,1): +1, (1,1): -1
        # S6 = (x10+x11)(y00-y01)
        # A part: x10 + x11, B part: y00 - y01
        # C contrib: (0,0): +1
        # S7 = (x01-x11)(y10+y11)
        # A part: x01 - x11, B part: y10 + y11
        # C contrib: (1,1): +1
        
        sub_terms = [
            # (A_dict, B_dict, {(p,q): coeff})
            (
                {**x00, **{k: x11.get(k, 0) + x00.get(k, 0) for k in x11}},  # x00+x11
                {**y00, **{k: y11.get(k, 0) + y00.get(k, 0) for k in y11}},  # y00+y11
                {(0, 0): 1, (1, 1): 1}
            ),
            (
                dict(x11),
                {**y00, **{k: y11.get(k, 0) + y00.get(k, 0) for k in y11}},
                {(0, 0): -1, (1, 0): 1}
            ),
            (
                {**x00, **{k: -x11.get(k, 0) + x00.get(k, 0) for k in x11}},  # x00-x11
                {**y10, **{k: y11.get(k, 0) + y10.get(k, 0) for k in y11}},  # y10+y11
                {(0, 1): 1, (1, 1): 1}
            ),
            (
                dict(x00),
                {**y00, **{k: -y10.get(k, 0) + y00.get(k, 0) for k in y10}},  # y00-y10
                {(0, 0): 1, (1, 0): 1}
            ),
            (
                {**x00, **{k: x01.get(k, 0) + x00.get(k, 0) for k in x01}},  # x00+x01
                dict(y11),
                {(0, 1): 1, (1, 1): -1}
            ),
            (
                {**x10, **{k: x11.get(k, 0) + x10.get(k, 0) for k in x11}},  # x10+x11
                {**y00, **{k: -y01.get(k, 0) + y00.get(k, 0) for k in y01}},  # y00-y01
                {(0, 0): 1}
            ),
            (
                {**x01, **{k: -x11.get(k, 0) + x01.get(k, 0) for k in x11}},  # x01-x11
                {**y10, **{k: y11.get(k, 0) + y10.get(k, 0) for k in y11}},  # y10+y11
                {(1, 1): 1}
            ),
        ]
        
        for s_idx in range(7):
            A_dict, B_dict, C_local = sub_terms[s_idx]
            
            # Clean up zero coefficients
            A_dict = {k: v for k, v in A_dict.items() if abs(v) > 1e-15}
            B_dict = {k: v for k, v in B_dict.items() if abs(v) > 1e-15}
            
            if not A_dict or not B_dict:
                r_idx += 1
                continue
            
            # Add U contributions
            for (i, j), coeff in A_dict.items():
                U[i * 4 + j, r_idx] += coeff
            
            # Add V contributions
            for (j, k), coeff in B_dict.items():
                V[j * 4 + k, r_idx] += coeff
            
            # Add W contributions
            for (p, q), c_coeff in C_local.items():
                for block_idx, block_coeff in c_contrib.items():
                    block_row = block_idx // 2
                    block_col = block_idx % 2
                    gi = 2 * block_row + p
                    gk = 2 * block_col + q
                    W[gk * 4 + gi, r_idx] += block_coeff * c_coeff
            
            r_idx += 1
    
    return U, V, W


def _build_strassen_4x4x8():
    """
    Build a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication
    using nested Strassen.
    
    Strategy: Split B into B_left (4x4) and B_right (4x4).
    Compute A*B_left with rank-49 Strassen, A*B_right with rank-49 Strassen.
    Total rank = 98.
    
    Returns U, V, W as numpy arrays of shape (16, 98), (32, 98), (32, 98) respectively.
    Indexing: U[i*4+j, r], V[j*8+k, r], W[k*4+i, r]
    """
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Get the 4x4*4x4 decomposition
    U4, V4, W4 = _build_strassen_4x4x4()
    
    # For B_left (columns 0-3 of B):
    # V4[j*4+k, r] corresponds to B[j,k] for k in 0..3
    # In the 4x4*4x8 problem, B[j,k] for k in 0..3 is at V[j*8+k, r]
    for r in range(49):
        for j in range(4):
            for k in range(4):
                V[j * 8 + k, r] = V4[j * 4 + k, r]
    
    # For B_right (columns 4-7 of B):
    # We use the same A-side and C-side structure, but B entries are B[j, k+4]
    # The C output is the same positions (C[i, k] for k in 0..3), but we need
    # to map B[j, k] -> B[j, k+4]
    # Actually, A*B_right gives C_right which is C[:, 4:8]
    # Wait, no. A is 4x4, B is 4x8. C = A*B is 4x8.
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # For k in 0..3: C[i,k] uses B[j,k]
    # For k in 4..7: C[i,k] uses B[j,k]
    # So A*B_left gives C[:, 0:4] and A*B_right gives C[:, 4:8]
    
    # For the right half, we use the same Strassen structure but with B entries
    # shifted: B[j, k] -> B[j, k+4]
    # The C output for the right half is C[:, 4:8], so C[i, k] for k in 4..7
    # In W indexing: W[k*4+i, r] for k in 4..7
    
    for r in range(49):
        for j in range(4):
            for k in range(4):
                # B[j, k] in the 4x4*4x4 corresponds to B[j, k+4] in 4x4*4x8
                V[j * 8 + (k + 4), 49 + r] = V4[j * 4 + k, r]
        
        # W for the right half: C[i, k+4] instead of C[i, k]
        for i in range(4):
            for k in range(4):
                W[(k + 4) * 4 + i, 49 + r] = W4[k * 4 + i, r]
    
    # U is the same for both halves (A is the same)
    U[:, :49] = U4
    U[:, 49:] = U4
    
    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=8):
    """
    Verify that the decomposition is correct.
    For each (i, j, k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1.
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    errors = 0
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(s - 1)
                if err > max_err:
                    max_err = err
                if err > 1e-9:
                    errors += 1
    
    return errors, max_err


def construct():
    """
    Construct a rank-98 exact tensor decomposition for 4x4 * 4x8 matrix multiplication
    using nested Strassen decomposition.
    
    Falls back to schoolbook (rank 128) if verification fails.
    """
    try:
        U, V, W = _build_strassen_4x4x8()
        
        # Verify correctness
        errors, max_err = _verify_decomposition(U, V, W)
        
        if errors == 0:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    except Exception:
        # Final fallback to schoolbook
        n, m, p = DIMENSIONS
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
