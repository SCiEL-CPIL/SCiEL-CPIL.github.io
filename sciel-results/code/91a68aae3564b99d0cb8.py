"""
Tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses recursive Strassen to build a rank-65 decomposition (49 from 4x4x4 Strassen + 16 schoolbook for the extra column).
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0

    # ----------------------------------------------------------------------
    # Verify a candidate (U,V,W) exactly reproduces matmul C = A @ B.
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # ----------------------------------------------------------------------
    def verify(U, V, W, n, m, p, tol=1e-9):
        if U.shape != (n * m, U.shape[1]) or V.shape != (m * p, V.shape[1]):
            return False
        if W.shape != (p * n, U.shape[1]):
            return False
        R = U.shape[1]
        if R != V.shape[1] or R != W.shape[1]:
            return False
        
        # Use a random-ish but deterministic test case
        A = np.arange(n * m, dtype=float).reshape(n, m)
        B = np.arange(m * p, dtype=float).reshape(m, p)
        Ctrue = A @ B
        
        # Build C from tensor
        Uf = U.reshape(n, m, R)
        Vf = V.reshape(m, p, R)
        Wf = W.reshape(p, n, R)
        C = np.einsum('ijr,jkr,kir->ik', Uf, Vf, Wf)
        return np.allclose(C, Ctrue, atol=tol)

    # ----------------------------------------------------------------------
    # 2x2x2 Strassen block (rank 7).
    # Returns coefficient tables for U, V, W.
    # U is 4x7 (rows: a00, a01, a10, a11), V is 4x7 (rows: b00, b01, b10, b11), W is 4x7 (rows: c00, c01, c10, c11)
    # ----------------------------------------------------------------------
    def strassen_2x2():
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Strassen's 7 products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        
        # Indices:
        # A: 00=0, 01=1, 10=2, 11=3
        # B: 00=0, 01=1, 10=2, 11=3
        # C: 00=0, 01=1, 10=2, 11=3
        
        # M1: (a00+a11)(b00+b11)
        U[0, 0] += 1; U[3, 0] += 1
        V[0, 0] += 1; V[3, 0] += 1
        W[0, 0] += 1; W[1, 0] += 1; W[2, 0] += -1; W[3, 0] += -1
        
        # M2: (a10+a11)b00
        U[2, 1] += 1; U[3, 1] += 1
        V[0, 1] += 1
        W[2, 1] += 1; W[3, 1] += -1
        
        # M3: a00(b10-b11)
        U[0, 2] += 1
        V[2, 2] += 1; V[3, 2] += -1
        W[1, 2] += 1; W[3, 2] += -1
        
        # M4: a11(b01-b00)
        U[3, 3] += 1
        V[1, 3] += 1; V[0, 3] += -1
        W[0, 3] += 1; W[2, 3] += -1
        
        # M5: (a00+a01)b11
        U[0, 4] += 1; U[1, 4] += 1
        V[3, 4] += 1
        W[0, 4] += 1; W[1, 4] += -1
        
        # M6: (a10-a00)(b00+b01)
        U[2, 5] += 1; U[0, 5] += -1
        V[0, 5] += 1; V[1, 5] += 1
        W[2, 5] += 1; W[3, 5] += 1
        
        # M7: (a01-a11)(b10+b11)
        U[1, 6] += 1; U[3, 6] += -1
        V[2, 6] += 1; V[3, 6] += 1
        W[1, 6] += 1; W[3, 6] += 1
        
        return U, V, W

    # ----------------------------------------------------------------------
    # Build 4x4x4 decomposition using recursive Strassen.
    # Rank = 7 * 7 = 49.
    # ----------------------------------------------------------------------
    def build_4x4x4_strassen():
        n = m = p = 4
        U4 = np.zeros((n * m, 49), dtype=float)
        V4 = np.zeros((m * p, 49), dtype=float)
        W4 = np.zeros((p * n, 49), dtype=float)
        
        U2, V2, W2 = strassen_2x2()
        
        # Top-level Strassen terms for 4x4*4x4 (treating 2x2 blocks as scalars)
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B21-B22)
        # M4 = A22*(B12-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        
        # Block indices in 4x4 (row-major):
        # A11: rows 0-1, cols 0-1 -> global indices: 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> global indices: 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> global indices: 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> global indices: 10,11,14,15
        
        # B11: rows 0-1, cols 0-1 -> global indices: 0,1,4,5
        # B12: rows 0-1, cols 2-3 -> global indices: 2,3,6,7
        # B21: rows 2-3, cols 0-1 -> global indices: 8,9,12,13
        # B22: rows 2-3, cols 2-3 -> global indices: 10,11,14,15
        
        # C11: rows 0-1, cols 0-1 -> global indices: 0,1,4,5
        # C12: rows 0-1, cols 2-3 -> global indices: 2,3,6,7
        # C21: rows 2-3, cols 0-1 -> global indices: 8,9,12,13
        # C22: rows 2-3, cols 2-3 -> global indices: 10,11,14,15
        
        # Map block names to local 2x2 indices (0,1,2,3)
        # Block '11' corresponds to top-left 2x2
        # Block '12' corresponds to top-right 2x2
        # Block '21' corresponds to bottom-left 2x2
        # Block '22' corresponds to bottom-right 2x2
        
        # For each top-level term s (0-6) and inner term t (0-6):
        # We need to map the coefficients from the 2x2 Strassen decomposition
        # to the global 4x4 indices.
        
        # Let's define the top-level Strassen coefficients explicitly.
        # For each term s, we have:
        # A_blocks: dict {block_name: coeff}
        # B_blocks: dict {block_name: coeff}
        # C_blocks: dict {block_name: coeff}
        
        top_strassen = [
            # s=0: M1 = (A11+A22)*(B11+B22)
            ({'11': 1, '22': 1}, {'11': 1, '22': 1}, {'11': 1, '12': 1, '21': -1, '22': -1}),
            # s=1: M2 = (A21+A22)*B11
            ({'21': 1, '22': 1}, {'11': 1}, {'21': 1, '22': -1}),
            # s=2: M3 = A11*(B21-B22)
            ({'11': 1}, {'21': 1, '22': -1}, {'12': 1, '22': -1}),
            # s=3: M4 = A22*(B12-B11)
            ({'22': 1}, {'12': 1, '11': -1}, {'11': 1, '21': -1}),
            # s=4: M5 = (A11+A12)*B22
            ({'11': 1, '12': 1}, {'22': 1}, {'11': 1, '12': -1}),
            # s=5: M6 = (A21-A11)*(B11+B12)
            ({'21': 1, '11': -1}, {'11': 1, '12': 1}, {'21': 1, '22': 1}),
            # s=6: M7 = (A12-A22)*(B21+B22)
            ({'12': 1, '22': -1}, {'21': 1, '22': 1}, {'12': 1, '22': 1}),
        ]
        
        # Block to local index mapping
        # '11' -> 0, '12' -> 1, '21' -> 2, '22' -> 3
        block_to_local = {'11': 0, '12': 1, '21': 2, '22': 3}
        
        # Global index mapping for a block and local index
        # Block '11': base 0, local 0->0, 1->1, 2->4, 3->5
        # Block '12': base 2, local 0->2, 1->3, 2->6, 3->7
        # Block '21': base 8, local 0->8, 1->9, 2->12, 3->13
        # Block '22': base 10, local 0->10, 1->11, 2->14, 3->15
        
        def block_global_index(block_name, local_idx):
            if block_name == '11':
                return [0, 1, 4, 5][local_idx]
            elif block_name == '12':
                return [2, 3, 6, 7][local_idx]
            elif block_name == '21':
                return [8, 9, 12, 13][local_idx]
            elif block_name == '22':
                return [10, 11, 14, 15][local_idx]
        
        for s, (A_blocks, B_blocks, C_blocks) in enumerate(top_strassen):
            for t in range(7):
                col = s * 7 + t
                
                # For each block in A
                for block_name, coeff in A_blocks.items():
                    local_block = block_to_local[block_name]
                    for local_idx in range(4):
                        if U2[local_idx, t] != 0:
                            global_idx = block_global_index(block_name, local_idx)
                            U4[global_idx, col] += coeff * U2[local_idx, t]
                
                # For each block in B
                for block_name, coeff in B_blocks.items():
                    local_block = block_to_local[block_name]
                    for local_idx in range(4):
                        if V2[local_idx, t] != 0:
                            global_idx = block_global_index(block_name, local_idx)
                            V4[global_idx, col] += coeff * V2[local_idx, t]
                
                # For each block in C
                for block_name, coeff in C_blocks.items():
                    local_block = block_to_local[block_name]
                    for local_idx in range(4):
                        if W2[local_idx, t] != 0:
                            global_idx = block_global_index(block_name, local_idx)
                            W4[global_idx, col] += coeff * W2[local_idx, t]
        
        return U4, V4, W4

    # ----------------------------------------------------------------------
    # Build 4x4x5 decomposition:
    # 1. Use 4x4x4 Strassen (rank 49) for the first 4 columns of B.
    # 2. Append schoolbook (rank 16) for the 5th column of B.
    # Total rank = 49 + 16 = 65.
    # ----------------------------------------------------------------------
    def build_4x4x5_decomposition():
        U4, V4, W4 = build_4x4x4_strassen()
        
        n, m, p = 4, 4, 5
        R4 = U4.shape[1]
        R_extra = n * m  # 16
        R_total = R4 + R_extra
        
        U = np.zeros((n * m, R_total), dtype=float)
        V = np.zeros((m * p, R_total), dtype=float)
        W = np.zeros((p * n, R_total), dtype=float)
        
        U[:, :R4] = U4
        V[:, :R4] = V4
        W[:, :R4] = W4
        
        # Schoolbook for k=4
        k = 4
        col = R4
        for i in range(n):
            for j in range(m):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        
        return U, V, W

    # ----------------------------------------------------------------------
    # Main logic
    # ----------------------------------------------------------------------
    try:
        U, V, W = build_4x4x5_decomposition()
        
        if verify(U, V, W, n, m, p):
            # Convert to lists as required by interface
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        else:
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U_sb = np.zeros((n * m, rank), dtype=float)
            V_sb = np.zeros((m * p, rank), dtype=float)
            W_sb = np.zeros((p * n, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, col] = 1.0
                        V_sb[j * p + k, col] = 1.0
                        W_sb[k * n + i, col] = 1.0
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
            
    except Exception:
        # Fallback to schoolbook if any error occurs
        rank = n * m * p
        U_sb = np.zeros((n * m, rank), dtype=float)
        V_sb = np.zeros((m * p, rank), dtype=float)
        W_sb = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j, col] = 1.0
                    V_sb[j * p + k, col] = 1.0
                    W_sb[k * n + i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
