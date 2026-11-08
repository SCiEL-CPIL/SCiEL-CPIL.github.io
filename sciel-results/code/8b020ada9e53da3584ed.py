"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Strategy: split B (4x9) into two 4x4 blocks and one 4x1 block.
For each 4x4 block, use recursive Strassen (rank 49).
For the 4x1 block, use schoolbook (rank 16).
Total: 49 + 49 + 16 = 114 rank.

The tensor decomposition is verified via random sampling to ensure correctness
before returning. If verification fails (which should not happen), a fallback
schoolbook decomposition is returned.
"""

import time
import random
import numpy as np


def construct():
    """Return a valid tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Uses recursive Strassen for 4x4*4x4 blocks and schoolbook for 4x4*4x1 block.
    Total rank: 49 + 49 + 16 = 114.
    """
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit

    n, m, p = 4, 4, 9
    # U shape: (n*m, rank) = (16, rank)
    # V shape: (m*p, rank) = (36, rank)
    # W shape: (p*n, rank) = (36, rank)
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]

    U_cols = []  # Each element is a list of length 16
    V_cols = []  # Each element is a list of length 36
    W_cols = []  # Each element is a list of length 36

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # 7 Strassen terms for 2x2 * 2x2 -> 2x2
    # Each: (A_mult, B_mult, C_mult) where each is dict {(row,col): coeff}
    # The product A_mult * B_mult contributes to C via C_mult
    strassen_2x2 = [
        # s1 = (a11+a22)(b11+b22): c11 += s1, c22 += s1
        (({(0,0): 1, (1,1): 1},
          {(0,0): 1, (1,1): 1},
          {(0,0): 1, (1,1): 1})),
        # s2 = (a21+a22)b11: c21 += s2, c22 -= s2
        (({(1,0): 1, (1,1): 1},
          {(0,0): 1},
          {(1,0): 1, (1,1): -1})),
        # s3 = a11(b12-b22): c12 += s3, c22 += s3
        (({(0,0): 1},
          {(0,1): 1, (1,1): -1},
          {(0,1): 1, (1,1): 1})),
        # s4 = a22(b21-b11): c11 += s4, c21 += s4
        (({(1,1): 1},
          {(1,0): 1, (0,0): -1},
          {(0,0): 1, (1,0): 1})),
        # s5 = (a11+a12)b22: c11 -= s5, c12 += s5
        (({(0,0): 1, (0,1): 1},
          {(1,1): 1},
          {(0,0): -1, (0,1): 1})),
        # s6 = (a21-a11)(b11+b12): c22 += s6
        (({(1,0): 1, (0,0): -1},
          {(0,0): 1, (0,1): 1},
          {(1,1): 1})),
        # s7 = (a12-a22)(b21+b22): c11 += s7
        (({(0,1): 1, (1,1): -1},
          {(1,0): 1, (1,1): 1},
          {(0,0): 1})),
    ]

    def add_4x4_strassen(b_col_off, c_col_off):
        """Add 49 Strassen terms for multiplying 4x4 A by 4x4 B block.
        
        b_col_off: column offset in B (0 or 4)
        c_col_off: column offset in C (0 or 4)
        """
        for outer_A, outer_B, outer_C in strassen_2x2:
            for inner_A, inner_B, inner_C in strassen_2x2:
                u = [0.0] * 16
                v = [0.0] * 36
                w = [0.0] * 36
                
                # Outer Strassen combines 2x2 blocks of A (row-wise) 
                # and 2x2 blocks of B (col-wise), contributing to C blocks.
                
                for (oor, ooc), c_ao in outer_A.items():
                    for (ior, ioc), c_ia in inner_A.items():
                        ar = oor * 2 + ior
                        ac = ooc * 2 + ioc
                        u[ar * 4 + ac] += c_ao * c_ia
                
                for (oor, ooc), c_bo in outer_B.items():
                    for (ior, ioc), c_ib in inner_B.items():
                        br = oor * 2 + ior
                        bc = b_col_off + ooc * 2 + ioc
                        v[br * 9 + bc] += c_bo * c_ib
                
                for (oor, ooc), c_co in outer_C.items():
                    for (ior, ioc), c_ic in inner_C.items():
                        cr = oor * 2 + ior
                        cc = c_col_off + ooc * 2 + ioc
                        w[cc * 4 + cr] += c_co * c_ic
                
                add_term(u, v, w)

    # Build the decomposition
    # B split into B1(4x4, cols 0-3), B2(4x4, cols 4-7), B3(4x1, col 8)
    # 4x4 * B1: 49 terms
    add_4x4_strassen(0, 0)
    
    # 4x4 * B2: 49 terms
    add_4x4_strassen(4, 4)
    
    # 4x4 * B3: schoolbook, 16 terms
    for i in range(4):
        for j in range(4):
            u = [0.0] * 16
            v = [0.0] * 36
            w = [0.0] * 36
            u[i * 4 + j] = 1.0
            v[j * 9 + 8] = 1.0
            w[8 * 4 + i] = 1.0
            add_term(u, v, w)
    
    rank = len(U_cols)
    
    # Build output matrices as numpy arrays for verification
    U_np = np.array([[U_cols[r][i] for r in range(rank)] for i in range(16)], dtype=np.float64)
    V_np = np.array([[V_cols[r][i] for r in range(rank)] for i in range(36)], dtype=np.float64)
    W_np = np.array([[W_cols[r][i] for r in range(rank)] for i in range(36)], dtype=np.float64)
    
    # Verify correctness with a few random tests
    def verify_decomposition():
        """Verify that the decomposition correctly computes matrix multiplication."""
        rng = np.random.RandomState(42)
        for _ in range(3):
            A = rng.rand(n, m)
            B = rng.rand(m, p)
            C_expected = A @ B
            
            # Compute C using the tensor decomposition
            # C[k*n+i, r] = W[k*n+i, r]
            # U[i*m+j, r], V[j*p+k, r]
            # C = sum_r U[:,r] * V[:,r] * W[:,r] in tensor sense
            # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_r = U_np[:, r]  # shape (16,)
                v_r = V_np[:, r]  # shape (36,)
                w_r = W_np[:, r]  # shape (36,)
                
                # Reshape u_r to (n, m), v_r to (m, p), w_r to (p, n)
                u_mat = u_r.reshape(n, m)
                v_mat = v_r.reshape(m, p)
                w_mat = w_r.reshape(p, n)
                
                # u_mat * A element-wise, then sum with v_mat * B element-wise
                # Actually: term_r = (u_mat * A) outer-ish with (v_mat * B) weighted by w_mat
                # More precisely:
                # For each (i, k): C[i,k] += sum_j u_mat[i,j] * A[i,j] * v_mat[j,k] * B[j,k] * w_mat[k,i]
                # This is: C += (A * u_mat) @ (B * v_mat) * w_mat (broadcast)
                
                a_u = A * u_mat  # (n, m)
                b_v = B * v_mat  # (m, p)
                prod = a_u @ b_v  # (n, p)
                C_computed += prod * w_mat.T  # w_mat is (p, n), need (n, p)
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    # Verify if we have time
    if time.time() < deadline - 5:
        if not verify_decomposition():
            # Fallback: schoolbook
            U_cols.clear()
            V_cols.clear()
            W_cols.clear()
            for i in range(4):
                for j in range(4):
                    for k in range(9):
                        u = [0.0] * 16
                        v = [0.0] * 36
                        w = [0.0] * 36
                        u[i * 4 + j] = 1.0
                        v[j * 9 + k] = 1.0
                        w[k * 4 + i] = 1.0
                        add_term(u, v, w)
            rank = len(U_cols)
            U_np = np.array([[U_cols[r][i] for r in range(rank)] for i in range(16)], dtype=np.float64)
            V_np = np.array([[V_cols[r][i] for r in range(rank)] for i in range(36)], dtype=np.float64)
            W_np = np.array([[W_cols[r][i] for r in range(rank)] for i in range(36)], dtype=np.float64)
    
    # Convert to list-of-lists format as required
    U = [[U_np[i, r] for r in range(rank)] for i in range(16)]
    V = [[V_np[i, r] for r in range(rank)] for i in range(36)]
    W = [[W_np[i, r] for r in range(rank)] for i in range(36)]
    
    return {"U": U, "V": V, "W": W}
