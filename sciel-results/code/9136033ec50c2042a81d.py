import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    deadline = time.time() + 110
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Use 3 blocks of 3 columns, each using a rank-13 2x3 x 3x3 algorithm
    # Laderman-style decomposition for 2x3 x 3x3
    for block in range(3):
        if time.time() > deadline:
            break
        k0 = block * 3
        k1 = block * 3 + 1
        k2 = block * 3 + 2
        
        # 13 terms for 2x3 x 3x3 multiplication
        # Term 1: (A00+A11)*(B00+B11) -> C00 + C11
        add_term(
            {0: 1, 4: 1},
            {k0: 1, p+k1: 1},
            {k0*n: 1, k1*n+1: 1}
        )
        # Term 2: (A10+A11)*B00 -> C10 - C11
        add_term(
            {3: 1, 4: 1},
            {k0: 1},
            {k0*n+1: 1, k1*n+1: -1}
        )
        # Term 3: A00*(B01-B11) -> C01 + C11
        add_term(
            {0: 1},
            {k1: 1, p+k1: -1},
            {k1*n: 1, k1*n+1: 1}
        )
        # Term 4: A11*(B10-B00) -> C00 + C10
        add_term(
            {4: 1},
            {p+k0: 1, k0: -1},
            {k0*n: 1, k0*n+1: 1}
        )
        # Term 5: (A00+A01)*B11 -> -C00 + C01
        add_term(
            {0: 1, 1: 1},
            {p+k1: 1},
            {k0*n: -1, k1*n: 1}
        )
        # Term 6: (A10-A00)*(B00+B01) -> C11
        add_term(
            {3: 1, 0: -1},
            {k0: 1, k1: 1},
            {k1*n+1: 1}
        )
        # Term 7: (A01-A11)*(B10+B11) -> C00
        add_term(
            {1: 1, 4: -1},
            {p+k0: 1, p+k1: 1},
            {k0*n: 1}
        )
        # Terms 8-13: Handle column k2 (the third column)
        # C[0,k2] = A[0,0]B[0,k2] + A[0,1]B[1,k2] + A[0,2]B[2,k2]
        # C[1,k2] = A[1,0]B[0,k2] + A[1,1]B[1,k2] + A[1,2]B[2,k2]
        
        # Term 8: A[0,0]*B[0,k2] -> C[0,k2]
        add_term(
            {0: 1},
            {k2: 1},
            {k2*n: 1}
        )
        # Term 9: A[0,1]*B[1,k2] -> C[0,k2]
        add_term(
            {1: 1},
            {p+k2: 1},
            {k2*n: 1}
        )
        # Term 10: A[0,2]*B[2,k2] -> C[0,k2]
        add_term(
            {2: 1},
            {2*p+k2: 1},
            {k2*n: 1}
        )
        # Term 11: A[1,0]*B[0,k2] -> C[1,k2]
        add_term(
            {3: 1},
            {k2: 1},
            {k2*n+1: 1}
        )
        # Term 12: A[1,1]*B[1,k2] -> C[1,k2]
        add_term(
            {4: 1},
            {p+k2: 1},
            {k2*n+1: 1}
        )
        # Term 13: A[1,2]*B[2,k2] -> C[1,k2]
        add_term(
            {5: 1},
            {2*p+k2: 1},
            {k2*n+1: 1}
        )
    
    rank = len(U_cols)
    
    # Verify correctness with a random test
    if time.time() < deadline - 5 and rank > 0:
        U_mat = np.array(U_cols).T
        V_mat = np.array(V_cols).T
        W_mat = np.array(W_cols).T
        
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute using decomposition
        C_computed = np.zeros((n, p))
        for r in range(rank):
            a_vec = U_mat[:, r]
            b_vec = V_mat[:, r]
            w_vec = W_mat[:, r]
            # a_vec is (n*m,), b_vec is (m*p,), w_vec is (p*n,)
            a_mat = a_vec.reshape(n, m)
            b_mat = b_vec.reshape(m, p)
            prod = a_mat @ b_mat  # (n, p)
            # w_vec is (p*n,), need to reshape to (n, p) using W[k*n+i, r] convention
            # W index: k*n + i where k is column of C, i is row of C
            w_mat = w_vec.reshape(p, n).T  # (n, p)
            C_computed += prod * w_mat
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback to schoolbook if verification fails
            U_cols = []
            V_cols = []
            W_cols = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        add_term(
                            {i*m+j: 1},
                            {j*p+k: 1},
                            {k*n+i: 1}
                        )
            rank = len(U_cols)
    
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][j] for r in range(rank)] for j in range(m*p)]
    W = [[W_cols[r][k] for r in range(rank)] for k in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
