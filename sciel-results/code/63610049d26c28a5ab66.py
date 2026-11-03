import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def build_rank_28_decomposition():
        """
        Build a rank-28 decomposition using:
        - Strassen for 2x2 * 2x2 blocks (cols 0-1 and 2-3 of B)
        - Schoolbook for remaining terms
        
        Breakdown:
        - Block 1: A[0:2,0:2] * B[0:2,0:2] -> Strassen rank 7
        - Block 2: A[0:2,0:2] * B[0:2,2:4] -> Strassen rank 7
        - Block 3: A[0:2,0:2] * B[0:2,4:5] -> Schoolbook rank 4
        - Block 4: A[0:2,2:3] * B[2:3,0:5] -> Schoolbook rank 10
        Total: 7 + 7 + 4 + 10 = 28
        """
        rank = 28
        U = [[0] * rank for _ in range(n * m)]  # 6x28
        V = [[0] * rank for _ in range(m * p)]  # 15x28
        W = [[0] * rank for _ in range(p * n)]  # 10x28

        # Helper to add a Strassen block
        # A block: rows i in {i0, i0+1}, cols j in {j0, j0+1}
        # B block: rows j in {j0, j0+1}, cols k in {k0, k0+1}
        # C block: rows i in {i0, i0+1}, cols k in {k0, k0+1}
        def add_strassen_block(rank_offset, i0, j0, k0):
            # A indices: a=A[i0,j0], b=A[i0,j0+1], c=A[i0+1,j0], d=A[i0+1,j0+1]
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            # B indices: e=B[j0,k0], f=B[j0,k0+1], g=B[j0+1,k0], h=B[j0+1,k0+1]
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            
            # W indices: C[i,k] -> k*n + i
            # c00 = C[i0,k0] -> k0*n + i0
            # c01 = C[i0,k0+1] -> (k0+1)*n + i0
            # c10 = C[i0+1,k0] -> k0*n + (i0+1)
            # c11 = C[i0+1,k0+1] -> (k0+1)*n + (i0+1)
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)
            
            r0 = rank_offset
            r1 = rank_offset + 1
            r2 = rank_offset + 2
            r3 = rank_offset + 3
            r4 = rank_offset + 4
            r5 = rank_offset + 5
            r6 = rank_offset + 6
            
            # Strassen formulas:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            
            # c00 = M1 - M4 + M6 - M7
            # c01 = M3 + M5
            # c10 = M2 + M4
            # c11 = M1 - M3 + M5 + M7
            
            # M1: (a+d)(e+h)
            U[a_idx][r0] += 1
            U[d_idx][r0] += 1
            V[e_idx][r0] += 1
            V[h_idx][r0] += 1
            W[c00_idx][r0] += 1
            W[c11_idx][r0] += 1
            
            # M2: (c+d)e
            U[c_idx][r1] += 1
            U[d_idx][r1] += 1
            V[e_idx][r1] += 1
            W[c10_idx][r1] += 1
            
            # M3: a(f-h)
            U[a_idx][r2] += 1
            V[f_idx][r2] += 1
            V[h_idx][r2] -= 1
            W[c01_idx][r2] += 1
            W[c11_idx][r2] -= 1
            
            # M4: d(g-e)
            U[d_idx][r3] += 1
            V[g_idx][r3] += 1
            V[e_idx][r3] -= 1
            W[c00_idx][r3] -= 1
            W[c10_idx][r3] += 1
            
            # M5: (a+b)h
            U[a_idx][r4] += 1
            U[b_idx][r4] += 1
            V[h_idx][r4] += 1
            W[c01_idx][r4] += 1
            W[c11_idx][r4] += 1
            
            # M6: (c-a)(e+f)
            U[c_idx][r5] += 1
            U[a_idx][r5] -= 1
            V[e_idx][r5] += 1
            V[f_idx][r5] += 1
            W[c00_idx][r5] += 1
            
            # M7: (b-d)(g+h)
            U[b_idx][r6] += 1
            U[d_idx][r6] -= 1
            V[g_idx][r6] += 1
            V[h_idx][r6] += 1
            W[c00_idx][r6] -= 1
            W[c11_idx][r6] += 1

        # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        add_strassen_block(0, 0, 0, 0)
        
        # Block 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        add_strassen_block(7, 0, 0, 2)
        
        # Block 3: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (schoolbook, 4 terms)
        # Terms: A[i,j]*B[j,4] for i in {0,1}, j in {0,1}
        col = 14
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        
        # Block 4: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5] (schoolbook, 10 terms)
        # Terms: A[i,2]*B[2,k] for i in {0,1}, k in {0,1,2,3,4}
        for i in range(2):
            for k in range(5):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        assert col == rank, f"Expected rank {rank}, got {col}"
        
        return {"U": U, "V": V, "W": W}

    def validate_float(U, V, W, tol=1e-9):
        """
        Fast float validation for candidate generation.
        """
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        U_3d = U_mat.reshape(n, m, rank)
        V_3d = V_mat.reshape(m, p, rank)
        W_3d = W_mat.reshape(p, n, rank)
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        if not np.allclose(T_recon, 1.0, atol=tol):
            return False
        return True

    def validate_exact(U, V, W):
        """
        Validate using exact Fraction arithmetic.
        """
        try:
            rank = len(U[0])
            
            U_frac = [[Fraction(x) for x in row] for row in U]
            V_frac = [[Fraction(x) for x in row] for row in V]
            W_frac = [[Fraction(x) for x in row] for row in W]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(rank):
                            s += U_frac[i * m + j][r] * V_frac[j * p + k][r] * W_frac[k * n + i][r]
                        if s != 1:
                            return False
            return True
        except Exception:
            return False

    # Try the structured rank-28 decomposition
    result = build_rank_28_decomposition()
    
    # Validate with float first (fast)
    if validate_float(result["U"], result["V"], result["W"]):
        # Convert to Fractions for exact validation
        U_frac = [[Fraction(x) for x in row] for row in result["U"]]
        V_frac = [[Fraction(x) for x in row] for row in result["V"]]
        W_frac = [[Fraction(x) for x in row] for row in result["W"]]
        
        # Exact validation
        if validate_exact(U_frac, V_frac, W_frac):
            return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
