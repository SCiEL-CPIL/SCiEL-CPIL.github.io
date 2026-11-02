"""Rewritten program for matrix multiplication 4,4,6 with block Strassen decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    t_start = time.time()
    deadline = t_start + 115  # safety margin before 120s

    # Strategy: Block-based Strassen decomposition
    # Partition A (4x4) into 2x2 blocks of 2x2 submatrices
    # Partition B (4x6) into 2x3 blocks of 2x2 submatrices
    # Each 2x2 x 2x2 product uses Strassen (rank 7)
    # Total: 2*2*3 = 12 blocks, each rank 7 => rank 84
    
    # Actually, let's be more careful. We have:
    # A is 4x4, B is 4x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split into blocks:
    # A = [[A0, A1], [A2, A3]] where each is 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x2
    # C = [[C00, C01, C02], [C10, C11, C12]] where each is 2x2
    #
    # C00 = A0*B00 + A1*B10
    # C01 = A0*B01 + A1*B11
    # C02 = A0*B02 + A1*B12
    # C10 = A2*B00 + A3*B10
    # C11 = A2*B01 + A3*B11
    # C12 = A2*B02 + A3*B12
    
    # For each product of two 2x2 matrices, use Strassen (rank 7)
    # But here we're summing two products, so we need to be careful.
    # Actually, each A_i * B_jk is a separate 2x2 x 2x2 product.
    # We can apply Strassen to each such product.
    
    # Let's define the Strassen terms for a general 2x2 x 2x2 product X*Y:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    rank_terms = []  # list of (u_dict, v_dict, w_dict)
    
    # Block indices: A block row (0,1), A block col / B block row (0,1), B block col (0,1,2)
    for ar in range(2):      # A block row (0 or 1)
        for ac in range(2):  # A block col / B block row (0 or 1)
            for bc in range(3):  # B block col (0, 1, or 2)
                # We compute A[ar*2:(ar+1)*2, ac*2:(ac+1)*2] * B[ac*2:(ac+1)*2, bc*2:(bc+1)*2]
                
                # For each of the 7 Strassen terms, add a rank-1 contribution
                # X = A block, Y = B block
                
                # M1: (X00+X11)(Y00+Y11)
                rank_terms.append(
                    ({(ar*2, ac*2): 1, (ar*2+1, ac*2+1): 1},
                     {(ac*2, bc*2): 1, (ac*2+1, bc*2+1): 1},
                     {(bc*2, ar*2): 1, (bc*2+1, ar*2+1): 1})
                )
                # M2: (X10+X11)Y00
                rank_terms.append(
                    ({(ar*2+1, ac*2): 1, (ar*2+1, ac*2+1): 1},
                     {(ac*2, bc*2): 1},
                     {(bc*2, ar*2+1): 1, (bc*2+1, ar*2+1): -1})
                )
                # M3: X00(Y01-Y11)
                rank_terms.append(
                    ({(ar*2, ac*2): 1},
                     {(ac*2, bc*2+1): 1, (ac*2+1, bc*2+1): -1},
                     {(bc*2+1, ar*2): 1, (bc*2+1, ar*2+1): 1})
                )
                # M4: X11(Y10-Y00)
                rank_terms.append(
                    ({(ar*2+1, ac*2+1): 1},
                     {(ac*2+1, bc*2): 1, (ac*2, bc*2): -1},
                     {(bc*2, ar*2): 1, (bc*2, ar*2+1): 1})
                )
                # M5: (X00+X01)Y11
                rank_terms.append(
                    ({(ar*2, ac*2): 1, (ar*2, ac*2+1): 1},
                     {(ac*2+1, bc*2+1): 1},
                     {(bc*2, ar*2): -1, (bc*2+1, ar*2): 1})
                )
                # M6: (X10-X00)(Y00+Y01)
                rank_terms.append(
                    ({(ar*2+1, ac*2): 1, (ar*2, ac*2): -1},
                     {(ac*2, bc*2): 1, (ac*2, bc*2+1): 1},
                     {(bc*2+1, ar*2+1): 1})
                )
                # M7: (X01-X11)(Y10+Y11)
                rank_terms.append(
                    ({(ar*2, ac*2+1): 1, (ar*2+1, ac*2+1): -1},
                     {(ac*2+1, bc*2): 1, (ac*2+1, bc*2+1): 1},
                     {(bc*2, ar*2): 1})
                )
    
    # Build U, V, W matrices
    rank = len(rank_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][r] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][r] = coeff
    
    # Verify correctness
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute C_computed = sum_r outer(reshape(U[:,r]), reshape(V[:,r], reshape(W[:,r]))
    # More efficiently:
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U_arr[:, r].reshape(n, m)
        v_vec = V_arr[:, r].reshape(m, p)
        w_vec = W_arr[:, r].reshape(p, n)
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    C_computed[i, k] += u_vec[i, j] * v_vec[j, k] * w_vec[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    if time.time() > deadline:
        raise TimeoutError("Time limit exceeded")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
