"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use structured bilinear forms
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We decompose into rank-1 terms: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Use a block-based approach with Strassen for 2x2 subblocks
    # A = [[A00, A01], [A10, A11]] each 2x2
    # B = [[B00, B01, B02], [B10, B11, B12]] each 2x2 (since p=6, 3 blocks of 2)
    # C = [[C00, C01, C02], [C10, C11, C12]] each 2x2
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # For each 2x2 product X*Y, use Strassen (rank 7):
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
    
    # For sum of two 2x2 products X*P + Y*Q, we can use:
    # Compute X*P and Y*Q separately with Strassen: 7+7 = 14 rank-1 terms
    # But we can potentially share some structure.
    
    # Let's use 14 rank-1 terms per C block (6 blocks * 14 = 84 total)
    # This is better than schoolbook 96.
    
    # Actually, let me try a more direct approach with fewer terms.
    # Use the fact that we have 6 C blocks, each needing 2 products.
    
    # Let me build explicit rank-1 tensors.
    # Each rank-1 tensor contributes: U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # where U has shape (n*m, 1), V has shape (m*p, 1), W has shape (p*n, 1)
    
    # For a 2x2 product X*Y (where X is 2x2 from A, Y is 2x2 from B):
    # The 7 Strassen terms each give a scalar product that contributes to Z.
    
    # Let me index carefully:
    # A rows: 0,1,2,3 -> block row 0: rows 0,1; block row 1: rows 2,3
    # A cols: 0,1,2,3 -> block col 0: cols 0,1; block col 1: cols 2,3
    # B rows: 0,1,2,3 -> same as A cols
    # B cols: 0,1,2,3,4,5 -> block col 0: cols 0,1; block col 1: cols 2,3; block col 2: cols 4,5
    # C rows: 0,1,2,3 -> same as A rows
    # C cols: 0,1,2,3,4,5 -> same as B cols
    
    # For Strassen on X*Y where X = A[i1:,:j1:], Y = B[j1:,:k1:]:
    # X00 = A[i1*2, j1*2], X01 = A[i1*2, j1*2+1]
    # X10 = A[i1*2+1, j1*2], X11 = A[i1*2+1, j1*2+1]
    # Y00 = B[j1*2, k1*2], Y01 = B[j1*2, k1*2+1]
    # Y10 = B[j1*2+1, k1*2], Y11 = B[j1*2+1, k1*2+1]
    
    # Z00 = C[i1*2, k1*2] = M1 + M4 - M5 + M7
    # Z01 = C[i1*2, k1*2+1] = M3 + M5
    # Z10 = C[i1*2+1, k1*2] = M2 + M4
    # Z11 = C[i1*2+1, k1*2+1] = M1 - M2 + M3 + M6
    
    # Each Ms is a scalar: Ms = (linear form in A) * (linear form in B)
    # In tensor rank terms, each Ms contributes one rank-1 term to each Z element it appears in.
    
    # Let me build all rank-1 terms.
    # Total terms: 6 C blocks * 2 products per block * 7 Strassen terms = 84
    
    # But wait - each Strassen term Ms appears in multiple Z elements with different coefficients.
    # So for each (i1, j1, k1) triple and each Strassen term s, we get one rank-1 tensor
    # that contributes to up to 4 C elements.
    
    # Let me enumerate:
    # For i1 in {0,1}, j1 in {0,1}, k1 in {0,1,2}:
    #   For s in {1,...,7}:
    #     Define the linear forms for A and B
    #     Add rank-1 term with appropriate W coefficients
    
    rank_terms = []  # list of (U_vec, V_vec, W_vec)
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(3):
                # Strassen terms for A[i1 block, j1 block] * B[j1 block, k1 block]
                # M1 = (A[i1*2,j1*2] + A[i1*2+1,j1*2+1]) * (B[j1*2,k1*2] + B[j1*2+1,k1*2+1])
                # M2 = (A[i1*2+1,j1*2] + A[i1*2+1,j1*2+1]) * B[j1*2,k1*2]
                # M3 = A[i1*2,j1*2] * (B[j1*2,k1*2+1] - B[j1*2+1,k1*2+1])
                # M4 = A[i1*2+1,j1*2+1] * (B[j1*2+1,k1*2] - B[j1*2,k1*2])
                # M5 = (A[i1*2,j1*2] + A[i1*2,j1*2+1]) * B[j1*2+1,k1*2+1]
                # M6 = (A[i1*2+1,j1*2] - A[i1*2,j1*2]) * (B[j1*2,k1*2] + B[j1*2,k1*2+1])
                # M7 = (A[i1*2,j1*2+1] - A[i1*2+1,j1*2+1]) * (B[j1*2+1,k1*2] + B[j1*2+1,k1*2+1])
                
                # M1 contributes to: Z00(+1), Z11(+1)
                # M2 contributes to: Z10(+1), Z11(-1)
                # M3 contributes to: Z01(+1), Z11(+1)
                # M4 contributes to: Z00(+1), Z10(+1)
                # M5 contributes to: Z00(-1), Z01(+1)
                # M6 contributes to: Z11(+1)
                # M7 contributes to: Z00(+1)
                
                # Define the 7 terms as (U_coeffs, V_coeffs, W_coeffs)
                # U_coeffs: dict mapping (i,j) -> coeff for A[i,j]
                # V_coeffs: dict mapping (j,k) -> coeff for B[j,k]
                # W_coeffs: dict mapping (i,k) -> coeff for C[i,k]
                
                terms = [
                    # M1: (A[i1*2,j1*2] + A[i1*2+1,j1*2+1]) * (B[j1*2,k1*2] + B[j1*2+1,k1*2+1])
                    ({(i1*2, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2+1): 1}),
                    # M2: (A[i1*2+1,j1*2] + A[i1*2+1,j1*2+1]) * B[j1*2,k1*2]
                    ({(i1*2+1, j1*2): 1, (i1*2+1, j1*2+1): 1},
                     {(j1*2, k1*2): 1},
                     {(i1*2+1, k1*2): 1, (i1*2+1, k1*2+1): -1}),
                    # M3: A[i1*2,j1*2] * (B[j1*2,k1*2+1] - B[j1*2+1,k1*2+1])
                    ({(i1*2, j1*2): 1},
                     {(j1*2, k1*2+1): 1, (j1*2+1, k1*2+1): -1},
                     {(i1*2, k1*2+1): 1, (i1*2+1, k1*2+1): 1}),
                    # M4: A[i1*2+1,j1*2+1] * (B[j1*2+1,k1*2] - B[j1*2,k1*2])
                    ({(i1*2+1, j1*2+1): 1},
                     {(j1*2+1, k1*2): 1, (j1*2, k1*2): -1},
                     {(i1*2, k1*2): 1, (i1*2+1, k1*2): 1}),
                    # M5: (A[i1*2,j1*2] + A[i1*2,j1*2+1]) * B[j1*2+1,k1*2+1]
                    ({(i1*2, j1*2): 1, (i1*2, j1*2+1): 1},
                     {(j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): -1, (i1*2, k1*2+1): 1}),
                    # M6: (A[i1*2+1,j1*2] - A[i1*2,j1*2]) * (B[j1*2,k1*2] + B[j1*2,k1*2+1])
                    ({(i1*2+1, j1*2): 1, (i1*2, j1*2): -1},
                     {(j1*2, k1*2): 1, (j1*2, k1*2+1): 1},
                     {(i1*2+1, k1*2+1): 1}),
                    # M7: (A[i1*2,j1*2+1] - A[i1*2+1,j1*2+1]) * (B[j1*2+1,k1*2] + B[j1*2+1,k1*2+1])
                    ({(i1*2, j1*2+1): 1, (i1*2+1, j1*2+1): -1},
                     {(j1*2+1, k1*2): 1, (j1*2+1, k1*2+1): 1},
                     {(i1*2, k1*2): 1}),
                ]
                
                for u_dict, v_dict, w_dict in terms:
                    rank_terms.append((u_dict, v_dict, w_dict))
    
    # Now build U, V, W matrices
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
    # Compute C from A*B and compare with sum_r U[:,r] * V[:,r] * W[:,r]
    # For random test
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        # U[:,r] has shape (n*m,), V[:,r] has shape (m*p,), W[:,r] has shape (p*n,)
        u_vec = np.array(U)[:, r].reshape(n, m)
        v_vec = np.array(V)[:, r].reshape(m, p)
        w_vec = np.array(W)[:, r].reshape(p, n)
        # Contribution: u_vec * v_vec * w_vec (outer product scaled)
        # C[i,k] += sum_j u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
