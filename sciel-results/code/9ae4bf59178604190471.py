"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Build a rank-42 decomposition for 4x4x6 matrix multiplication
    # Using 3 blocks of rank-14, where each block handles 2 output columns.
    # Each block computes C[:, 2k:2k+2] = A * B[:, 2k:2k+2]
    # Using a structured decomposition with sharing.
    
    # For a 4x4 * 4x2 product, we can use a rank-14 decomposition.
    # This is derived from considering the 4x4 matrix A and 4x2 matrix B.
    # Split A into 2x2 blocks: A00, A01, A10, A11
    # Split B into 2x2 blocks: B00, B10 (top and bottom 2 rows)
    # C = [A00*B00 + A01*B10; A10*B00 + A11*B10]
    
    # Using Strassen-like sharing between the two terms in each row block:
    # For top: A00*B00 + A01*B10 = (A00+A01)*B00 + A01*(B10-B00) ... no, that's not right.
    # Actually: X*Y + Z*W where X,Z are 2x2 (from A) and Y,W are 2x2 (from B)
    # This is a 2x2*2x2 + 2x2*2x2 = 8x8 tensor, rank at least 8.
    # With sharing: (X+Z)(Y+W) - X*W - Z*Y = 7 mults for one product, but we have two products.
    
    # Let me use a different approach: just use the naive 28 mults per block but with optimization.
    # Actually, 4x4 * 4x2 has 32 output elements. The tensor rank is at least 8.
    # A simple construction: for each of the 4 rows of A and each of the 4 columns of B (but B is 4x2),
    # we have 4*4 = 16 products, each contributing to 2 output elements.
    
    # Let me just use a straightforward rank-32 per block (4x4*4x2 naive) but that's 96 total.
    # That's worse than 84.
    
    # Let me stick with the rank-84 solution but verify it more carefully.
    # The previous code had a bug: it allocated rank=84 but the loop might not fill all columns.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for i in range(2):  # A row block index (0 or 1)
        for k in range(3):  # C col block index (0, 1, or 2)
            for j in range(2):  # Sum over j: A_ij * B_jk
                # Block A_ij: rows [2*i, 2*i+1], cols [2*j, 2*j+1]
                # Block B_jk: rows [2*j, 2*j+1], cols [2*k, 2*k+1]
                
                # A_ij elements:
                a_idx = 2*i * m + 2*j
                b_idx = 2*i * m + 2*j + 1
                c_idx = (2*i+1) * m + 2*j
                d_idx = (2*i+1) * m + 2*j + 1
                
                # B_jk elements:
                e_idx = 2*j * p + 2*k
                f_idx = 2*j * p + 2*k + 1
                g_idx = (2*j+1) * p + 2*k
                h_idx = (2*j+1) * p + 2*k + 1
                
                # C_ik block: rows [2*i, 2*i+1], cols [2*k, 2*k+1]
                # W index: W[col*n + row, r]
                c11_idx = (2*k) * n + 2*i
                c12_idx = (2*k+1) * n + 2*i
                c21_idx = (2*k) * n + 2*i + 1
                c22_idx = (2*k+1) * n + 2*i + 1
                
                # Strassen 7 multiplications
                # M1 = (a+d)(e+h)
                if col >= rank:
                    break
                U[a_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M2 = (c+d)e
                if col >= rank:
                    break
                U[c_idx][col] = 1
                U[d_idx][col] = 1
                V[e_idx][col] = 1
                W[c21_idx][col] = 1
                W[c22_idx][col] = -1
                col += 1
                
                # M3 = a(f-h)
                if col >= rank:
                    break
                U[a_idx][col] = 1
                V[f_idx][col] = 1
                V[h_idx][col] = -1
                W[c12_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M4 = d(g-e)
                if col >= rank:
                    break
                U[d_idx][col] = 1
                V[g_idx][col] = 1
                V[e_idx][col] = -1
                W[c11_idx][col] = 1
                W[c21_idx][col] = 1
                col += 1
                
                # M5 = (a+b)h
                if col >= rank:
                    break
                U[a_idx][col] = 1
                U[b_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = -1
                W[c12_idx][col] = 1
                col += 1
                
                # M6 = (c-a)(e+f)
                if col >= rank:
                    break
                U[c_idx][col] = 1
                U[a_idx][col] = -1
                V[e_idx][col] = 1
                V[f_idx][col] = 1
                W[c22_idx][col] = 1
                col += 1
                
                # M7 = (b-d)(g+h)
                if col >= rank:
                    break
                U[b_idx][col] = 1
                U[d_idx][col] = -1
                V[g_idx][col] = 1
                V[h_idx][col] = 1
                W[c11_idx][col] = 1
                col += 1
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
    
    # Compute from decomposition
    T_decomp = np.zeros_like(T_expected)
    for r in range(rank):
        if np.all(U_arr[:,r] == 0) and np.all(V_arr[:,r] == 0) and np.all(W_arr[:,r] == 0):
            continue
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    # Check correctness
    if not np.allclose(T_decomp, T_expected, atol=1e-10):
        # Fall back to naive
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i*m+j][col] = 1
                    V_naive[j*p+k][col] = 1
                    W_naive[k*n+i][col] = 1
                    col += 1
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
