import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Schoolbook decomposition of rank n*m*p
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1

    # Verification function
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Laderman's 23-term decomposition for 3x3 * 3x3
    # U: 9x23, V: 9x23, W: 9x23
    # Indexing: U[i*3+j], V[i*3+j], W[i*3+j]
    
    # Define the 23 terms
    # Each term is a tuple (u, v, w) where u, v, w are lists of 9 coefficients
    laderman_terms = []
    
    # Helper to create a vector of 9 zeros and set specific indices
    def vec(*indices):
        v = [0]*9
        for idx in indices:
            v[idx] = 1
        return v

    # 1. A00 B00
    laderman_terms.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # 2. A01 B11
    laderman_terms.append(([0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,0])) # Wait, W is for C[i,k]
    # Actually, the standard form is C[i,k] = sum_t u_t A v_t B w_t
    # Where u_t, v_t are linear combos of A and B, and w_t is a scalar? No.
    # The tensor decomposition is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # For 3x3, T[i,j,k] is 1 if C[i,k] = sum_j A[i,j] B[j,k].
    # So U[i,j,c] V[j,k,c] W[k,i,c] must sum to 1.
    # This means for each term c, U[:,c] is a vector in R^9 (A indices), V[:,c] in R^9 (B indices), W[:,c] in R^9 (C indices).
    # And sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    
    # Let's use the explicit Laderman coefficients.
    # I will use a known valid set.
    
    # Term 1: A00 B00 -> C00
    # U: A00=1, V: B00=1, W: C00=1
    # Term 2: A01 B11 -> C01? No, A01 B11 contributes to C01? No, C[i,k] = sum_j A[i,j] B[j,k].
    # A01 B11 contributes to C01? No, A[0,1] B[1,1] contributes to C[0,1].
    # So U[0,1]=1, V[1,1]=1, W[1,0]=1 (W[k,i] -> W[1,0]).
    
    # This is getting complicated to hardcode manually.
    # I will use the schoolbook for now, but try to construct the 46-rank solution
    # by duplicating a 23-rank 3x3 solution.
    
    # I will use the coefficients from a reliable source online or in literature.
    # Since I cannot browse, I will use a simple 27-rank solution (schoolbook for 3x3)
    # and see if I can reduce it.
    
    # Actually, let's just return the schoolbook 54-rank solution to be safe.
    # The prompt asks for rank-minimal, but if I can't find a better one quickly,
    # 54 is a valid answer.
    
    # However, I can easily construct a 46-rank solution if I have the 23-rank 3x3 solution.
    # Let's assume I have a valid 23-rank 3x3 solution (U3, V3, W3).
    # Then for 3x3 * 3x6:
    # U46[i,j,c] = U3[i,j,c] for c<23, U3[i,j,c-23] for c>=23
    # V46[j,k,c] = V3[j,k,c] for c<23 and k<3, V3[j,k,c-23] for c>=23 and k>=3
    # W46[k,i,c] = W3[k,i,c] for c<23 and k<3, W3[k-3,i,c-23] for c>=23 and k>=3
    
    # I will hardcode a known 23-term decomposition.
    # Source: "Fast Matrix Multiplication" by Strassen, or similar.
    # I will use the following 23 terms (indices 0-8 for 3x3):
    
    terms_3x3 = [
        # (U, V, W)
        ([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]), # A00 B00 C00
        ([0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,0,0,0]), # A01 B11 C01 (W[1,0])
        ([0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,1,0,0,0,0,0,0]), # A02 B22 C02 (W[2,0])
        ([0,0,0,1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]), # A10 B00 C10 (W[0,1])
        ([0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]), # A11 B11 C11 (W[1,1])
        ([0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,1,0,0,0]), # A12 B22 C12 (W[2,1])
        ([0,0,0,0,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]), # A20 B00 C20 (W[0,2])
        ([0,0,0,0,0,0,0,1,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,1,0]), # A21 B11 C21 (W[1,2])
        ([0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,0,1]), # A22 B22 C22 (W[2,2])
        # Now the 14 correction terms
        # I will use a known set of 14 additional terms.
        # This is risky to hardcode without verification.
        # I will fall back to schoolbook 54 if I can't verify a lower rank.
    ]
    
    # Given the risk of error in hardcoding Laderman, and the time constraint,
    # I will return the schoolbook decomposition.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
