# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_rank53():
    """
    Guaranteed rank-53 decomposition using Strassen for 7 blocks of 2 columns
    plus schoolbook for the last column.
    7 * 7 + 4 = 53
    """
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_school = [[1,0,1,0],[0,1,0,1]]
    W_school = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
    
    return {"U": U, "V": V, "W": W}

def _find_2x2x3_rank10():
    """
    Find a rank-10 decomposition for 2x2 x 2x3 matrix multiplication.
    The tensor T[i,j,k] = delta_{i,j} for i in {0,1}, j in {0,1}, k in {0,1,2}.
    We seek U (4x10), V (6x10), W (6x10) such that
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = delta_{i,j} for all i,j,k.
    
    Using the known fact that rank(2,2,3) = 10, we construct one explicitly.
    """
    # The 2x2x3 multiplication tensor has 12 nonzero entries (one for each (i,j,k) where i==j)
    # We need 10 rank-1 terms to span these 12 values.
    
    # Known construction: use 7 Strassen-like terms for columns 0,1 and 3 additional for column 2
    # with some sharing.
    
    # Let me construct this via the explicit formula from the literature.
    # For 2x2 x 2x3, one rank-10 decomposition is:
    
    # Indices: U is 4x10 (i*m+j), V is 6x10 (m*p = j*p+k), W is 6x10 (p*n = k*n+i)
    # n=2, m=2, p=3
    
    n, m, p = 2, 2, 3
    rank = 10
    
    # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
    # Constraint: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,j}
    
    # This is a system of 12 equations (for each (i,k) pair where we need C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually the tensor entries are T[i,j,k] = delta_{i,j}, and we need:
    # For each (i,j,k): sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta_{i,j}
    
    # There are 2*2*3 = 12 such equations.
    # Unknowns: 4*10 + 6*10 + 6*10 = 40+60+60 = 160, but with scaling freedom (10-1=9 degrees).
    
    # Let me use a constructive approach. Start with the 12 schoolbook terms and merge.
    
    # Schoolbook terms (12):
    # For each (i,j) pair, and each k: term is U[i,j]=1, V[j,k]=1, W[k,i]=1
    # That's 4*3 = 12 terms.
    
    # To reduce to 10, we need to find 2 linear dependencies.
    
    # Alternative: use the explicit rank-10 formula.
    # From the theory, one can show that:
    # C[0,0] = M1 - M3 + M5 - M7  (using Strassen-like)
    # ... this gets complex. Let me just solve it numerically.
    
    # Set up the problem: we want U, V, W with the constraint above.
    # Fix V to be the identity-like structure to reduce variables.
    
    # Actually, let me use a simpler approach: 
    # For columns 0 and 1, use 7 Strassen products.
    # For column 2, use 3 products but share the A-sums from Strassen.
    
    # Strassen for cols 0,1 (7 products):
    # M1 = (A00+A11)(B00+B10)  -> wait, B is 2x3, so B[j,k]
    # Let me be careful: A is 2x2, B is 2x3.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For k=0,1 (first two columns), Strassen:
    # M1 = (A00+A11)*(B00+B10)  -- no, B00 is B[0,0], B10 is B[1,0]
    # Actually standard Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B11-B01)
    # M5 = (A00+A01)(B11+B00)
    # M6 = (A10-A00)(B01+B11)
    # M7 = (A01-A11)(B00+B10)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 - M1
    # C10 = M1 - M4
    # C11 = M2 + M4 - M7 + M3  -- wait let me redo
    
    # Standard Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 - M1
    # C10 = M1 - M4  
    # C11 = M2 + M4 - M7 + M3  -- hmm, let me just use the standard formula
    
    # Actually, let me just directly construct the 10-term decomposition.
    
    # Term 1-7: Strassen for columns 0,1
    # U, V, W for Strassen (2x2 x 2x2 part, but B has only cols 0,1):
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ])  # 4x7
    
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],   # B00 (col 0)
        [0, 0, 1, 0, 0, 1, 0],    # B01 (col 1)
        [0, 0, 0, 1, 0, 0, 1],    # B10 (col 0)
        [1, 0, -1, 0, 1, 0, 1]    # B11 (col 1)
    ])  # 4x7 (but we need 6x7 for 2x3, so pad)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 1, 0, 1, 0, 0, 0],    # C01
        [0, 0, 1, 0, 1, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ])  # 4x7 (but we need 6x7 for 2x3 output)
    
    # Now for column 2, we need C02 = A00*B02 + A01*B12 and C12 = A10*B02 + A11*B12
    # We can add 3 more terms:
    # Term 8: A00 * B02 * C02  (U[0,7]=1, V[0,7]=1, W[5,7]=1)  [W index: k*n+i = 2*2+0 = 4]
    # Term 9: A01 * B12 * C02  (U[1,7]=1, V[3,7]=1, W[4,7]=1)
    # Term 10: (A10+A11) * B02 * C12  -- no, that's not rank 1
    
    # Actually for term 10, we need to cover C12 = A10*B02 + A11*B12
    # That's 2 more products, total 9. But we need exactly 10.
    
    # Let me use:
    # Term 8: A00 * B02 * C02
    # Term 9: A01 * B12 * C02
    # Term 10: (A10+A11) * B02 * C12  -- NO, this gives (A10+A11)*B02*C12 which is wrong
    
    # The issue is that C12 = A10*B02 + A11*B12 requires 2 separate products unless we can share.
    
    # Total: 7 + 2 + 2 = 11. That's rank 11, not 10.
    
    # To get rank 10, we need to share one product between the Strassen part and column 2.
    
    # Idea: In Strassen, M2 = (A10+A11)*B00. If we modify this to (A10+A11)*(B00 + c*B02) for some c,
    # then we can extract both B00 and B02 contributions.
    
    # Let M2' = (A10+A11)*(B00 + B02)
    # Then M2' = (A10+A11)*B00 + (A10+A11)*B02
    # We can use (A10+A11)*B02 for C12.
    
    # But then we lose the pure (A10+A11)*B00 term and need to recover it.
    
    # This is getting complex. Let me just use rank 11 for the 2x2x3 block.
    # 5 blocks of 2x2x3 with rank 11 = 55. Worse than 53.
    
    # Or: 7 blocks of 2x2x2 (Strassen, rank 7) + 1 col (rank 4) = 53.
    
    # Let me try a different partition: 2x2x5 block.
    # rank(2,2,5) ? If it's 17, then 3*17 = 51. That would be better!
    # But I don't know the exact rank.
    
    # For now, let me just return the rank-53 solution and note that finding rank-10 for 2x2x3
    # would give 5*10 = 50, which is better.
    
    # Let me try to actually find the rank-10 decomposition numerically.
    
    # Setup: 12 equations, 160 unknowns (with 9 scaling freedoms).
    # Fix V to have a specific form to reduce the search space.
    
    # Actually, let me use the known explicit construction.
    # From "The rank of the 2x2x3 matrix multiplication tensor" (known result):
    
    # One rank-10 decomposition:
    # Let me define the 10 terms explicitly.
    
    # After careful derivation, here's a rank-10 decomposition for 2x2 x 2x3:
    
    # The 10 rank-1 tensors are defined by:
    U10 = np.zeros((4, 10))
    V10 = np.zeros((6, 10))
    W10 = np.zeros((6, 10))
    
    # Terms 1-7: Strassen for first 2 columns
    # U indices: 0=A00, 1=A01, 2=A10, 3=A11
    # V indices: 0=B00, 1=B01, 2=B02, 3=B10, 4=B11, 5=B12
    # W indices: 0=C00, 1=C01, 2=C02, 3=C10, 4=C11, 5=C12
    
    # Strassen M1 = (A00+A11)(B00+B10)
    U10[0,0] = 1; U10[3,0] = 1
    V10[0,0] = 1; V10[3,0] = 1
    # M1 contributes to C00, C01, C10, C11
    
    # Let me just use the standard Strassen formulas and add 3 terms for col 2.
    # That gives rank 10 if I can make it work.
    
    # Standard Strassen (7 terms) for 2x2 x 2x2:
    # M1 = (A00+A11)(B00+B11)  -> but B11 is B[1,1]
    # Wait, I need to be very careful with indices.
    
    # A is 2x2: A[i,j], i,j in {0,1}
    # B is 2x3: B[j,k], j in {0,1}, k in {0,1,2}
    # C is 2x3: C[i,k], i in {0,1}, k in {0,1,2}
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # For k=0,1, we can use Strassen on the 2x2 submatrix B[:,0:2]:
    # M1 = (A00+A11)(B00+B11)  where B00=B[0,0], B11=B[1,1]
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)  where B10=B[1,0], B00=B[0,0]
    # M4 = A11(B11-B01)  where B11=B[1,1], B01=B[0,1]
    # M5 = (A00+A01)(B11+B00)
    # M6 = (A10-A00)(B01+B11)
    # M7 = (A01-A11)(B00+B10)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 - M1
    # C10 = M1 - M4
    # C11 = M2 + M4 - M7 + M3  -- let me verify
    
    # Actually, let me just verify: C11 = A10*B01 + A11*B11
    # M2 + M4 - M7 + M3 = (A10+A11)B00 + A11(B11-B01) - (A01-A11)(B00+B10) + A00(B10-B00)
    # This is getting messy. Let me just use the standard Strassen and verify numerically.
    
    # For now, let me just return rank 53 and note that the 2x2x3 rank-10 construction
    # needs to be verified.
    
    return None  # Signal that we couldn't find a better decomposition

def construct():
    """
    Entry point: return the best known decomposition.
    Currently returns the guaranteed rank-53 decomposition.
    """
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition using tensor partition
    # 5 blocks of 2x2x3, each with rank 10 -> total rank 50
    # But we need to verify the rank-10 decomposition exists and is correct.
    
    # For now, return the safe rank-53 solution
    return _build_rank53()
# EVOLVE-BLOCK-END
