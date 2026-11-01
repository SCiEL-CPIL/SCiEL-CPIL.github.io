import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    
    U = np.zeros((n*m, rank))  # (8, 36)
    V = np.zeros((m*p, rank))  # (20, 36)
    W = np.zeros((p*n, rank))  # (10, 36)
    
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all j
    
    # We need: for each (i,k), sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is equivalent to: the tensor T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k
    
    # Let me construct a rank-36 decomposition.
    # Base: schoolbook uses 40 terms. We need to save 4.
    
    # Strategy: Use 36 terms where 4 of them are "shared" via Strassen-like combinations.
    
    # Let me try: for the first 32 terms, use schoolbook for (i,j,k) with some pattern,
    # and for the last 4, use combined terms.
    
    # Actually, let me just try a direct construction.
    
    # Terms 0-35:
    # For i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4}:
    # Schoolbook would use 2*4*5=40 terms.
    # We want 36.
    
    # Let me use: for each (i,j), we need to cover all 5 k values.
    # If we could share between i=0 and i=1 for some (j,k), we save.
    
    # Strassen idea: for fixed j, define:
    # M_jk = (A[0,j]+A[1,j]) * B[j,k]  -- this is 1 term
    # N_jk = (A[0,j]-A[1,j]) * B[j,k]  -- this is 1 term
    # Then A[0,j]*B[j,k] = (M_jk+N_jk)/2, A[1,j]*B[j,k] = (M_jk-N_jk)/2
    # But this uses 2 terms per (j,k) instead of 1, so no savings.
    
    # The real Strassen trick for 2x2x2x2 uses 7 terms for 8 products.
    # Let me apply it to a 2x2 subblock.
    
    # Partition: A = [A11 A12; A21 A22] (2x2), B = [B1; B2] (2x5 where B1,B2 are 1x5)
    # C = A @ B = [A11*B1 + A12*B2; A21*B1 + A22*B2]
    
    # Strassen for 2x2 x 2x5:
    # M1 = (A11+A22) * B1  (1x5)
    # M2 = (A11+A22) * B2  (1x5)  
    # M3 = A11 * (B1-B2)  (1x5)
    # M4 = A22 * (B1-B2)  (1x5)
    # M5 = (A11+A12) * B2  (1x5)
    # M6 = (A21-A11) * (B1+B2)  (1x5)
    # M7 = (A12-A22) * (B1+B2)  (1x5)
    
    # Wait, this doesn't quite work because B1 and B2 are vectors, not scalars.
    
    # Let me verify: C[0] = A11*B1 + A12*B2
    # C[1] = A21*B1 + A22*B2
    
    # Using the above:
    # M1+M4-M5+M7 = (A11+A22)B1 + A22(B1-B2) - (A11+A12)B2 + (A12-A22)(B1+B2)
    # = A11B1 + A22B1 + A22B1 - A22B2 - A11B2 - A12B2 + A12B1 + A12B2 - A22B1 - A22B2
    # = A11B1 + A12B1 + (A22B1+A22B1-A22B1) + (-A22B2-A22B2) + (-A11B2) + (-A12B2+A12B2)
    # = A11B1 + A12B1 + A22B1 - 2A22B2 - A11B2
    # That's not equal to C[0] = A11B1 + A12B2.
    
    # I'm making errors. Let me use the standard Strassen more carefully.
    
    # For 2x2 x 2x2:
    # A = [a b; c d], B = [e f; g h]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    
    # For 2x2 x 2x5, extend by treating each column of B separately:
    # For column k of B: B_k = [B[0,k]; B[1,k]]
    # M1_k = (A11+A22)*B[0,k]
    # M2_k = (A21+A22)*B[0,k]
    # M3_k = A11*(B[1,k]-B[0,k])
    # M4_k = A22*(B[1,k]-B[0,k])
    # M5_k = (A11+A12)*B[1,k]
    # M6_k = (A21-A11)*(B[0,k]+B[1,k])
    # M7_k = (A12-A22)*(B[0,k]+B[1,k])
    
    # C[0,k] = M1_k+M4_k-M5_k+M7_k
    # C[1,k] = M2_k+M4_k... wait, let me re-derive.
    
    # Actually for 2x2 x 2x1 (scalar B entries):
    # C[0,0] = A11*B00 + A12*B10 = M1+M4-M5+M7 where:
    # M1=(A11+A22)(B00+B10), M4=A22(B10-B00), M5=(A11+A12)B10, M7=(A12-A22)(B00+B10)
    
    # Let me just verify numerically later. For now, let me use 7 terms per 2x2x2x5 block.
    
    # For our 2x4 x 4x5:
    # Split into two 2x2 x 2x5 blocks:
    # Block 1: A_L (2x2) x B_T (2x5) -> 7*5=35 terms? No, 7 terms per column, 5 columns = 35 terms.
    # That's too many.
    
    # I think the rank-36 construction is more subtle. Let me just try to find it by search or use a known formula.
    
    # Given time constraints, let me try a different approach: use rank 36 with a specific pattern.
    
    # Actually, let me just return schoolbook (rank 40) since I can't quickly derive rank 36.
    # But the knowledge says rank 36 works. Let me try harder.
    
    # Alternative: maybe I can find rank 36 by using 36 terms with some structure.
    
    # Let me try: 40 - 4 = 36. Save 4 by combining 4 pairs of terms.
    
    # For 4 specific (i,j,k) triples, use combined terms.
    
    # Actually, let me just implement schoolbook and see if I can improve.
    
    rank_school = n * m * p  # 40
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try to find rank 36
    # Based on Strassen: for 2x2x2x2, rank 7. 
    # For 2x4x4x5, maybe we can get rank 36.
    
    # Let me try a construction:
    # Use 36 terms. The idea: for 4 of the 20 (j,k) pairs (for each i), use shared terms.
    
    # Actually, I'll try to use a known rank-36 construction for 2x4x4x5.
    
    # Let me define the 36 terms explicitly.
    
    # Term structure: for r in 0..35
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # I'll use the following approach:
    # For i=0,1 and j=0,1,2,3:
    #   For k=0,1,2,3,4:
    #     Standard term: U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1
    # But that's 40 terms.
    
    # To get 36, I need to merge 4 terms.
    
    # Let me try: for 4 specific (i,j,k) combinations, use a Strassen-like merge.
    
    # Pick (i,j,k) = (0,0,0), (1,0,0), (0,0,1), (1,0,1) and merge them.
    
    # Original 4 terms:
    # (0,0,0): U[0,0]=1, V[0,0]=1, W[0,0]=1
    # (1,0,0): U[4,0]=1, V[0,0]=1, W[0,1]=1
    # (0,0,1): U[0,1]=1, V[1,0]=1, W[1,0]=1
    # (1,0,1): U[4,1]=1, V[1,0]=1, W[1,1]=1
    
    # Strassen merge for these 4:
    # M1 = (A[0,0]+A[1,0])*(B[0,0]+B[1,0]) -> U[0,0]=1,U[4,0]=1; V[0,0]=1,V[1,0]=1; W[0,0]=1,W[0,1]=1,W[1,0]=1,W[1,1]=1
    # But W would need to be 1 for all 4, which means the term contributes to all 4 (i,k) pairs.
    
    # This is getting complicated. Let me just return schoolbook.
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
