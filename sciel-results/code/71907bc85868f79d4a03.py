import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_schoolbook_222(i_off, j_off, k_off, sign_u=1, sign_v=1, sign_w=1):
        nonlocal col
        # Adds rank-1 tensors for a 2x2x2 schoolbook multiplication
        # U indices: (i_off + li) * m + (j_off + lj)
        # V indices: (j_off + lj) * p + (k_off + lk)
        # W indices: (k_off + lk) * n + (i_off + li)
        
        for li in range(2):
            for lj in range(2):
                for lk in range(2):
                    # This is not schoolbook. Schoolbook is sum over j.
                    # C[i,k] = sum_j A[i,j] B[j,k]
                    # Rank 1 term: A[i,j] * B[j,k] * 1
                    # So for each (i,j,k) we have a term? No.
                    # Schoolbook for 2x2:
                    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
                    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
                    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
                    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
                    
                    # The rank-1 tensors are:
                    # (A[0,0], B[0,0], C[0,0]) -> U[0,0]=1, V[0,0]=1, W[0,0]=1
                    # (A[0,1], B[1,0], C[0,0]) -> U[0,1]=1, V[1,0]=1, W[0,0]=1
                    # (A[0,0], B[0,1], C[0,1]) -> U[0,0]=1, V[0,1]=1, W[0,1]=1
                    # (A[0,1], B[1,1], C[0,1]) -> U[0,1]=1, V[1,1]=1, W[0,1]=1
                    # ... etc.
                    
                    pass

    # Correct Schoolbook 2x2x2 decomposition
    def add_schoolbook_222(i_off, j_off, k_off):
        nonlocal col
        # Term 1: A[0,0] B[0,0] -> C[0,0]
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # Term 2: A[0,1] B[1,0] -> C[0,0]
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        # Term 3: A[0,0] B[0,1] -> C[0,1]
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # Term 4: A[0,1] B[1,1] -> C[0,1]
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        # Term 5: A[1,0] B[0,0] -> C[1,0]
        U[(i_off+1)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # Term 6: A[1,1] B[1,0] -> C[1,0]
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        # Term 7: A[1,0] B[0,1] -> C[1,1]
        U[(i_off+1)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        # Term 8: A[1,1] B[1,1] -> C[1,1]
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1

    # This adds 8 terms. But we want rank 4?
    # No, Schoolbook for 2x2 is rank 4?
    # C[0,0] = A00 B00 + A01 B10.
    # This is a sum of 2 rank-1 terms.
    # C has 4 entries.
    # Total rank is 4?
    # The tensor T_ijk for 2x2x2.
    # T_000 = 1, T_010 = 1, T_001 = 1, T_011 = 1 (for C00? No)
    # T_{ijk} = 1 if j is the inner index.
    # For 2x2, indices i,j,k in 0,1.
    # T_{000}=1, T_{010}=1 (A01 B10 contributes to C00? No, C00 = A00 B00 + A01 B10. So T_000 and T_010 contribute to C_00? No.
    # The tensor is T_{ijk}.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So T_{ijk} = 1 for all i,j,k.
    # Rank of this tensor is 4?
    # Yes, it is the identity tensor? No.
    # It is the matrix multiplication tensor.
    # Rank is 4 for 2x2? No, Strassen is 7. Schoolbook is 8?
    # Wait. Schoolbook uses 8 multiplications.
    # Rank 8.
    # Strassen uses 7.
    # So Schoolbook rank is 8.
    
    # My previous calculation of 28 was based on rank 4 per block.
    # But Schoolbook for 2x2 is rank 8.
    # So 7 blocks * 8 = 56.
    # This is worse than 44.
    
    # Can we use Strassen for the 2x2 blocks?
    # Rank 7 per block.
    # 7 blocks * 7 = 49.
    # Still worse than 44.
    
    # So the padding to 4x4 and using Strassen at the top level doesn't help if we use Strassen at the bottom level.
    # And if we use Schoolbook, it's 56.
    
    # What if we use Strassen for the top level, but optimize the sub-blocks?
    # The sub-blocks are 2x2.
    # The best rank for 2x2 is 7.
    # So 49 is the best we can get with this structure.
    
    # Current best is 44.
    # So we should stick with 44 or find a different structure.

    # Let's return the 44 rank solution as it is the best known.
    
    # Re-implement the 44 rank solution from the prompt.
    
    rank = 44
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = -1
        col += 1
        
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        U[(i_off+1)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = -1
        W[(k_off+0)*n + (i_off+0), col] = 1
        W[(k_off+0)*n + (i_off+1), col] = 1
        col += 1
        
        U[(i_off+0)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = -1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        U[(i_off+1)*m + (j_off+0), col] = 1
        U[(i_off+0)*m + (j_off+0), col] = -1
        V[(j_off+0)*p + (k_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+1), col] = 1
        col += 1
        
        U[(i_off+0)*m + (j_off+1), col] = 1
        U[(i_off+1)*m + (j_off+1), col] = -1
        V[(j_off+1)*p + (k_off+0), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1

    def add_122(i_off, j_off, k_off):
        nonlocal col
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+0), col] = 1
        W[(k_off+0)*n + (i_off+0), col] = 1
        col += 1
        
        U[(i_off+0)*m + (j_off+0), col] = 1
        V[(j_off+0)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1
        
        U[(i_off+0)*m + (j_off+1), col] = 1
        V[(j_off+1)*p + (k_off+1), col] = 1
        W[(k_off+1)*n + (i_off+0), col] = 1
        col += 1

    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    
    add_122(2, 0, 0)
    add_122(2, 2, 0)
    add_122(2, 0, 2)
    add_122(2, 2, 2)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
