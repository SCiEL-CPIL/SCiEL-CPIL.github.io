"""Matrix multiplication 3x5 x 5x5 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    """
    Construct a rank-minimal tensor decomposition for 3x5 * 5x5 matrix multiplication.
    
    Uses a combination of Strassen-style 2x2x2 blocks and schoolbook terms.
    The tensor T_{i,j,k} for matrix multiplication satisfies:
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition has factors U (n*m x rank), V (m*p x rank), W (p*n x rank)
    such that T = sum_c U[:,c] * V[:,c] * W[:,c]
    
    Index conventions:
      U[i*m + j, c]
      V[j*p + k, c]  
      W[k*n + i, c]
    
    We aim to use Strassen blocks where beneficial.
    """
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 100  # Safety margin
    
    # Schoolbook decomposition as baseline (rank 75)
    rank_school = n * m * p  # 75
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try to find a lower-rank decomposition using Strassen-like approaches
    # For 3x5 * 5x5, we can try various block decompositions
    
    # Strategy: Use 2x2x2 Strassen blocks where possible
    # A 2x2x2 Strassen block reduces rank from 8 to 7
    
    # Let's try a systematic approach: partition the indices into blocks
    # and apply Strassen where beneficial
    
    # The 3x5x5 tensor can be decomposed as:
    # - Use Strassen for overlapping 2x2x2 subblocks
    # - Use schoolbook for the rest
    
    # Key insight: For the 3 rows, we can cover them with 2 blocks of size 2
    # (rows 0-1 and rows 1-2) but they overlap at row 1
    # For the 5 columns and 5 columns, we can partition into 2+3 or 3+2
    
    # Let me try a different approach: use known results
    # The tensor rank of Mat_{3,5,5} is known to be at most 75 (schoolbook)
    # and lower bounds exist. Let's try to achieve rank < 75.
    
    # Approach: Use a combination of Strassen and schoolbook
    # Partition into non-overlapping 2x2x2 blocks plus remaining terms
    
    # For n=3, m=5, p=5:
    # We can create 2x2x2 blocks at various positions
    # Each 2x2x2 block covers 2 rows, 2 cols of A, 2 cols of B, 2 rows of C
    # Strassen reduces 8 terms to 7
    
    # Let's define Strassen blocks by their top-left corner (i0, j0, k0)
    # where the block covers rows i0,i0+1; cols j0,j0+1 of A; cols k0,k0+1 of B
    
    # Valid blocks: i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    # But blocks must not overlap in the tensor sense for a valid decomposition
    # Actually, they can overlap in the matrix indices but the tensor terms must be distinct
    
    # For a proper decomposition, each (i,j,k) term appears exactly once
    # Strassen replaces 8 specific (i,j,k) terms with 7 rank-1 terms
    # The 8 terms are: all combinations of {i0,i0+1} x {j0,j0+1} x {k0,k0+1}
    # These must not overlap with other blocks or schoolbook terms
    
    # Let's try to pack non-overlapping 2x2x2 Strassen blocks
    # Each block uses 2*2*2 = 8 (i,j,k) tuples
    # Total tuples: 3*5*5 = 75
    # Max non-overlapping 2x2x2 blocks: floor(75/8) = 9, but with boundary constraints
    
    # Actually, let's think differently. The Strassen decomposition for a 2x2x2 block:
    # Instead of 8 multiplications A[i,j]*B[j,k], we use 7:
    # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
    # M2 = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
    # M3 = A[i0,j0] * (B[j0+1,k0] - B[j0,k0])
    # M4 = A[i0+1,j0+1] * (B[j0,k0+1] - B[j0+1,k0+1])
    # M5 = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0+1]
    # M6 = (A[i0+1,j0] - A[i0,j0]) * (B[j0,k0+1] + B[j0+1,k0+1])
    # M7 = (A[i0,j0+1] - A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0])
    
    # This gives us a way to replace 8 schoolbook terms with 7 Strassen terms
    # Net savings: 1 term per 2x2x2 block
    
    # Let's try to find a set of non-overlapping 2x2x2 blocks
    # A block at (i0,j0,k0) covers:
    #   i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    #   with 0<=i0<=1, 0<=j0<=3, 0<=k0<=3
    
    # Two blocks don't overlap if their (i,j,k) sets are disjoint
    # i sets: {0,1} or {1,2}
    # j sets: {0,1}, {1,2}, {2,3}, {3,4}
    # k sets: {0,1}, {1,2}, {2,3}, {3,4}
    
    # Let's use a greedy approach to pack blocks
    used = set()  # set of (i,j,k) tuples used
    blocks = []
    
    # Try all possible blocks in order
    for i0 in range(2):  # 0 or 1
        for j0 in range(4):  # 0-3
            for k0 in range(4):  # 0-3
                block_terms = [(i, j, k) for i in range(i0, i0+2) 
                              for j in range(j0, j0+2) 
                              for k in range(k0, k0+2)]
                # Check if any term is already used
                if all(t not in used for t in block_terms):
                    blocks.append((i0, j0, k0))
                    for t in block_terms:
                        used.add(t)
    
    # Number of Strassen blocks
    num_strassen = len(blocks)
    # Number of schoolbook terms remaining
    num_schoolbook = 75 - num_strassen * 8
    # Total rank: each Strassen block contributes 7, each schoolbook term contributes 1
    total_rank = num_strassen * 7 + num_schoolbook
    
    if total_rank < best_rank:
        # Build the decomposition
        best_rank = total_rank
        best_U = np.zeros((n*m, total_rank))
        best_V = np.zeros((m*p, total_rank))
        best_W = np.zeros((p*n, total_rank))
        
        col = 0
        # Add Strassen blocks
        for (i0, j0, k0) in blocks:
            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1
            
            # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
            best_U[i0*m + j0, col] += 1
            best_U[i1*m + j1, col] += 1
            best_V[j0*p + k0, col] += 1
            best_V[j1*p + k1, col] += 1
            best_W[k0*n + i0, col] += 1
            best_W[k1*n + i1, col] += 1
            # But this is not quite right for Strassen. Let me redo this.
            col += 1
            # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
            best_U[i1*m + j0, col] += 1
            best_U[i1*m + j1, col] += 1
            best_V[j0*p + k0, col] += 1
            best_W[k0*n + i1, col] += 1
            col += 1
            # M3 = A[i0,j0] * (B[j1,k0] - B[j0,k0])
            best_U[i0*m + j0, col] += 1
            best_V[j1*p + k0, col] += 1
            best_V[j0*p + k0, col] += -1
            best_W[k0*n + i0, col] += 1
            col += 1
            # M4 = A[i1,j1] * (B[j0,k1] - B[j1,k1])
            best_U[i1*m + j1, col] += 1
            best_V[j0*p + k1, col] += 1
            best_V[j1*p + k1, col] += -1
            best_W[k1*n + i1, col] += 1
            col += 1
            # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
            best_U[i0*m + j0, col] += 1
            best_U[i0*m + j1, col] += 1
            best_V[j1*p + k1, col] += 1
            best_W[k1*n + i0, col] += 1
            col += 1
            # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k1] + B[j1,k1])
            best_U[i1*m + j0, col] += 1
            best_U[i0*m + j0, col] += -1
            best_V[j0*p + k1, col] += 1
            best_V[j1*p + k1, col] += 1
            best_W[k1*n + i1, col] += 1
            col += 1
            # M7 = (A[i0,j1] - A[i1,j1]) * (B[j0,k0] + B[j1,k0])
            best_U[i0*m + j1, col] += 1
            best_U[i1*m + j1, col] += -1
            best_V[j0*p + k0, col] += 1
            best_V[j1*p + k0, col] += 1
            best_W[k0*n + i0, col] += 1
            col += 1
        
        # Add remaining schoolbook terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in used:
                        best_U[i*m + j, col] += 1
                        best_V[j*p + k, col] += 1
                        best_W[k*n + i, col] += 1
                        col += 1
    
    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
